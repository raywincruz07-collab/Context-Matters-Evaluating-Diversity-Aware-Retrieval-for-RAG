import argparse
import json
import logging
import os
import subprocess
import time
import traceback

import pandas as pd

from data_prep import prepare_data
from evaluation import evaluate_single, retrieval_diversity
from retrievers.factory import get_retriever


def main():
    parser = argparse.ArgumentParser(
        description="Safely evaluate multiple retrievers on PubMedQA."
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of documents to retrieve."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit the number of questions evaluated.",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25", "dpr", "contriever", "colbertv2"],
        help="List of retrievers to evaluate.",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also evaluate LLM generation (requires API).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/raw",
        help="Directory to save raw results.",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Fixed generator config
    FIXED_GENERATION_CONFIG = {
        "llm_provider": "Mannheim Maki OpenAI-compatible API",
        "llm_model": os.environ.get("MAKI_MODEL", "qwen3.5-122b"),
        "llm_host": os.environ.get("MAKI_HOST", os.environ.get("MAKI_BASE_URL", "https://maki.uni-mannheim.de/v1")),
        "temperature": 0.0,
        "max_tokens": 512,
        "top_k": args.top_k,
    }

    # Write EXPERIMENT_METADATA.json
    metadata_path = os.path.join(args.results_dir, "EXPERIMENT_METADATA.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "retrievers": args.retrievers,
                "top_k": args.top_k,
                "limit": args.limit,
                "with_generation": args.with_generation,
                "generator_config": (
                    FIXED_GENERATION_CONFIG if args.with_generation else None
                ),
            },
            f,
            indent=4,
        )

    print("Loading data...")
    corpus, qa_pairs = prepare_data()

    if args.limit:
        qa_pairs = qa_pairs[: args.limit]
        print(f"Limited evaluation to first {args.limit} questions.")
    else:
        print(f"Evaluating on all {len(qa_pairs)} questions.")

    summary_data = []
    prefix = "fullrag_" if args.with_generation else "retrieval_"

    for retriever_name in args.retrievers:
        print(f"\n{'='*50}\nEvaluating retriever: {retriever_name}\n{'='*50}")
        csv_path = os.path.join(
            args.results_dir, f"{prefix}{retriever_name}_top{args.top_k}.csv"
        )

        # Load existing progress
        completed_qa_ids = set()
        results_rows = []
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                if "qa_id" in existing_df.columns:
                    completed_qa_ids = set(existing_df["qa_id"].tolist())
                    results_rows = existing_df.to_dict("records")
                    print(
                        f"Resuming {retriever_name}: found {len(completed_qa_ids)} completed questions."
                    )
            except Exception as e:
                print(
                    f"Warning: could not read existing progress for {retriever_name}: {e}"
                )

        try:
            # 1. Initialize and Index
            t0 = time.time()
            retriever = get_retriever(retriever_name)
            print(f"[{retriever_name}] Building/loading index...")
            retriever.index(corpus)
            index_time_sec = time.time() - t0
            print(
                f"[{retriever_name}] Indexing complete in {index_time_sec:.2f} seconds."
            )

            # 2. Generator setup if needed
            generator_obj = None
            if args.with_generation:
                from generator import MakiGenerator

                generator_obj = MakiGenerator(
                    model=FIXED_GENERATION_CONFIG["llm_model"],
                    host=FIXED_GENERATION_CONFIG["llm_host"],
                )

                if not generator_obj.is_available():
                    raise ValueError(
                        "MAKI_API_KEY is missing. Set it before running generation."
                    )

                print("Fixed generator initialized once.")
                print("Generator model:", generator_obj.model)
                print("Generator host:", generator_obj.host)

            # 3. Retrieve and Evaluate
            retrieval_times = []

            for i, qa in enumerate(qa_pairs):
                qa_id = qa.get("qa_id", i)
                if qa_id in completed_qa_ids:
                    continue

                if (i + 1) % 10 == 0 or i == 0:
                    print(
                        f"[{retriever_name}] Evaluating question {i + 1}/{len(qa_pairs)}..."
                    )

                # Row initialization
                row = {
                    "question_index": i,
                    "qa_id": qa_id,
                    "retriever": retriever_name,
                    "question": qa["question"],
                    "gold_doc_ids": str(qa.get("gold_doc_ids", [])),
                    "retrieved_doc_ids": "",
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "llm_model": (
                        FIXED_GENERATION_CONFIG["llm_model"]
                        if args.with_generation
                        else ""
                    ),
                    "temperature": (
                        FIXED_GENERATION_CONFIG["temperature"]
                        if args.with_generation
                        else ""
                    ),
                    "max_tokens": (
                        FIXED_GENERATION_CONFIG["max_tokens"]
                        if args.with_generation
                        else ""
                    ),
                    "prediction": "",
                    "ground_truth": qa.get("long_answer", ""),
                    "exact_match": 0.0,
                    "f1": 0.0,
                    "rouge_l": 0.0,
                    "row_status": "OK",
                    "row_error": "",
                }

                try:
                    # Retrieval
                    t1 = time.time()
                    retrieved = retriever.retrieve(qa["question"], top_k=args.top_k)
                    retrieval_times.append(time.time() - t1)

                    retrieved_doc_ids = [doc.get("doc_id", "") for doc, _ in retrieved]
                    gold_doc_ids = qa.get("gold_doc_ids", [])
                    row["retrieved_doc_ids"] = str(retrieved_doc_ids)

                    # Generation
                    if args.with_generation and generator_obj:
                        context_docs = [doc for doc, _ in retrieved]
                        prediction = generator_obj.generate(
                            qa["question"],
                            context_docs,
                            temperature=FIXED_GENERATION_CONFIG["temperature"],
                            max_tokens=FIXED_GENERATION_CONFIG["max_tokens"],
                        )
                        row["prediction"] = prediction

                    # Compute metrics
                    metrics = evaluate_single(
                        prediction=row["prediction"],
                        ground_truth=row["ground_truth"],
                        retrieved_doc_ids=retrieved_doc_ids,
                        gold_doc_ids=gold_doc_ids,
                    )

                    row["recall_at_k"] = metrics.get("recall_at_k", 0.0)
                    row["mrr"] = metrics.get("mrr", 0.0)
                    if args.with_generation:
                        row["exact_match"] = metrics.get("exact_match", 0.0)
                        row["f1"] = metrics.get("f1", 0.0)
                        row["rouge_l"] = metrics.get("rouge_l", 0.0)

                except Exception as e:
                    row["row_status"] = "ERROR"
                    row["row_error"] = str(e)
                    print(f"Error on question {i}: {e}")

                results_rows.append(row)
                completed_qa_ids.add(qa_id)

                # Save intermediate detailed results
                df_res = pd.DataFrame(results_rows)
                df_res.to_csv(csv_path, index=False)

            # Compute summary stats from completed rows
            df_final = pd.DataFrame(results_rows)
            valid_rows = df_final[df_final["row_status"] == "OK"]

            avg_ret_time = (
                sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
            )
            avg_recall = valid_rows["recall_at_k"].mean() if not valid_rows.empty else 0
            avg_mrr = valid_rows["mrr"].mean() if not valid_rows.empty else 0

            summary_row = {
                "retriever": retriever_name,
                "num_questions": len(qa_pairs),
                "top_k": args.top_k,
                "index_time_sec": index_time_sec,
                "avg_retrieval_time_sec": avg_ret_time,
                "avg_recall_at_k": avg_recall,
                "avg_mrr": avg_mrr,
                "status": "OK",
                "error": "",
            }
            if args.with_generation:
                summary_row["avg_em"] = (
                    valid_rows["exact_match"].mean() if not valid_rows.empty else 0
                )
                summary_row["avg_f1"] = (
                    valid_rows["f1"].mean() if not valid_rows.empty else 0
                )
                summary_row["avg_rouge_l"] = (
                    valid_rows["rouge_l"].mean() if not valid_rows.empty else 0
                )

            summary_data.append(summary_row)

        except Exception as e:
            print(f"[{retriever_name}] Error during evaluation: {e}")
            traceback.print_exc()
            summary_row = {
                "retriever": retriever_name,
                "num_questions": len(qa_pairs),
                "top_k": args.top_k,
                "index_time_sec": 0,
                "avg_retrieval_time_sec": 0,
                "avg_recall_at_k": 0,
                "avg_mrr": 0,
                "status": "ERROR",
                "error": str(e),
            }
            summary_data.append(summary_row)

    # Final Summary
    df_sum = pd.DataFrame(summary_data)
    sum_csv_path = os.path.join(
        args.results_dir, f"{prefix}summary_top{args.top_k}.csv"
    )
    df_sum.to_csv(sum_csv_path, index=False)
    print(f"\nSaved final summary to {sum_csv_path}")

    # Ranked Retriever Comparison
    valid_res = [r for r in summary_data if r["status"] == "OK"]
    if valid_res:
        # Sort by: highest recall, then highest MRR
        valid_res.sort(key=lambda x: (x["avg_recall_at_k"], x["avg_mrr"]), reverse=True)
        ranked_path = os.path.join(
            args.results_dir, f"{prefix}ranked_retriever_comparison_top{args.top_k}.csv"
        )
        pd.DataFrame(valid_res).to_csv(ranked_path, index=False)
        print(f"Saved ranked comparison to {ranked_path}")

        best = valid_res[0]
        print("\n🏆 Best Retriever:")
        print(f"  Name: {best['retriever']}")
        print(f"  Recall@{args.top_k}: {best['avg_recall_at_k']:.4f}")
        print(f"  MRR: {best['avg_mrr']:.4f}")
    else:
        print("\nNo retrievers completed successfully to determine the best.")


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def main_sprint2():
    """Sprint 2 evaluation: 4 retrievers × 6 MMR conditions = 24 combinations on HotpotQA."""
    parser = argparse.ArgumentParser(
        description="Sprint 2: evaluate retrievers + MMR on HotpotQA."
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also evaluate LLM generation and faithfulness_nli (requires API + GPU).",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/raw",
        help="Directory for output CSVs and metadata.",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25", "dpr", "contriever", "colbertv2"],
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # --- error log (append mode so partial reruns accumulate) ---------------
    error_log_path = os.path.join(args.results_dir, "sprint2_errors.log")
    logging.basicConfig(
        filename=error_log_path,
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    error_logger = logging.getLogger("sprint2")

    # --- fixed generator config (same as Sprint 1) --------------------------
    FIXED_GENERATION_CONFIG = {
        "llm_provider": "Mannheim Maki OpenAI-compatible API",
        "llm_model": os.environ.get("MAKI_MODEL", "qwen3.5-122b"),
        "llm_host": os.environ.get("MAKI_HOST", os.environ.get("MAKI_BASE_URL", "https://maki.uni-mannheim.de/v1")),
        "temperature": 0.0,
        "max_tokens": 512,
        "top_k": args.top_k,
    }

    from config import (
        HOTPOT_SAMPLE_SIZE,
        HOTPOT_SEED,
        MMR_CANDIDATE_POOL,
        MMR_LAMBDAS,
        NLI_MODEL,
    )

    conditions = ["none"] + [f"mmr_{lam}" for lam in MMR_LAMBDAS]
    print(f"[seed={HOTPOT_SEED}] Sprint 2: {len(args.retrievers)} retrievers × {len(conditions)} conditions = {len(args.retrievers)*len(conditions)} combinations")
    print(f"Conditions : {conditions}")
    print(f"Retrievers : {args.retrievers}")

    # --- load HotpotQA data -------------------------------------------------
    from data_prep_hotpot import prepare_hotpot_data
    hotpot_corpus, hotpot_qa = prepare_hotpot_data()
    print(f"HotpotQA corpus: {len(hotpot_corpus)} docs | QA pairs: {len(hotpot_qa)}")

    # --- pre-compute Contriever corpus embeddings (for MMR + diversity) ------
    from diversification.mmr import get_hotpot_corpus_embeddings
    from retrievers.dense_retriever import ContrieverRetriever

    print("Loading Contriever for MMR diversity embedding ...")
    contriever_for_mmr = ContrieverRetriever()
    contriever_for_mmr._load_model()
    corpus_embs = get_hotpot_corpus_embeddings(hotpot_corpus, contriever_for_mmr._encode)
    doc_id_to_emb: dict = {
        doc["doc_id"]: corpus_embs[i] for i, doc in enumerate(hotpot_corpus)
    }

    # --- optional NLI model -------------------------------------------------
    nli_pipeline = None
    if args.with_generation:
        try:
            from transformers import pipeline as hf_pipeline
            print(f"Loading NLI model: {NLI_MODEL} ...")
            nli_pipeline = hf_pipeline(
                "zero-shot-classification",
                model=NLI_MODEL,
                device=0,  # GPU 0 on Colab
            )
            print("NLI model ready.")
        except Exception as exc:
            print(f"Warning: could not load NLI model ({exc}). faithfulness_nli will be skipped.")
            error_logger.error("NLI model load failed: %s", traceback.format_exc())

    # --- generator (reuse Sprint 1 pattern) ----------------------------------
    generator_obj = None
    if args.with_generation:
        from generator import MakiGenerator
        generator_obj = MakiGenerator(
            model=FIXED_GENERATION_CONFIG["llm_model"],
            host=FIXED_GENERATION_CONFIG["llm_host"],
        )
        if not generator_obj.is_available():
            raise ValueError("MAKI_API_KEY is missing. Set it before running generation.")
        print("Generator ready:", generator_obj.model)

    total_errors = 0
    summary_data = []

    for retriever_name in args.retrievers:
        print(f"\n{'='*60}\nRetriever: {retriever_name}\n{'='*60}")

        try:
            retriever = get_retriever(retriever_name)
            retriever.index(hotpot_corpus)
        except Exception as exc:
            msg = f"[{retriever_name}] Indexing failed: {exc}"
            print(msg)
            error_logger.error(msg + "\n" + traceback.format_exc())
            total_errors += 1
            for cond in conditions:
                summary_data.append({
                    "retriever": retriever_name,
                    "condition": cond,
                    "status": "ERROR_INDEX",
                    "error": str(exc),
                })
            continue

        for condition in conditions:
            print(f"\n  [{retriever_name}] condition={condition}")

            csv_path = os.path.join(
                args.results_dir,
                f"sprint2_{retriever_name}_{condition}_top{args.top_k}.csv",
            )

            # Checkpoint: load existing progress
            completed_qa_ids: set = set()
            results_rows: list = []
            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path)
                    if "qa_id" in existing_df.columns:
                        if args.with_generation:
                            # Only count rows as "done" if they actually have a
                            # non-null prediction. Retrieval-only checkpoints
                            # (from a run without --with-generation) must NOT
                            # be treated as complete when generation is requested.
                            has_prediction = existing_df["prediction"].notna() if "prediction" in existing_df.columns else pd.Series([False] * len(existing_df))
                            done_df = existing_df[has_prediction]
                            completed_qa_ids = set(done_df["qa_id"].tolist())
                            results_rows = done_df.to_dict("records")
                            skipped = len(existing_df) - len(done_df)
                            print(f"    Resuming: {len(completed_qa_ids)} rows already done (generation).")
                            if skipped > 0:
                                print(f"    Note: {skipped} retrieval-only rows found without generation — will be regenerated.")
                        else:
                            completed_qa_ids = set(existing_df["qa_id"].tolist())
                            results_rows = existing_df.to_dict("records")
                            print(f"    Resuming: {len(completed_qa_ids)} rows already done.")
                except Exception as exc:
                    print(f"    Warning: could not read checkpoint ({exc}), starting fresh.")

            # Parse MMR lambda if applicable
            use_mmr = condition != "none"
            mmr_lambda = float(condition.split("_", 1)[1]) if use_mmr else None

            from diversification.mmr import mmr_rerank

            retrieval_times: list = []
            condition_errors = 0

            for i, qa in enumerate(hotpot_qa):
                qa_id = qa.get("qa_id", i)
                if qa_id in completed_qa_ids:
                    continue

                row: dict = {
                    "question_index": i,
                    "qa_id": qa_id,
                    "retriever": retriever_name,
                    "condition": condition,
                    "question": qa["question"],
                    "gold_doc_ids": str(qa.get("gold_doc_ids", [])),
                    "retrieved_doc_ids": "",
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "retrieval_diversity": 0.0,
                    "faithfulness_nli": 0.0,
                    "llm_model": FIXED_GENERATION_CONFIG["llm_model"] if args.with_generation else "",
                    "prediction": "",
                    "ground_truth": qa.get("long_answer", ""),
                    "exact_match": 0.0,
                    "f1": 0.0,
                    "rouge_l": 0.0,
                    "row_status": "OK",
                    "row_error": "",
                }

                try:
                    t1 = time.time()
                    if use_mmr:
                        candidates = retriever.retrieve(qa["question"], top_k=MMR_CANDIDATE_POOL)
                        retrieved = mmr_rerank(
                            qa["question"],
                            candidates,
                            top_k=args.top_k,
                            lambda_param=mmr_lambda,
                            embed_fn=contriever_for_mmr._encode,
                            precomputed_embs=doc_id_to_emb,
                        )
                    else:
                        retrieved = retriever.retrieve(qa["question"], top_k=args.top_k)
                    retrieval_times.append(time.time() - t1)

                    retrieved_doc_ids = [doc.get("doc_id") for doc, _ in retrieved]
                    gold_doc_ids = qa.get("gold_doc_ids", [])
                    row["retrieved_doc_ids"] = str(retrieved_doc_ids)

                    # Retrieval metrics
                    from evaluation import mrr as compute_mrr, retrieval_recall_at_k
                    row["recall_at_k"] = retrieval_recall_at_k(retrieved_doc_ids, gold_doc_ids)
                    row["mrr"] = compute_mrr(retrieved_doc_ids, gold_doc_ids)

                    # Retrieval diversity (Sprint 2 metric)
                    retrieved_embs = [
                        doc_id_to_emb[did]
                        for did in retrieved_doc_ids
                        if did in doc_id_to_emb
                    ]
                    row["retrieval_diversity"] = retrieval_diversity(retrieved_embs)

                    # Generation + NLI faithfulness
                    if args.with_generation and generator_obj:
                        context_docs = [doc for doc, _ in retrieved]
                        prediction = generator_obj.generate(
                            qa["question"],
                            context_docs,
                            temperature=FIXED_GENERATION_CONFIG["temperature"],
                            max_tokens=FIXED_GENERATION_CONFIG["max_tokens"],
                        )
                        row["prediction"] = prediction
                        if prediction.startswith("Error:"):
                            raise RuntimeError(
                                f"Generator returned an error string: {prediction}"
                            )

                        from evaluation import exact_match, rouge_l, token_f1
                        row["exact_match"] = exact_match(prediction, row["ground_truth"])
                        row["f1"] = token_f1(prediction, row["ground_truth"])
                        row["rouge_l"] = rouge_l(prediction, row["ground_truth"])

                        if nli_pipeline is not None:
                            from evaluation import faithfulness_nli
                            context_texts = [doc.get("text", "") for doc in context_docs]
                            row["faithfulness_nli"] = faithfulness_nli(
                                prediction, context_texts, nli_pipeline
                            )

                except Exception as exc:
                    row["row_status"] = "ERROR"
                    row["row_error"] = str(exc)
                    condition_errors += 1
                    total_errors += 1
                    error_logger.error(
                        "qa_id=%s retriever=%s condition=%s\n%s",
                        qa_id,
                        retriever_name,
                        condition,
                        traceback.format_exc(),
                    )
                    if (i + 1) % 10 == 0:
                        print(f"    Error on qa_id={qa_id}: {exc}")

                results_rows.append(row)
                completed_qa_ids.add(qa_id)

                # Per-question checkpoint (same pattern as Sprint 1)
                pd.DataFrame(results_rows).to_csv(csv_path, index=False)

            df_final = pd.DataFrame(results_rows)
            valid_rows = df_final[df_final["row_status"] == "OK"]
            avg_ret_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0

            summary_data.append({
                "retriever": retriever_name,
                "condition": condition,
                "num_questions": len(hotpot_qa),
                "top_k": args.top_k,
                "avg_retrieval_time_sec": avg_ret_time,
                "avg_recall_at_k": valid_rows["recall_at_k"].mean() if not valid_rows.empty else 0.0,
                "avg_mrr": valid_rows["mrr"].mean() if not valid_rows.empty else 0.0,
                "avg_retrieval_diversity": valid_rows["retrieval_diversity"].mean() if not valid_rows.empty else 0.0,
                "avg_faithfulness_nli": valid_rows["faithfulness_nli"].mean() if not valid_rows.empty else 0.0,
                "error_count": condition_errors,
                "status": "OK" if condition_errors == 0 else "PARTIAL",
            })

            print(f"    Done. recall@{args.top_k}={summary_data[-1]['avg_recall_at_k']:.4f}  errors={condition_errors}")

    # --- summary CSV --------------------------------------------------------
    df_sum = pd.DataFrame(summary_data)
    sum_path = os.path.join(args.results_dir, f"sprint2_summary_top{args.top_k}.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"\nSaved summary to {sum_path}")

    # --- EXPERIMENT_METADATA_sprint2.json -----------------------------------
    meta = {
        "sprint": 2,
        "dataset": "HotpotQA distractor",
        "retrievers": args.retrievers,
        "conditions": conditions,
        "sample_size": HOTPOT_SAMPLE_SIZE,
        "seed": HOTPOT_SEED,
        "top_k": args.top_k,
        "mmr_candidate_pool": MMR_CANDIDATE_POOL,
        "mmr_lambdas": MMR_LAMBDAS,
        "with_generation": args.with_generation,
        "generator_config": FIXED_GENERATION_CONFIG if args.with_generation else None,
        "true_error_count": total_errors,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit_hash(),
    }
    meta_path = os.path.join(args.results_dir, "EXPERIMENT_METADATA_sprint2.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Saved metadata to {meta_path}")
    print(f"\nSprint 2 complete. Total errors: {total_errors}  (see {error_log_path})")


def main_sprint2_beir():
    """Sprint 2 BEIR evaluation: 4 retrievers × 6 MMR conditions on BEIR HotpotQA.

    Uses Option B corpus (gold passages + 50 random negatives per query, seed=42).
    Since BEIR HotpotQA qrels do not include free-text answers, exact_match / f1 /
    rouge_l are fixed to 0.0 and never computed — only recall@k, mrr,
    retrieval_diversity, and faithfulness_nli are meaningful here.
    """
    parser = argparse.ArgumentParser(
        description="Sprint 2 BEIR: evaluate retrievers + MMR on BEIR HotpotQA."
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also evaluate LLM generation and faithfulness_nli (requires API + GPU).",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/raw",
        help="Directory for output CSVs and metadata.",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25", "dpr", "contriever", "colbertv2"],
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    error_log_path = os.path.join(args.results_dir, "sprint2_beir_errors.log")
    logging.basicConfig(
        filename=error_log_path,
        filemode="a",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    error_logger = logging.getLogger("sprint2_beir")

    FIXED_GENERATION_CONFIG = {
        "llm_provider": "Mannheim Maki OpenAI-compatible API",
        "llm_model": os.environ.get("MAKI_MODEL", "qwen3.5-122b"),
        "llm_host": os.environ.get("MAKI_HOST", os.environ.get("MAKI_BASE_URL", "https://maki.uni-mannheim.de/v1")),
        "temperature": 0.0,
        "max_tokens": 512,
        "top_k": args.top_k,
    }

    from config import (
        HOTPOT_BEIR_NEGATIVES_PER_QUERY,
        HOTPOT_BEIR_SAMPLE_SIZE,
        HOTPOT_BEIR_SEED,
        MMR_CANDIDATE_POOL,
        MMR_LAMBDAS,
        NLI_MODEL,
    )

    conditions = ["none"] + [f"mmr_{lam}" for lam in MMR_LAMBDAS]
    print(
        f"[seed={HOTPOT_BEIR_SEED}] Sprint 2 BEIR: "
        f"{len(args.retrievers)} retrievers × {len(conditions)} conditions = "
        f"{len(args.retrievers) * len(conditions)} combinations"
    )
    print(f"Conditions : {conditions}")
    print(f"Retrievers : {args.retrievers}")

    # Load BEIR corpus and qa_pairs
    from data_prep_hotpot_beir import prepare_hotpot_beir_data
    beir_corpus, beir_qa = prepare_hotpot_beir_data()
    print(f"BEIR corpus: {len(beir_corpus)} docs | QA pairs: {len(beir_qa)}")

    # Pre-compute Contriever embeddings for MMR and diversity metric
    from diversification.mmr import get_hotpot_corpus_embeddings
    from retrievers.dense_retriever import ContrieverRetriever

    print("Loading Contriever for MMR diversity embedding …")
    contriever_for_mmr = ContrieverRetriever()
    contriever_for_mmr._load_model()
    corpus_embs = get_hotpot_corpus_embeddings(beir_corpus, contriever_for_mmr._encode)
    doc_id_to_emb: dict = {
        doc["doc_id"]: corpus_embs[i] for i, doc in enumerate(beir_corpus)
    }

    # Optional NLI model for faithfulness
    nli_pipeline = None
    if args.with_generation:
        try:
            from transformers import pipeline as hf_pipeline
            print(f"Loading NLI model: {NLI_MODEL} …")
            nli_pipeline = hf_pipeline(
                "zero-shot-classification",
                model=NLI_MODEL,
                device=0,
            )
            print("NLI model ready.")
        except Exception as exc:
            print(f"Warning: could not load NLI model ({exc}). faithfulness_nli skipped.")
            error_logger.error("NLI model load failed: %s", traceback.format_exc())

    generator_obj = None
    if args.with_generation:
        from generator import MakiGenerator
        generator_obj = MakiGenerator(
            model=FIXED_GENERATION_CONFIG["llm_model"],
            host=FIXED_GENERATION_CONFIG["llm_host"],
        )
        if not generator_obj.is_available():
            raise ValueError("MAKI_API_KEY is missing. Set it before running generation.")
        print("Generator ready:", generator_obj.model)

    total_errors = 0
    summary_data = []

    for retriever_name in args.retrievers:
        print(f"\n{'='*60}\nRetriever: {retriever_name}\n{'='*60}")

        try:
            retriever = get_retriever(retriever_name)
            retriever.index(beir_corpus)
        except Exception as exc:
            msg = f"[{retriever_name}] Indexing failed: {exc}"
            print(msg)
            error_logger.error(msg + "\n" + traceback.format_exc())
            total_errors += 1
            for cond in conditions:
                summary_data.append({
                    "retriever": retriever_name,
                    "condition": cond,
                    "status": "ERROR_INDEX",
                    "error": str(exc),
                })
            continue

        for condition in conditions:
            print(f"\n  [{retriever_name}] condition={condition}")

            csv_path = os.path.join(
                args.results_dir,
                f"sprint2_beir_{retriever_name}_{condition}_top{args.top_k}.csv",
            )

            completed_qa_ids: set = set()
            results_rows: list = []
            if os.path.exists(csv_path):
                try:
                    existing_df = pd.read_csv(csv_path)
                    if "qa_id" in existing_df.columns:
                        if args.with_generation:
                            # Only count rows as "done" if they actually have a
                            # non-null prediction. Retrieval-only checkpoints
                            # (from a run without --with-generation) must NOT
                            # be treated as complete when generation is requested.
                            has_prediction = existing_df["prediction"].notna() if "prediction" in existing_df.columns else pd.Series([False] * len(existing_df))
                            done_df = existing_df[has_prediction]
                            completed_qa_ids = set(done_df["qa_id"].tolist())
                            results_rows = done_df.to_dict("records")
                            skipped = len(existing_df) - len(done_df)
                            print(f"    Resuming: {len(completed_qa_ids)} rows already done (generation).")
                            if skipped > 0:
                                print(f"    Note: {skipped} retrieval-only rows found without generation — will be regenerated.")
                        else:
                            completed_qa_ids = set(existing_df["qa_id"].tolist())
                            results_rows = existing_df.to_dict("records")
                            print(f"    Resuming: {len(completed_qa_ids)} rows already done.")
                except Exception as exc:
                    print(f"    Warning: could not read checkpoint ({exc}), starting fresh.")

            use_mmr = condition != "none"
            mmr_lambda = float(condition.split("_", 1)[1]) if use_mmr else None

            from diversification.mmr import mmr_rerank

            retrieval_times: list = []
            condition_errors = 0

            for i, qa in enumerate(beir_qa):
                qa_id = qa.get("qa_id", i)
                if qa_id in completed_qa_ids:
                    continue

                row: dict = {
                    "question_index": i,
                    "qa_id": qa_id,
                    "retriever": retriever_name,
                    "condition": condition,
                    "question": qa["question"],
                    "beir_query_id": qa.get("beir_query_id", ""),
                    "gold_doc_ids": str(qa.get("gold_doc_ids", [])),
                    "retrieved_doc_ids": "",
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "retrieval_diversity": 0.0,
                    "faithfulness_nli": 0.0,
                    "llm_model": FIXED_GENERATION_CONFIG["llm_model"] if args.with_generation else "",
                    "prediction": "",
                    # answer is always "" for BEIR HotpotQA — do not compute
                    # exact_match / f1 / rouge_l against this empty reference
                    "exact_match": 0.0,
                    "f1": 0.0,
                    "rouge_l": 0.0,
                    "row_status": "OK",
                    "row_error": "",
                }

                try:
                    t1 = time.time()
                    if use_mmr:
                        candidates = retriever.retrieve(qa["question"], top_k=MMR_CANDIDATE_POOL)
                        retrieved = mmr_rerank(
                            qa["question"],
                            candidates,
                            top_k=args.top_k,
                            lambda_param=mmr_lambda,
                            embed_fn=contriever_for_mmr._encode,
                            precomputed_embs=doc_id_to_emb,
                        )
                    else:
                        retrieved = retriever.retrieve(qa["question"], top_k=args.top_k)
                    retrieval_times.append(time.time() - t1)

                    retrieved_doc_ids = [doc.get("doc_id") for doc, _ in retrieved]
                    gold_doc_ids = qa.get("gold_doc_ids", [])
                    row["retrieved_doc_ids"] = str(retrieved_doc_ids)

                    from evaluation import mrr as compute_mrr, retrieval_recall_at_k
                    row["recall_at_k"] = retrieval_recall_at_k(retrieved_doc_ids, gold_doc_ids)
                    row["mrr"] = compute_mrr(retrieved_doc_ids, gold_doc_ids)

                    retrieved_embs = [
                        doc_id_to_emb[did]
                        for did in retrieved_doc_ids
                        if did in doc_id_to_emb
                    ]
                    row["retrieval_diversity"] = retrieval_diversity(retrieved_embs)

                    # Generation + NLI faithfulness (answer field is empty — skip lexical metrics)
                    if args.with_generation and generator_obj:
                        context_docs = [doc for doc, _ in retrieved]
                        prediction = generator_obj.generate(
                            qa["question"],
                            context_docs,
                            temperature=FIXED_GENERATION_CONFIG["temperature"],
                            max_tokens=FIXED_GENERATION_CONFIG["max_tokens"],
                        )
                        row["prediction"] = prediction
                        if prediction.startswith("Error:"):
                            raise RuntimeError(
                                f"Generator returned an error string: {prediction}"
                            )
                        # exact_match / f1 / rouge_l intentionally left as 0.0 —
                        # BEIR HotpotQA has no free-text reference answers.

                        if nli_pipeline is not None:
                            from evaluation import faithfulness_nli
                            context_texts = [doc.get("text", "") for doc in context_docs]
                            row["faithfulness_nli"] = faithfulness_nli(
                                prediction, context_texts, nli_pipeline
                            )

                except Exception as exc:
                    row["row_status"] = "ERROR"
                    row["row_error"] = str(exc)
                    condition_errors += 1
                    total_errors += 1
                    error_logger.error(
                        "qa_id=%s retriever=%s condition=%s\n%s",
                        qa_id, retriever_name, condition,
                        traceback.format_exc(),
                    )
                    if (i + 1) % 10 == 0:
                        print(f"    Error on qa_id={qa_id}: {exc}")

                results_rows.append(row)
                completed_qa_ids.add(qa_id)
                pd.DataFrame(results_rows).to_csv(csv_path, index=False)

            df_final = pd.DataFrame(results_rows)
            valid_rows = df_final[df_final["row_status"] == "OK"]
            avg_ret_time = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0

            summary_data.append({
                "retriever": retriever_name,
                "condition": condition,
                "num_questions": len(beir_qa),
                "top_k": args.top_k,
                "avg_retrieval_time_sec": avg_ret_time,
                "avg_recall_at_k": valid_rows["recall_at_k"].mean() if not valid_rows.empty else 0.0,
                "avg_mrr": valid_rows["mrr"].mean() if not valid_rows.empty else 0.0,
                "avg_retrieval_diversity": valid_rows["retrieval_diversity"].mean() if not valid_rows.empty else 0.0,
                "avg_faithfulness_nli": valid_rows["faithfulness_nli"].mean() if not valid_rows.empty else 0.0,
                "error_count": condition_errors,
                "status": "OK" if condition_errors == 0 else "PARTIAL",
            })
            print(
                f"    Done. recall@{args.top_k}={summary_data[-1]['avg_recall_at_k']:.4f}  "
                f"errors={condition_errors}"
            )

    df_sum = pd.DataFrame(summary_data)
    sum_path = os.path.join(args.results_dir, f"sprint2_beir_summary_top{args.top_k}.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"\nSaved summary to {sum_path}")

    meta = {
        "sprint": 2,
        "dataset": "BEIR HotpotQA (Option B subsampling)",
        "retrievers": args.retrievers,
        "conditions": conditions,
        "sample_size": HOTPOT_BEIR_SAMPLE_SIZE,
        "seed": HOTPOT_BEIR_SEED,
        "negatives_per_query": HOTPOT_BEIR_NEGATIVES_PER_QUERY,
        "top_k": args.top_k,
        "mmr_candidate_pool": MMR_CANDIDATE_POOL,
        "mmr_lambdas": MMR_LAMBDAS,
        "with_generation": args.with_generation,
        "generator_config": FIXED_GENERATION_CONFIG if args.with_generation else None,
        "true_error_count": total_errors,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit_hash(),
    }
    meta_path = os.path.join(args.results_dir, "EXPERIMENT_METADATA_sprint2_beir.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"Saved metadata to {meta_path}")
    print(f"\nSprint 2 BEIR complete. Total errors: {total_errors}  (see {error_log_path})")


if __name__ == "__main__":
    import sys
    if "--beir" in sys.argv:
        sys.argv.remove("--beir")
        main_sprint2_beir()
    elif "--sprint2" in sys.argv:
        sys.argv.remove("--sprint2")
        main_sprint2()
    else:
        main()
