import argparse
import json
import os
import time
import traceback

import pandas as pd

from data_prep import prepare_data
from evaluation import evaluate_single
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
        "llm_model": os.environ.get("MAKI_MODEL", "ministral-3-14b"),
        "llm_host": os.environ.get("MAKI_HOST", "https://maki.uni-mannheim.de/v1"),
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


if __name__ == "__main__":
    main()
