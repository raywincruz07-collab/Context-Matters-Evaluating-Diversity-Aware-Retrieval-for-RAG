import argparse
import os
import time
import traceback

import pandas as pd

from data_prep import prepare_data
from evaluation import evaluate_single
from retrievers.factory import get_retriever


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple retrievers on PubMedQA."
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
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

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
            generator = None
            if args.with_generation:
                from generator import MakiGenerator

                # Note: MakiGenerator will use environment variables for API keys and host
                generator = MakiGenerator()

            # 3. Retrieve and Evaluate
            retrieval_times = []
            all_metrics = []
            results_rows = []

            for i, qa in enumerate(qa_pairs):
                if (i + 1) % 10 == 0 or i == 0:
                    print(
                        f"[{retriever_name}] Evaluating question {i + 1}/{len(qa_pairs)}..."
                    )

                # Retrieval
                t1 = time.time()
                retrieved = retriever.retrieve(qa["question"], top_k=args.top_k)
                retrieval_times.append(time.time() - t1)

                retrieved_doc_ids = [doc["doc_id"] for doc, _ in retrieved]
                gold_doc_ids = qa["gold_doc_ids"]

                # Generation
                if args.with_generation and generator:
                    context_docs = [doc for doc, _ in retrieved]
                    prediction = generator.generate(
                        qa["question"], context_docs, temperature=0.0
                    )
                else:
                    prediction = ""

                ground_truth = qa["long_answer"]

                # Compute metrics
                metrics = evaluate_single(
                    prediction=prediction,
                    ground_truth=ground_truth,
                    retrieved_doc_ids=retrieved_doc_ids,
                    gold_doc_ids=gold_doc_ids,
                )
                all_metrics.append(metrics)

                row = {
                    "qa_id": qa.get("qa_id", i),
                    "question": qa["question"],
                    "recall_at_k": metrics["recall_at_k"],
                    "mrr": metrics["mrr"],
                }
                if args.with_generation:
                    row["exact_match"] = metrics["exact_match"]
                    row["f1"] = metrics["f1"]
                    row["rouge_l"] = metrics["rouge_l"]

                results_rows.append(row)

            # Save detailed results
            df_res = pd.DataFrame(results_rows)
            csv_path = os.path.join(
                "results", f"{prefix}{retriever_name}_top{args.top_k}.csv"
            )
            df_res.to_csv(csv_path, index=False)
            print(f"[{retriever_name}] Saved detailed results to {csv_path}")

            # Compute summary stats
            avg_ret_time = (
                sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
            )
            avg_recall = (
                sum(m["recall_at_k"] for m in all_metrics) / len(all_metrics)
                if all_metrics
                else 0
            )
            avg_mrr = (
                sum(m["mrr"] for m in all_metrics) / len(all_metrics)
                if all_metrics
                else 0
            )

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
                summary_row["avg_em"] = sum(
                    m["exact_match"] for m in all_metrics
                ) / len(all_metrics)
                summary_row["avg_f1"] = sum(m["f1"] for m in all_metrics) / len(
                    all_metrics
                )
                summary_row["avg_rouge_l"] = sum(
                    m["rouge_l"] for m in all_metrics
                ) / len(all_metrics)

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
    sum_csv_path = os.path.join("results", f"{prefix}summary_top{args.top_k}.csv")
    df_sum.to_csv(sum_csv_path, index=False)
    print(f"\nSaved final summary to {sum_csv_path}")

    print("\n--- Final Summary Table ---")
    print(df_sum.to_string(index=False))

    # Identify Best Retriever
    valid_res = [r for r in summary_data if r["status"] == "OK"]
    if valid_res:
        # Sort by: highest recall, then highest MRR, then lowest retrieval time
        valid_res.sort(
            key=lambda x: (
                x["avg_recall_at_k"],
                x["avg_mrr"],
                -x["avg_retrieval_time_sec"],
            ),
            reverse=True,
        )
        best = valid_res[0]
        print("\n🏆 Best Retriever:")
        print(f"  Name: {best['retriever']}")
        print(f"  Recall@{args.top_k}: {best['avg_recall_at_k']:.4f}")
        print(f"  MRR: {best['avg_mrr']:.4f}")
        print(f"  Avg Retrieval Time: {best['avg_retrieval_time_sec']:.4f}s")
    else:
        print("\nNo retrievers completed successfully to determine the best.")


if __name__ == "__main__":
    main()
