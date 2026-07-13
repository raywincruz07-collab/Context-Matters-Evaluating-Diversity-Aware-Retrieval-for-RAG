"""
RAG Pipeline: ties together retriever, generator, and evaluation.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

from config import DATA_DIR, TOP_K
from data_prep import load_data, prepare_data
from evaluation import evaluate_batch, evaluate_single
from generator import MakiGenerator
from retrievers.factory import get_retriever


class RAGPipeline:
    def __init__(self, retriever_name: str = "bm25", top_k: int = TOP_K):
        self.retriever_name = retriever_name
        self.top_k = top_k
        self.retriever = None
        self.generator = MakiGenerator()
        self.corpus = None
        self.qa_pairs = None

    def setup(self):
        """Initialize data, retriever, and generator."""
        # Load or prepare data
        print("Loading data...")
        self.corpus, self.qa_pairs = prepare_data()
        print(f"Corpus: {len(self.corpus)} docs | QA pairs: {len(self.qa_pairs)}")

        # Initialize retriever
        print(f"\nInitializing {self.retriever_name} retriever...")
        self.retriever = get_retriever(self.retriever_name)
        self.retriever.index(self.corpus)
        print(f"{self.retriever_name} retriever ready.")

        # Check generator
        if self.generator.is_available():
            print(f"\nUniversity GPU generator ready (model: {self.generator.model})")
        else:
            available = self.generator.list_models()
            print(
                f"\nWarning: University GPU model '{self.generator.model}' not found."
            )
            if available:
                print(f"Available models: {available}")
            else:
                print("Set MAKI_API_KEY or check the API host/model.")

    def _get_mmr_embed_fn(self):
        """Lazy-load a ContrieverRetriever and return its _encode method."""
        if not hasattr(self, "_mmr_retriever") or self._mmr_retriever is None:
            from retrievers.dense_retriever import ContrieverRetriever
            self._mmr_retriever = ContrieverRetriever()
            self._mmr_retriever._load_model()
        return self._mmr_retriever._encode

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        diversification: Optional[str] = None,
    ) -> Dict:
        """
        Run full RAG pipeline for a single question.

        Args:
            question: Input question string.
            top_k: Override the default retrieval depth.
            diversification: Optional reranking strategy.  Accepted values:
                - None: standard retrieval (Sprint 1 behaviour unchanged)
                - "mmr_<lambda>": MMR reranking with given lambda, e.g. "mmr_0.5"

        Returns dict with:
            - question
            - retrieved_docs: list of (doc, score)
            - answer: generated answer
            - retrieval_time: seconds
            - generation_time: seconds
            - diversification: the strategy used (or None)
        """
        from config import MMR_CANDIDATE_POOL
        from diversification.mmr import mmr_rerank

        k = top_k or self.top_k

        t0 = time.time()
        if diversification is not None and diversification.startswith("mmr_"):
            lambda_param = float(diversification.split("_", 1)[1])
            candidates = self.retriever.retrieve(question, top_k=MMR_CANDIDATE_POOL)
            embed_fn = self._get_mmr_embed_fn()
            retrieved = mmr_rerank(
                question, candidates, top_k=k, lambda_param=lambda_param, embed_fn=embed_fn
            )
        else:
            retrieved = self.retriever.retrieve(question, top_k=k)
        retrieval_time = time.time() - t0

        # Generate
        context_docs = [doc for doc, score in retrieved]
        t0 = time.time()
        answer = self.generator.generate(question, context_docs)
        generation_time = time.time() - t0

        return {
            "question": question,
            "retrieved_docs": retrieved,
            "answer": answer,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "diversification": diversification,
        }

    def evaluate(
        self, num_samples: Optional[int] = None, top_k: Optional[int] = None
    ) -> Dict:
        """
        Run evaluation on QA pairs.

        Args:
            num_samples: Number of samples to evaluate (None = all)
            top_k: Override top_k for this evaluation run

        Returns:
            Dict with aggregated metrics and per-example results
        """
        k = top_k or self.top_k
        samples = self.qa_pairs[:num_samples] if num_samples else self.qa_pairs

        all_results = []
        for i, qa in enumerate(samples):
            print(f"\rEvaluating {i+1}/{len(samples)}...", end="", flush=True)

            # Retrieve
            retrieved = self.retriever.retrieve(qa["question"], top_k=k)
            retrieved_doc_ids = [doc["doc_id"] for doc, _ in retrieved]

            # Generate
            context_docs = [doc for doc, _ in retrieved]
            answer = self.generator.generate(qa["question"], context_docs)

            # Evaluate
            metrics = evaluate_single(
                prediction=answer,
                ground_truth=qa["long_answer"],
                retrieved_doc_ids=retrieved_doc_ids,
                gold_doc_ids=qa["gold_doc_ids"],
            )

            all_results.append(
                {
                    "qa_id": qa["qa_id"],
                    "question": qa["question"],
                    "prediction": answer,
                    "ground_truth": qa["long_answer"],
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "gold_doc_ids": qa["gold_doc_ids"],
                    **metrics,
                }
            )

        print()  # newline after progress

        aggregated = evaluate_batch(all_results)

        return {
            "retriever": self.retriever_name,
            "top_k": k,
            "num_samples": len(samples),
            "aggregated_metrics": aggregated,
            "per_example": all_results,
        }

    def save_results(self, results: Dict, filename: Optional[str] = None):
        """Save evaluation results to JSON."""
        if filename is None:
            filename = f"eval_{self.retriever_name}_top{self.top_k}.json"

        path = os.path.join(DATA_DIR, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {path}")
