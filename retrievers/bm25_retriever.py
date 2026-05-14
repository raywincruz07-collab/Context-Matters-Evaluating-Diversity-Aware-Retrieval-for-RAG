"""
BM25 sparse retriever using rank_bm25.
"""

import os
import pickle
from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi

from retrievers import BaseRetriever
from config import INDEX_DIR


class BM25Retriever(BaseRetriever):
    def __init__(self):
        super().__init__("bm25")
        self.bm25 = None
        self.tokenized_corpus = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def index(self, corpus: List[Dict]):
        """Build BM25 index."""
        self.corpus = corpus
        index_path = os.path.join(INDEX_DIR, "bm25_index.pkl")

        if os.path.exists(index_path):
            print("Loading BM25 index from disk...")
            with open(index_path, "rb") as f:
                self.bm25, self.tokenized_corpus = pickle.load(f)
        else:
            print("Building BM25 index...")
            self.tokenized_corpus = [self._tokenize(doc["text"]) for doc in corpus]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

            with open(index_path, "wb") as f:
                pickle.dump((self.bm25, self.tokenized_corpus), f)
            print("BM25 index saved.")

        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve using BM25 scoring."""
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.corpus[idx], float(scores[idx])))

        return results
