"""
Live retriever: fetches PubMed abstracts on-the-fly for a query,
then runs BM25 over the fetched documents.

Used for custom user questions where the static corpus has no coverage.
"""

from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

from pubmed_fetcher import fetch_docs_for_query


class LivePubMedRetriever:
    """
    For each query:
    1. Search PubMed for relevant papers
    2. Fetch their abstracts
    3. Run BM25 over the fetched sections
    4. Return top-k

    No pre-built index needed — fully dynamic.
    """

    def __init__(self, fetch_n: int = 20):
        self.name = "live_pubmed"
        self.fetch_n = fetch_n  # how many papers to pull from PubMed

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Tuple[Dict, float]], List[Dict]]:
        """
        Retrieve top-k docs for query via live PubMed fetch + BM25.

        Returns:
            (results, all_fetched_docs)
            results: list of (doc, score) sorted by relevance
            all_fetched_docs: all docs fetched from PubMed (for display)
        """
        # Fetch from PubMed
        fetched_docs = fetch_docs_for_query(
            query=query,
            max_results=self.fetch_n,
            doc_id_offset=1_000_000,  # avoid collision with static corpus IDs
        )

        if not fetched_docs:
            return [], []

        # BM25 over fetched docs
        tokenized = [doc["text"].lower().split() for doc in fetched_docs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())

        top_indices = scores.argsort()[-top_k:][::-1]
        results = [
            (fetched_docs[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

        return results, fetched_docs
