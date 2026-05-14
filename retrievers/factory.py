"""
Factory function to get retriever by name.
"""

from retrievers import BaseRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import ContrieverRetriever
from retrievers.dpr_original_retriever import OriginalDPRRetriever
from retrievers.colbert_retriever import ColBERTRetriever


def get_retriever(name: str) -> BaseRetriever:
    """Get retriever instance by name."""
    retrievers = {
        "bm25": BM25Retriever,
        "dpr": OriginalDPRRetriever,
        "contriever": ContrieverRetriever,
        "colbertv2": ColBERTRetriever,
    }

    if name not in retrievers:
        raise ValueError(f"Unknown retriever: {name}. Choose from {list(retrievers.keys())}")

    return retrievers[name]()
