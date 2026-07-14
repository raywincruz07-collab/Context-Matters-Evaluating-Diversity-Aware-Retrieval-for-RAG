"""
Shared helpers for diversification modules (DPP, clustering).

These replicate the candidate-preparation preamble from mmr.py so that
dpp.py and clustering.py can share the logic without touching the frozen mmr.py.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def prepare_candidates(
    candidates: List[Tuple[Dict, float]],
    embed_fn: Callable[[List[str]], np.ndarray],
    precomputed_embs: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Unpack candidates, min-max-normalise scores, and return L2-normalised embeddings.

    Returns:
        docs         — list of doc dicts, same order as candidates
        norm_scores  — np.ndarray float64, shape [n], values in [0, 1]
        embeddings   — np.ndarray float32, shape [n, dim], L2-normalised
    """
    docs = [doc for doc, _ in candidates]
    raw_scores = np.array([score for _, score in candidates], dtype=np.float64)

    score_min, score_max = raw_scores.min(), raw_scores.max()
    if score_max > score_min:
        norm_scores = (raw_scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.ones_like(raw_scores)

    if precomputed_embs is not None:
        try:
            embeddings = np.array(
                [precomputed_embs[doc["doc_id"]] for doc in docs], dtype=np.float32
            )
        except KeyError as exc:
            raise KeyError(
                f"doc_id {exc} missing from precomputed_embs; "
                "ensure corpus embeddings cover all candidates."
            ) from exc
    else:
        texts = [doc.get("text", "") for doc in docs]
        embeddings = embed_fn(texts).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    return docs, norm_scores, embeddings


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return [n, n] cosine similarity matrix for L2-normalised embeddings."""
    return (embeddings @ embeddings.T).astype(np.float64)
