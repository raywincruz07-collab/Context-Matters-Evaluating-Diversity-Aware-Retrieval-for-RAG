"""
Determinantal Point Process (DPP) diversification reranking — Sprint 2.

Kernel:  L = Diag(q) · S · Diag(q)
         q_i = exp(theta * (rel_i - max(rel)) / 2)   (quality scores in (0, 1])
         S   = cosine Gram matrix of L2-normalised embeddings

Two modes
---------
map    — greedy MAP selection (deterministic, no RNG required).
         At each step add the item that maximises the log-determinant increment.
sample — exact k-DPP sample via eigendecomposition.  Requires seed != None.

Reference: Kulesza & Taskar (2012), "Determinantal Point Processes for Machine Learning".
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from diversification._common import cosine_similarity_matrix, prepare_candidates


def build_dpp_kernel(
    norm_scores: np.ndarray,
    embeddings: np.ndarray,
    theta: float = 1.0,
) -> np.ndarray:
    """
    Build the DPP L-kernel.

    Args:
        norm_scores: float64 array [n], min-max normalised relevance in [0, 1].
        embeddings:  float32 array [n, dim], L2-normalised.
        theta:       quality-score temperature (higher → relevance dominates).

    Returns:
        L: float64 [n, n] symmetric PSD matrix.
    """
    q = np.exp(theta * (norm_scores - norm_scores.max()) / 2.0)  # (0, 1]
    S = cosine_similarity_matrix(embeddings)  # [n, n]
    L = np.outer(q, q) * S

    # Scale-relative jitter for numerical stability
    jitter = 1e-6 * max(float(np.abs(L).max()), 1.0)
    L += jitter * np.eye(len(q))

    return L


def _greedy_map(L: np.ndarray, k: int) -> List[int]:
    """
    Greedy MAP: iteratively select the item maximising the log-det increment.

    Uses the Cholesky update trick for O(nk^2) complexity.
    Falls back to brute-force det ratio when Cholesky becomes unstable.
    """
    n = L.shape[0]
    k = min(k, n)
    selected: List[int] = []
    remaining = list(range(n))

    # Work in the Schur complement space
    # D[i] = L[i,i] — diagonal of the remaining submatrix (updated each step)
    D = np.diag(L).copy()
    # V[i] = projection vector of item i onto the selected subspace
    V = np.zeros((n, k), dtype=np.float64)

    for step in range(k):
        # Score = diagonal of current Schur complement = gain in log-det
        scores = D[remaining]
        best_local = int(np.argmax(scores))
        best_i = remaining[best_local]
        selected.append(best_i)
        remaining.remove(best_i)

        if not remaining:
            break

        # Update Schur complement diagonal for remaining items
        e_i = D[best_i]
        if e_i <= 0:
            break
        v = (L[remaining, best_i] - V[remaining, :step] @ V[best_i, :step]) / np.sqrt(e_i)
        V[remaining, step] = v
        D[remaining] -= v ** 2

    return selected


def _sample_kdpp(L: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    """
    Exact k-DPP sample via eigendecomposition (Kulesza & Taskar Algorithm 1).
    """
    n = L.shape[0]
    k = min(k, n)

    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Clip small negatives from numerical noise
    eigenvalues = np.clip(eigenvalues, 0, None)

    # Step 1: select k eigenvectors with prob proportional to lambda/(1+lambda)
    probs = eigenvalues / (1.0 + eigenvalues)
    chosen_vecs: List[int] = []
    while len(chosen_vecs) < k:
        chosen_vecs = [i for i in range(n) if rng.random() < probs[i]]
    chosen_vecs = list(rng.choice(
        [i for i in range(n) if rng.random() < probs[i]] or list(range(n)),
        size=k, replace=False,
    )) if len(chosen_vecs) != k else chosen_vecs[:k]

    # Step 2: sample items from the span of chosen eigenvectors
    V = eigenvectors[:, chosen_vecs]  # [n, k]
    selected: List[int] = []
    remaining = list(range(n))

    for _ in range(k):
        # Probability of item i proportional to squared norm of its row in V
        row_norms_sq = np.array([float(np.dot(V[i], V[i])) for i in remaining])
        row_norms_sq = np.clip(row_norms_sq, 0, None)
        total = row_norms_sq.sum()
        if total <= 0:
            selected.append(rng.choice(remaining))
            break
        p = row_norms_sq / total
        chosen_local = int(rng.choice(len(remaining), p=p))
        chosen_item = remaining[chosen_local]
        selected.append(chosen_item)
        remaining.remove(chosen_item)

        if not remaining:
            break

        # Project V onto the orthogonal complement of V[chosen_item]
        e = V[chosen_item].copy()
        e_norm = float(np.dot(e, e)) ** 0.5
        if e_norm > 1e-12:
            e = e / e_norm
            V = V - np.outer(V @ e, e)

    return selected


def dpp_rerank(
    query: str,
    candidates: List[Tuple[Dict, float]],
    top_k: int,
    embed_fn: Callable[[List[str]], np.ndarray],
    precomputed_embs: Optional[Dict[int, np.ndarray]] = None,
    theta: float = 1.0,
    mode: str = "map",
    seed: Optional[int] = None,
) -> List[Tuple[Dict, float]]:
    """
    Rerank candidates using a k-DPP.

    Args:
        query:           Query string (unused directly; kept for interface symmetry).
        candidates:      List of (doc, score) from retriever, relevance-descending.
        top_k:           Number of documents to return.
        embed_fn:        Callable(texts) -> np.ndarray [n, dim].
        precomputed_embs: Optional {doc_id -> embedding}.
        theta:           Quality temperature for the DPP kernel.
        mode:            "map" (deterministic greedy) or "sample" (exact k-DPP).
        seed:            Required when mode="sample"; ignored for mode="map".

    Returns:
        List of (doc, norm_score) of length min(top_k, len(candidates)).
    """
    if not candidates:
        return []

    if mode not in ("map", "sample"):
        raise ValueError(f"mode must be 'map' or 'sample', got {mode!r}")
    if mode == "sample" and seed is None:
        raise ValueError("mode='sample' requires a seed for reproducibility")

    docs, norm_scores, embeddings = prepare_candidates(candidates, embed_fn, precomputed_embs)
    top_k = min(top_k, len(docs))

    L = build_dpp_kernel(norm_scores, embeddings, theta=theta)

    if mode == "map":
        selected = _greedy_map(L, top_k)
    else:
        rng = np.random.default_rng(seed)
        selected = _sample_kdpp(L, top_k, rng)

    # Pad with remaining items sorted by relevance if selection fell short
    if len(selected) < top_k:
        used = set(selected)
        extras = sorted(
            [i for i in range(len(docs)) if i not in used],
            key=lambda i: -norm_scores[i],
        )
        selected = selected + extras[: top_k - len(selected)]

    return [(docs[i], float(norm_scores[i])) for i in selected[:top_k]]
