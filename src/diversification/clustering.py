"""
Clustering-based diversification reranking — Sprint 2.

After grouping candidates into k clusters by their Contriever embeddings, the
top-ranked document from each cluster is selected first, then round-robin fills
remaining slots from each cluster's runners-up.

Supported methods: "kmeans", "agglomerative".
When n_clusters == 1 the method degenerates to a plain relevance top-k.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from diversification._common import prepare_candidates


def cluster_rerank(
    query: str,
    candidates: List[Tuple[Dict, float]],
    top_k: int,
    embed_fn: Callable[[List[str]], np.ndarray],
    precomputed_embs: Optional[Dict[int, np.ndarray]] = None,
    n_clusters: int = 5,
    method: str = "kmeans",
    seed: int = 42,
) -> List[Tuple[Dict, float]]:
    """
    Rerank candidates by selecting the most relevant item from each cluster first,
    then filling remaining slots in round-robin order.

    Args:
        query:           Query string (unused; kept for interface symmetry).
        candidates:      List of (doc, score), relevance-descending.
        top_k:           Number of documents to return.
        embed_fn:        Callable(texts) -> np.ndarray [n, dim].
        precomputed_embs: Optional {doc_id -> embedding}.
        n_clusters:      Number of clusters. Clamped to len(candidates).
                         n_clusters=1 degenerates to relevance top-k.
        method:          "kmeans" or "agglomerative".
        seed:            Random seed for KMeans initialisation.

    Returns:
        List of (doc, norm_score) of length min(top_k, len(candidates)).

    Raises:
        ValueError: if method is not recognised.
    """
    if not candidates:
        return []
    if method not in ("kmeans", "agglomerative"):
        raise ValueError(f"method must be 'kmeans' or 'agglomerative', got {method!r}")

    docs, norm_scores, embeddings = prepare_candidates(candidates, embed_fn, precomputed_embs)
    top_k = min(top_k, len(docs))
    n_clusters = max(1, min(n_clusters, len(docs)))

    if n_clusters == 1:
        # Degenerate case: pure relevance ranking
        order = list(np.argsort(-norm_scores))
        return [(docs[i], float(norm_scores[i])) for i in order[:top_k]]

    # Cluster the candidate embeddings
    labels = _cluster(embeddings, n_clusters=n_clusters, method=method, seed=seed)

    # Build per-cluster queues sorted by relevance (best first)
    cluster_queues: Dict[int, List[int]] = {c: [] for c in range(n_clusters)}
    for idx, label in enumerate(labels):
        cluster_queues[label].append(idx)
    for c in cluster_queues:
        cluster_queues[c].sort(key=lambda i: -norm_scores[i])

    # Round-robin fill
    selected: List[int] = []
    cluster_order = list(range(n_clusters))
    pos = {c: 0 for c in cluster_order}

    while len(selected) < top_k:
        made_progress = False
        for c in cluster_order:
            if len(selected) >= top_k:
                break
            queue = cluster_queues[c]
            while pos[c] < len(queue) and queue[pos[c]] in selected:
                pos[c] += 1
            if pos[c] < len(queue):
                selected.append(queue[pos[c]])
                pos[c] += 1
                made_progress = True
        if not made_progress:
            break

    return [(docs[i], float(norm_scores[i])) for i in selected[:top_k]]


def _cluster(
    embeddings: np.ndarray,
    n_clusters: int,
    method: str,
    seed: int,
) -> np.ndarray:
    """Return integer label array [n] using the requested clustering method."""
    if method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        return km.fit_predict(embeddings)

    # method == "agglomerative"
    from sklearn.cluster import AgglomerativeClustering

    ag = AgglomerativeClustering(n_clusters=n_clusters)
    return ag.fit_predict(embeddings)
