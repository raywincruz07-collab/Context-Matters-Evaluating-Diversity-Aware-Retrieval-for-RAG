"""
Tests for DPP, clustering, and dispatch diversification modules — Sprint 2.

Fixture: 12 candidates in 3 clusters of 4, top-4 rigged to the same cluster
so that diversity-aware methods must pull from other clusters.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diversification._common import cosine_similarity_matrix, prepare_candidates
from diversification.dpp import build_dpp_kernel, dpp_rerank
from diversification.clustering import cluster_rerank
from diversification.dispatch import is_diversified, parse_condition, rerank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_embed_fn(embeddings: np.ndarray):
    """Return an embed_fn that returns the given matrix (ignores text)."""
    def embed_fn(texts):
        return embeddings.astype(np.float32)
    return embed_fn


@pytest.fixture
def candidates_12():
    """
    12 candidates in 3 clusters of 4.
    Cluster A (indices 0-3): direction [1, 0, 0, ...]
    Cluster B (indices 4-7): direction [0, 1, 0, ...]
    Cluster C (indices 8-11): direction [0, 0, 1, ...]

    Relevance scores: 1.0, 0.9, 0.8, 0.7  (cluster A best),
                      0.6, 0.5, 0.4, 0.3  (cluster B mid),
                      0.2, 0.1, 0.05, 0.01 (cluster C low)
    -> top-4 by relevance are all in cluster A.
    """
    n = 12
    dim = 8

    embs = np.zeros((n, dim), dtype=np.float32)
    # Cluster A
    for i in range(4):
        embs[i, 0] = 1.0
        embs[i, i % dim] += 0.05  # tiny perturbation so they're not identical
    # Cluster B
    for i in range(4, 8):
        embs[i, 1] = 1.0
        embs[i, i % dim] += 0.05
    # Cluster C
    for i in range(8, 12):
        embs[i, 2] = 1.0
        embs[i, i % dim] += 0.05

    # L2-normalise
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms > 0, norms, 1.0)

    scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    docs = [{"doc_id": i, "text": f"doc_{i}"} for i in range(n)]
    candidates = list(zip(docs, scores))

    return candidates, embs


# ---------------------------------------------------------------------------
# _common tests
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_output_shapes(self, candidates_12):
        cands, embs = candidates_12
        docs, norm_scores, embeddings = prepare_candidates(cands, _make_embed_fn(embs))
        assert len(docs) == 12
        assert norm_scores.shape == (12,)
        assert embeddings.shape[0] == 12

    def test_norm_scores_range(self, candidates_12):
        cands, embs = candidates_12
        _, norm_scores, _ = prepare_candidates(cands, _make_embed_fn(embs))
        assert float(norm_scores.min()) >= 0.0
        assert float(norm_scores.max()) <= 1.0 + 1e-9

    def test_embeddings_l2_normalised(self, candidates_12):
        cands, embs = candidates_12
        _, _, embeddings = prepare_candidates(cands, _make_embed_fn(embs))
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(12), atol=1e-5)

    def test_uniform_scores_give_all_ones(self):
        docs = [{"doc_id": i, "text": ""} for i in range(3)]
        cands = [(d, 5.0) for d in docs]
        embs = np.eye(3, dtype=np.float32)
        _, norm_scores, _ = prepare_candidates(cands, _make_embed_fn(embs))
        np.testing.assert_array_equal(norm_scores, np.ones(3))

    def test_cosine_sim_matrix_diagonal(self, candidates_12):
        _, embs = candidates_12
        S = cosine_similarity_matrix(embs)
        np.testing.assert_allclose(np.diag(S), np.ones(12), atol=1e-5)


# ---------------------------------------------------------------------------
# build_dpp_kernel tests
# ---------------------------------------------------------------------------

class TestDPPKernel:
    def test_kernel_shape(self, candidates_12):
        cands, embs = candidates_12
        _, norm_scores, embeddings = prepare_candidates(cands, _make_embed_fn(embs))
        L = build_dpp_kernel(norm_scores, embeddings)
        assert L.shape == (12, 12)

    def test_kernel_symmetric(self, candidates_12):
        cands, embs = candidates_12
        _, norm_scores, embeddings = prepare_candidates(cands, _make_embed_fn(embs))
        L = build_dpp_kernel(norm_scores, embeddings)
        np.testing.assert_allclose(L, L.T, atol=1e-9)

    def test_kernel_psd(self, candidates_12):
        cands, embs = candidates_12
        _, norm_scores, embeddings = prepare_candidates(cands, _make_embed_fn(embs))
        L = build_dpp_kernel(norm_scores, embeddings)
        eigenvalues = np.linalg.eigvalsh(L)
        assert float(eigenvalues.min()) >= -1e-6


# ---------------------------------------------------------------------------
# dpp_rerank tests
# ---------------------------------------------------------------------------

class TestDPPRerank:
    def test_returns_top_k(self, candidates_12):
        cands, embs = candidates_12
        result = dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs))
        assert len(result) == 4

    def test_map_is_deterministic(self, candidates_12):
        cands, embs = candidates_12
        r1 = dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="map")
        r2 = dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="map")
        ids1 = [d["doc_id"] for d, _ in r1]
        ids2 = [d["doc_id"] for d, _ in r2]
        assert ids1 == ids2

    def test_sample_requires_seed(self, candidates_12):
        cands, embs = candidates_12
        with pytest.raises(ValueError, match="seed"):
            dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="sample")

    def test_sample_same_seed_deterministic(self, candidates_12):
        cands, embs = candidates_12
        r1 = dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="sample", seed=7)
        r2 = dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="sample", seed=7)
        ids1 = [d["doc_id"] for d, _ in r1]
        ids2 = [d["doc_id"] for d, _ in r2]
        assert ids1 == ids2

    def test_map_promotes_diversity(self, candidates_12):
        """MAP should not return all 4 results from cluster A (indices 0-3)."""
        cands, embs = candidates_12
        result = dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="map")
        ids = {d["doc_id"] for d, _ in result}
        cluster_a = {0, 1, 2, 3}
        assert ids != cluster_a, "DPP MAP returned all items from one cluster — no diversity"

    def test_empty_candidates(self):
        result = dpp_rerank("q", [], top_k=4, embed_fn=lambda t: np.zeros((0, 8)))
        assert result == []

    def test_top_k_larger_than_candidates(self, candidates_12):
        cands, embs = candidates_12
        result = dpp_rerank("q", cands[:3], top_k=10, embed_fn=_make_embed_fn(embs[:3]))
        assert len(result) == 3

    def test_invalid_mode(self, candidates_12):
        cands, embs = candidates_12
        with pytest.raises(ValueError, match="mode"):
            dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="bad")


# ---------------------------------------------------------------------------
# cluster_rerank tests
# ---------------------------------------------------------------------------

class TestClusterRerank:
    def test_returns_top_k(self, candidates_12):
        cands, embs = candidates_12
        result = cluster_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), n_clusters=3)
        assert len(result) == 4

    def test_kmeans_promotes_diversity(self, candidates_12):
        """k-means should pull from multiple clusters."""
        cands, embs = candidates_12
        result = cluster_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs),
                                n_clusters=3, method="kmeans", seed=42)
        ids = {d["doc_id"] for d, _ in result}
        cluster_a = {0, 1, 2, 3}
        assert ids != cluster_a

    def test_agglomerative_promotes_diversity(self, candidates_12):
        cands, embs = candidates_12
        result = cluster_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs),
                                n_clusters=3, method="agglomerative")
        ids = {d["doc_id"] for d, _ in result}
        cluster_a = {0, 1, 2, 3}
        assert ids != cluster_a

    def test_n_clusters_1_is_relevance_topk(self, candidates_12):
        cands, embs = candidates_12
        result = cluster_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), n_clusters=1)
        ids = [d["doc_id"] for d, _ in result]
        assert ids == [0, 1, 2, 3]

    def test_empty_candidates(self):
        result = cluster_rerank("q", [], top_k=4, embed_fn=lambda t: np.zeros((0, 8)))
        assert result == []

    def test_top_k_larger_than_candidates(self, candidates_12):
        cands, embs = candidates_12
        result = cluster_rerank("q", cands[:3], top_k=10, embed_fn=_make_embed_fn(embs[:3]),
                                n_clusters=2)
        assert len(result) == 3

    def test_invalid_method(self, candidates_12):
        cands, embs = candidates_12
        with pytest.raises(ValueError, match="method"):
            cluster_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), method="bad")


# ---------------------------------------------------------------------------
# parse_condition / is_diversified tests
# ---------------------------------------------------------------------------

class TestParseCondition:
    def test_none(self):
        family, kwargs = parse_condition("none")
        assert family == "none"
        assert kwargs == {}

    def test_mmr(self):
        family, kwargs = parse_condition("mmr_0.5")
        assert family == "mmr"
        assert abs(kwargs["lambda_param"] - 0.5) < 1e-9

    def test_mmr_zero(self):
        family, kwargs = parse_condition("mmr_0.0")
        assert family == "mmr"
        assert kwargs["lambda_param"] == 0.0

    def test_mmr_one(self):
        family, kwargs = parse_condition("mmr_1.0")
        assert family == "mmr"
        assert kwargs["lambda_param"] == 1.0

    def test_kmeans(self):
        family, kwargs = parse_condition("kmeans_k5")
        assert family == "kmeans"
        assert kwargs["n_clusters"] == 5
        assert kwargs["method"] == "kmeans"

    def test_agglo(self):
        family, kwargs = parse_condition("agglo_k3")
        assert family == "agglo"
        assert kwargs["n_clusters"] == 3
        assert kwargs["method"] == "agglomerative"

    def test_dpp_map(self):
        family, kwargs = parse_condition("dpp_map")
        assert family == "dpp_map"
        assert kwargs["mode"] == "map"

    def test_dpp_seed(self):
        family, kwargs = parse_condition("dpp_seed42")
        assert family == "dpp_sample"
        assert kwargs["mode"] == "sample"
        assert kwargs["seed"] == 42

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown condition"):
            parse_condition("bad_condition")

    def test_is_diversified_none(self):
        assert is_diversified("none") is False

    def test_is_diversified_mmr(self):
        assert is_diversified("mmr_0.5") is True

    def test_is_diversified_dpp(self):
        assert is_diversified("dpp_map") is True


# ---------------------------------------------------------------------------
# dispatch.rerank integration tests
# ---------------------------------------------------------------------------

class TestDispatchRerank:
    def test_none_returns_top_k(self, candidates_12):
        cands, embs = candidates_12
        result = rerank("none", "q", cands, top_k=4, embed_fn=_make_embed_fn(embs))
        assert len(result) == 4
        ids = [d["doc_id"] for d, _ in result]
        assert ids == [0, 1, 2, 3]

    def test_mmr_dispatch(self, candidates_12):
        cands, embs = candidates_12
        result = rerank("mmr_0.5", "q", cands, top_k=4, embed_fn=_make_embed_fn(embs))
        assert len(result) == 4

    def test_kmeans_dispatch(self, candidates_12):
        cands, embs = candidates_12
        result = rerank("kmeans_k3", "q", cands, top_k=4, embed_fn=_make_embed_fn(embs))
        assert len(result) == 4

    def test_agglo_dispatch(self, candidates_12):
        cands, embs = candidates_12
        result = rerank("agglo_k3", "q", cands, top_k=4, embed_fn=_make_embed_fn(embs))
        assert len(result) == 4

    def test_dpp_map_dispatch(self, candidates_12):
        cands, embs = candidates_12
        result = rerank("dpp_map", "q", cands, top_k=4, embed_fn=_make_embed_fn(embs))
        assert len(result) == 4

    def test_dpp_seed_dispatch(self, candidates_12):
        cands, embs = candidates_12
        result = rerank("dpp_seed99", "q", cands, top_k=4, embed_fn=_make_embed_fn(embs))
        assert len(result) == 4

    def test_unknown_condition_raises(self, candidates_12):
        cands, embs = candidates_12
        with pytest.raises(ValueError):
            rerank("bad_cond", "q", cands, top_k=4, embed_fn=_make_embed_fn(embs))
