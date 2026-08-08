"""
Tests for DPP, clustering, and dispatch diversification modules — Sprint 2.

Fixture: 12 candidates in 3 clusters of 4, top-4 rigged to the same cluster
so that diversity-aware methods must pull from other clusters.
"""

import sys
import os
from collections import Counter
from itertools import combinations

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diversification._common import cosine_similarity_matrix, prepare_candidates
import diversification.dpp as dpp_module
from diversification.dpp import _greedy_map, _sample_kdpp, build_dpp_kernel, dpp_rerank
import diversification.clustering as clustering_module
from diversification.clustering import _cluster, cluster_rerank
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

    @pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, "not-a-score"])
    def test_invalid_relevance_scores_raise(self, score):
        docs = [{"doc_id": 0, "text": "a"}, {"doc_id": 1, "text": "b"}]
        candidates = [(docs[0], 1.0), (docs[1], score)]
        with pytest.raises(ValueError, match="relevance scores"):
            prepare_candidates(candidates, lambda texts: np.eye(2))

    @pytest.mark.parametrize("score", [True, False, np.bool_(True)])
    def test_boolean_relevance_scores_raise(self, score):
        docs = [{"doc_id": 0, "text": "a"}, {"doc_id": 1, "text": "b"}]
        candidates = [(docs[0], 1.0), (docs[1], score)]
        with pytest.raises(ValueError, match="real numeric"):
            prepare_candidates(candidates, lambda texts: np.eye(2))

    @pytest.mark.parametrize("embeddings,error", [
        (np.array([[1.0, 0.0], [np.nan, 1.0]]), "finite"),
        (np.array([[1.0, 0.0], [np.inf, 1.0]]), "finite"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), "nonzero row norms"),
        (np.array([1.0, 2.0]), "2-D"),
        (np.ones((1, 2)), "one row per candidate"),
        (np.empty((2, 0)), "positive feature dimension"),
        (np.array([["a", "b"], ["c", "d"]]), "numeric"),
    ])
    def test_invalid_embeddings_raise(self, embeddings, error):
        docs = [{"doc_id": 0, "text": "a"}, {"doc_id": 1, "text": "b"}]
        candidates = [(docs[0], 1.0), (docs[1], 0.0)]
        with pytest.raises(ValueError, match=error):
            prepare_candidates(candidates, lambda texts: embeddings)

    def test_on_demand_complex_embeddings_raise(self):
        docs = [{"doc_id": 0, "text": "a"}, {"doc_id": 1, "text": "b"}]
        candidates = [(docs[0], 1.0), (docs[1], 0.0)]
        embeddings = np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0 - 1.0j]])
        with pytest.raises(ValueError, match="real-valued"):
            prepare_candidates(candidates, lambda texts: embeddings)

    def test_precomputed_complex_embeddings_raise(self):
        docs = [{"doc_id": 0, "text": "a"}, {"doc_id": 1, "text": "b"}]
        candidates = [(docs[0], 1.0), (docs[1], 0.0)]
        precomputed = {
            0: np.array([1.0 + 1.0j, 0.0]),
            1: np.array([0.0, 1.0 - 1.0j]),
        }
        with pytest.raises(ValueError, match="real-valued"):
            prepare_candidates(candidates, lambda texts: None, precomputed)

    def test_missing_precomputed_id_raises(self):
        docs = [{"doc_id": 0, "text": "a"}, {"doc_id": 1, "text": "b"}]
        candidates = [(docs[0], 1.0), (docs[1], 0.0)]
        with pytest.raises(KeyError, match="missing from precomputed_embs"):
            prepare_candidates(
                candidates, lambda texts: pytest.fail("must not fall back"),
                precomputed_embs={0: np.array([1.0, 0.0])},
            )

    def test_valid_negative_relevance_scores(self):
        docs = [{"doc_id": i, "text": str(i)} for i in range(3)]
        candidates = list(zip(docs, [-5.0, -3.0, -1.0]))
        _, norm_scores, _ = prepare_candidates(candidates, lambda texts: np.eye(3))
        np.testing.assert_array_equal(norm_scores, np.array([0.0, 0.5, 1.0]))

    def test_valid_precomputed_embeddings(self):
        docs = [{"doc_id": 10, "text": "a"}, {"doc_id": 11, "text": "b"}]
        candidates = [(docs[0], 2.0), (docs[1], 1.0)]
        precomputed = {
            10: np.array([3.0, 4.0]),
            11: np.array([0.0, 2.0]),
        }
        returned_docs, scores, embeddings = prepare_candidates(
            candidates, lambda texts: pytest.fail("must not call embed_fn"), precomputed
        )
        assert returned_docs == docs
        np.testing.assert_array_equal(scores, np.array([1.0, 0.0]))
        np.testing.assert_allclose(embeddings, np.array([[0.6, 0.8], [0.0, 1.0]]))

    def test_valid_on_demand_output_regression(self):
        docs = [{"doc_id": i, "text": str(i)} for i in range(3)]
        candidates = list(zip(docs, [-2.0, 0.0, 2.0]))
        raw_embeddings = np.array([[3.0, 4.0], [0.0, 2.0], [-5.0, 0.0]])
        returned_docs, scores, embeddings = prepare_candidates(
            candidates, lambda texts: raw_embeddings
        )
        assert returned_docs == docs
        np.testing.assert_array_equal(scores, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(
            embeddings, np.array([[0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]])
        )
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), np.ones(3))

    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    def test_real_integer_and_float_embeddings_remain_valid(self, dtype):
        docs = [{"doc_id": 0, "text": "a"}, {"doc_id": 1, "text": "b"}]
        candidates = [(docs[0], 2), (docs[1], 1.0)]
        raw_embeddings = np.array([[3, 4], [0, 2]], dtype=dtype)
        _, scores, embeddings = prepare_candidates(
            candidates, lambda texts: raw_embeddings
        )
        np.testing.assert_array_equal(scores, np.array([1.0, 0.0]))
        np.testing.assert_allclose(embeddings, np.array([[0.6, 0.8], [0.0, 1.0]]))


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
# deterministic forward-greedy DPP tests
# ---------------------------------------------------------------------------

class TestGreedyMap:
    @staticmethod
    def _direct_greedy_reference(L, k):
        selected = []
        remaining = list(range(len(L)))
        for _ in range(k):
            log_determinants = []
            for candidate in remaining:
                subset = selected + [candidate]
                sign, log_determinant = np.linalg.slogdet(L[np.ix_(subset, subset)])
                assert sign > 0
                log_determinants.append(log_determinant)
            best_local = int(np.argmax(log_determinants))
            selected.append(remaining.pop(best_local))
        return selected

    @pytest.mark.parametrize("L,k", [
        (np.diag([4.0, 3.0, 2.0, 1.0]), 3),
        (
            np.array([
                [1.0, 0.2, 0.1],
                [0.2, 1.3, 0.4],
                [0.1, 0.4, 0.9],
            ]),
            3,
        ),
    ])
    def test_matches_direct_stepwise_determinant_greedy(self, L, k):
        assert _greedy_map(L, k) == self._direct_greedy_reference(L, k)

    def test_greedy_is_not_global_map(self):
        B = np.array([
            [0.751939396, -0.658760320, -1.228674986],
            [0.257557768, 0.312902918, -0.130811690],
            [1.269983120, -0.092962458, -0.066150889],
            [-1.108214467, 0.135956851, 1.347077764],
        ])
        L = B @ B.T + 0.05 * np.eye(4)
        greedy_set = set(_greedy_map(L, 2))
        subsets = list(combinations(range(4), 2))
        determinants = {
            subset: float(np.linalg.det(L[np.ix_(subset, subset)]))
            for subset in subsets
        }
        global_optimum = max(subsets, key=determinants.get)

        assert greedy_set == {2, 3}
        assert set(global_optimum) == {0, 2}
        assert determinants[global_optimum] > determinants[tuple(sorted(greedy_set))]

    @pytest.mark.parametrize("L,k", [
        (np.zeros((3, 3)), 2),
        (np.diag([2.0, 0.0, 0.0]), 2),
    ])
    def test_nonpositive_pivot_raises(self, L, k):
        with pytest.raises(RuntimeError, match="nonpositive or nonfinite pivot"):
            _greedy_map(L, k)


# ---------------------------------------------------------------------------
# exact k-DPP sampler tests
# ---------------------------------------------------------------------------

class TestSampleKDPP:
    def test_exact_cardinality_and_uniqueness(self):
        L = np.diag([0.2, 0.5, 1.0, 2.0, 4.0])
        for seed in range(20):
            sample = _sample_kdpp(L, 3, np.random.default_rng(seed))
            assert len(sample) == 3
            assert len(set(sample)) == 3
            assert all(0 <= item < 5 for item in sample)

    def test_fixed_seed_reproducibility(self):
        L = np.array([[1.0, 0.3, 0.1], [0.3, 1.2, 0.2], [0.1, 0.2, 0.8]])
        first = _sample_kdpp(L, 2, np.random.default_rng(17))
        second = _sample_kdpp(L, 2, np.random.default_rng(17))
        assert first == second

    def test_k_zero(self):
        assert _sample_kdpp(np.eye(3), 0, np.random.default_rng(1)) == []

    def test_nonsquare_kernel_raises(self):
        with pytest.raises(ValueError, match="square"):
            _sample_kdpp(np.ones((2, 3)), 0, np.random.default_rng(1))

    def test_nonfinite_kernel_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _sample_kdpp(np.diag([1.0, np.inf]), 0, np.random.default_rng(1))

    def test_nonsymmetric_kernel_raises(self):
        L = np.array([[1.0, 0.5], [0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            _sample_kdpp(L, 0, np.random.default_rng(1))

    @pytest.mark.parametrize("k", [-1, 4, 1.5])
    def test_invalid_k(self, k):
        with pytest.raises(ValueError, match="k must"):
            _sample_kdpp(np.eye(3), k, np.random.default_rng(1))

    def test_rank_below_k_raises(self):
        with pytest.raises(ValueError, match="zero mass"):
            _sample_kdpp(np.diag([2.0, 0.0, 0.0]), 2, np.random.default_rng(1))

    def test_zero_kernel_raises(self):
        with pytest.raises(ValueError, match="zero mass"):
            _sample_kdpp(np.zeros((3, 3)), 1, np.random.default_rng(1))

    def test_tiny_negative_eigenvalue_is_tolerated(self):
        L = np.diag([-1e-12, 1.0, 2.0])
        sample = _sample_kdpp(L, 2, np.random.default_rng(3))
        assert set(sample) == {1, 2}

    def test_materially_indefinite_kernel_raises(self):
        with pytest.raises(ValueError, match="positive semidefinite"):
            _sample_kdpp(np.diag([-1e-4, 1.0, 2.0]), 2, np.random.default_rng(1))

    def test_tiny_scale_materially_indefinite_kernel_raises(self):
        L = np.diag([-1e-15, 1e-14, 2e-14])
        with pytest.raises(ValueError, match="positive semidefinite"):
            _sample_kdpp(L, 1, np.random.default_rng(1))

    def test_identity_kernel_is_uniform(self):
        L = np.eye(4)
        subsets = list(combinations(range(4), 2))
        counts = Counter()
        rng = np.random.default_rng(20260808)
        draws = 12_000
        for _ in range(draws):
            counts[tuple(sorted(_sample_kdpp(L, 2, rng)))] += 1
        empirical = np.array([counts[subset] / draws for subset in subsets])
        assert 0.5 * np.abs(empirical - 1.0 / len(subsets)).sum() < 0.03

    def test_diagonal_kernel_matches_analytic_distribution(self):
        diagonal = np.array([0.2, 0.7, 1.5, 3.0])
        L = np.diag(diagonal)
        subsets = list(combinations(range(4), 2))
        weights = np.array([diagonal[i] * diagonal[j] for i, j in subsets])
        expected = weights / weights.sum()
        counts = Counter()
        rng = np.random.default_rng(314159)
        draws = 20_000
        for _ in range(draws):
            counts[tuple(sorted(_sample_kdpp(L, 2, rng)))] += 1
        empirical = np.array([counts[subset] / draws for subset in subsets])
        assert 0.5 * np.abs(empirical - expected).sum() < 0.025

    def test_distribution_matches_exhaustive_determinants(self):
        B = np.array([
            [1.0, 0.2, 0.0, 0.1, 0.0],
            [0.3, 0.9, 0.1, 0.0, 0.2],
            [0.0, 0.4, 1.1, 0.2, 0.0],
            [0.2, 0.0, 0.3, 0.8, 0.1],
            [0.1, 0.2, 0.0, 0.3, 0.7],
        ])
        L = B @ B.T
        subsets = list(combinations(range(5), 2))
        weights = np.array([np.linalg.det(L[np.ix_(subset, subset)]) for subset in subsets])
        expected = weights / weights.sum()
        counts = Counter()
        rng = np.random.default_rng(271828)
        draws = 30_000
        for _ in range(draws):
            counts[tuple(sorted(_sample_kdpp(L, 2, rng)))] += 1
        empirical = np.array([counts[subset] / draws for subset in subsets])
        assert 0.5 * np.abs(empirical - expected).sum() < 0.025
        assert np.max(np.abs(empirical - expected)) < 0.015

    def test_zero_probability_subsets_are_never_sampled(self):
        L = np.diag([1.0, 2.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        samples = {_sample_kdpp(L, 2, rng)[0] for _ in range(100)}
        assert samples <= {0, 1}
        for _ in range(100):
            assert set(_sample_kdpp(L, 2, rng)) == {0, 1}


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

    def test_map_short_result_raises_without_padding(self, candidates_12, monkeypatch):
        cands, embs = candidates_12
        monkeypatch.setattr(dpp_module, "_greedy_map", lambda L, k: [0])
        with pytest.raises(RuntimeError, match="wrong cardinality"):
            dpp_rerank(
                "q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="map"
            )

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
        """Greedy DPP should not return all 4 results from cluster A."""
        cands, embs = candidates_12
        result = dpp_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), mode="map")
        ids = {d["doc_id"] for d, _ in result}
        cluster_a = {0, 1, 2, 3}
        assert ids != cluster_a, "Greedy DPP returned all items from one cluster"

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
        with pytest.raises(ValueError, match="cannot exceed"):
            cluster_rerank("q", [], top_k=4, embed_fn=lambda t: np.zeros((0, 8)))

    def test_top_k_larger_than_candidates(self, candidates_12):
        cands, embs = candidates_12
        result = cluster_rerank("q", cands[:3], top_k=10, embed_fn=_make_embed_fn(embs[:3]),
                                n_clusters=2)
        assert len(result) == 3

    def test_invalid_method(self, candidates_12):
        cands, embs = candidates_12
        with pytest.raises(ValueError, match="method"):
            cluster_rerank("q", cands, top_k=4, embed_fn=_make_embed_fn(embs), method="bad")

    def test_label_permutation_invariance(self, monkeypatch):
        cands = [
            ({"doc_id": i, "text": f"d{i}"}, score)
            for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 1.0, 0.5])
        ]
        embs = np.eye(6, dtype=np.float32)
        partitions = iter([
            np.array([0, 0, 1, 1, 2, 2]),
            np.array([2, 2, 0, 0, 1, 1]),
        ])
        monkeypatch.setattr(clustering_module, "_cluster", lambda *args, **kwargs: next(partitions))
        first = cluster_rerank("q", cands, 5, _make_embed_fn(embs), n_clusters=3)
        second = cluster_rerank("q", cands, 5, _make_embed_fn(embs), n_clusters=3)
        assert [doc["doc_id"] for doc, _ in first] == [4, 0, 2, 5, 1]
        assert [doc["doc_id"] for doc, _ in first] == [
            doc["doc_id"] for doc, _ in second
        ]

    def test_truncated_round_uses_cluster_best_relevance(self, monkeypatch):
        cands = [
            ({"doc_id": i, "text": f"d{i}"}, score)
            for i, score in enumerate([0.4, 0.3, 1.0, 0.9, 0.8, 0.7])
        ]
        embs = np.eye(6, dtype=np.float32)
        monkeypatch.setattr(
            clustering_module, "_cluster",
            lambda *args, **kwargs: np.array([0, 0, 2, 2, 1, 1]),
        )
        result = cluster_rerank("q", cands, 5, _make_embed_fn(embs), n_clusters=3)
        assert [doc["doc_id"] for doc, _ in result] == [2, 4, 0, 3, 5]

    def test_unsorted_candidates_rank_members_by_relevance(self, monkeypatch):
        cands = [
            ({"doc_id": i, "text": f"d{i}"}, score)
            for i, score in enumerate([0.1, 0.2, 1.0, 0.9])
        ]
        embs = np.eye(4, dtype=np.float32)
        monkeypatch.setattr(
            clustering_module, "_cluster",
            lambda *args, **kwargs: np.array([0, 1, 0, 1]),
        )
        result = cluster_rerank("q", cands, 4, _make_embed_fn(embs), n_clusters=2)
        assert [doc["doc_id"] for doc, _ in result] == [2, 3, 0, 1]

    @pytest.mark.parametrize("method", ["kmeans", "agglomerative"])
    def test_obvious_fixed_clusters(self, method):
        embeddings = np.array([
            [1.0, 0.0], [0.99, 0.1],
            [-1.0, 0.0], [-0.99, -0.1],
            [0.0, 1.0], [0.1, 0.99],
        ])
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = _cluster(embeddings, n_clusters=3, method=method, seed=42)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[4] == labels[5]
        assert len({labels[0], labels[2], labels[4]}) == 3

    def test_kmeans_same_seed_is_reproducible(self, candidates_12):
        cands, embs = candidates_12
        first = cluster_rerank("q", cands, 5, _make_embed_fn(embs), n_clusters=3, seed=17)
        second = cluster_rerank("q", cands, 5, _make_embed_fn(embs), n_clusters=3, seed=17)
        assert [doc["doc_id"] for doc, _ in first] == [doc["doc_id"] for doc, _ in second]

    def test_kmeans_parameters_are_explicit(self, monkeypatch):
        captured = {}

        class FakeKMeans:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit_predict(self, embeddings):
                return np.array([0, 1])

        monkeypatch.setattr("sklearn.cluster.KMeans", FakeKMeans)
        _cluster(np.eye(2), 2, "kmeans", seed=17)
        assert captured == {
            "n_clusters": 2, "init": "k-means++", "n_init": 1,
            "random_state": 17, "algorithm": "lloyd", "max_iter": 300,
            "tol": 1e-4,
        }

    def test_agglomerative_parameters_are_explicit(self, monkeypatch):
        captured = {}

        class FakeAgglomerative:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit_predict(self, embeddings):
                return np.array([0, 1])

        monkeypatch.setattr("sklearn.cluster.AgglomerativeClustering", FakeAgglomerative)
        _cluster(np.eye(2), 2, "agglomerative", seed=42)
        assert captured["metric"] == "euclidean"
        assert captured["linkage"] == "ward"

    @pytest.mark.parametrize("n_clusters", [0, -1, 1.5, True])
    def test_invalid_cluster_count_raises(self, candidates_12, n_clusters):
        cands, embs = candidates_12
        with pytest.raises(ValueError, match="n_clusters"):
            cluster_rerank("q", cands, 4, _make_embed_fn(embs), n_clusters=n_clusters)

    def test_cluster_count_above_candidates_raises(self, candidates_12):
        cands, embs = candidates_12
        with pytest.raises(ValueError, match="cannot exceed"):
            cluster_rerank("q", cands[:2], 2, _make_embed_fn(embs[:2]), n_clusters=3)

    @pytest.mark.parametrize("top_k", [-1, 1.5, True])
    def test_invalid_top_k_raises(self, candidates_12, top_k):
        cands, embs = candidates_12
        with pytest.raises(ValueError, match="top_k"):
            cluster_rerank("q", cands, top_k, _make_embed_fn(embs), n_clusters=3)

    def test_top_k_zero_returns_empty_without_embedding_call(self, candidates_12):
        cands, _ = candidates_12
        result = cluster_rerank(
            "q", cands, 0,
            lambda texts: pytest.fail("embedding function should not be called"),
            n_clusters=3,
        )
        assert result == []

    @pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
    def test_nonfinite_relevance_raises(self, candidates_12, score):
        cands, embs = candidates_12
        bad = list(cands[:3])
        bad[1] = (bad[1][0], score)
        with pytest.raises(ValueError, match="relevance scores"):
            cluster_rerank("q", bad, 2, _make_embed_fn(embs[:3]), n_clusters=2)

    @pytest.mark.parametrize("embeddings,error", [
        (np.array([1.0, 2.0]), "2-D"),
        (np.ones((2, 2)), "one row per candidate"),
        (np.empty((3, 0)), "positive feature dimension"),
        (np.array([[1.0, 0.0], [np.nan, 1.0], [0.0, 1.0]]), "finite"),
        (np.array([[1.0, 0.0], [np.inf, 1.0], [0.0, 1.0]]), "finite"),
        (np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]), "nonzero row norms"),
    ])
    def test_invalid_embeddings_raise(self, candidates_12, embeddings, error):
        cands, _ = candidates_12
        with pytest.raises(ValueError, match=error):
            cluster_rerank("q", cands[:3], 2, lambda texts: embeddings, n_clusters=2)

    def test_missing_precomputed_id_raises(self, candidates_12):
        cands, embs = candidates_12
        precomputed = {0: embs[0], 1: embs[1]}
        with pytest.raises(KeyError, match="missing from precomputed_embs"):
            cluster_rerank(
                "q", cands[:3], 2, lambda texts: None,
                precomputed_embs=precomputed, n_clusters=2,
            )

    def test_effective_cluster_collapse_raises(self):
        from sklearn.exceptions import ConvergenceWarning

        cands = [({"doc_id": i, "text": f"d{i}"}, 3 - i) for i in range(3)]
        embeddings = np.ones((3, 2), dtype=np.float32)
        with pytest.warns(ConvergenceWarning):
            with pytest.raises(RuntimeError, match="effective clusters"):
                cluster_rerank("q", cands, 2, lambda texts: embeddings, n_clusters=2)


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

    @pytest.mark.parametrize("condition", ["kmeans_k0", "agglo_k0"])
    def test_zero_cluster_condition_raises(self, condition):
        with pytest.raises(ValueError, match="at least 1"):
            parse_condition(condition)

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
