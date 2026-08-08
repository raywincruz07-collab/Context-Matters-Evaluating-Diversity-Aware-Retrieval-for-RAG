"""
Unit tests for MMR reranking (Task 3).

Run from project root:
    cd src && python -m pytest ../tests/test_mmr.py -v

Tests assert the boundary conditions required by the spec:
  - lambda=1.0 reduces to pure top-k-by-relevance
  - lambda=0.0 reduces to maximal pairwise distance selection

All tests use mocked embeddings — no models are loaded.
"""

import sys
import os

# Make src/ importable when pytest runs from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from diversification.mmr import mmr_rerank
from diversification.dispatch import parse_condition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidates(n: int, scores=None):
    """Build n fake candidates with doc_id=0..n-1 and optional score list."""
    if scores is None:
        scores = [float(n - i) for i in range(n)]  # n, n-1, ..., 1
    return [
        ({"text": f"doc_{i}", "doc_id": i}, scores[i])
        for i in range(n)
    ]


def _unit_embed_fn(embeddings_map: dict):
    """Return a mock embed_fn that looks up embeddings by text."""
    def embed_fn(texts):
        rows = [embeddings_map[t] for t in texts]
        return np.array(rows, dtype=np.float32)
    return embed_fn


# ---------------------------------------------------------------------------
# lambda=1.0 — pure relevance ranking
# ---------------------------------------------------------------------------

class TestLambdaOne:
    """lambda=1.0 must produce the same order as sorting by relevance score."""

    def test_order_matches_relevance_descending(self):
        # Candidates with descending relevance: A(0.9) > B(0.5) > C(0.3)
        candidates = [
            ({"text": "doc A", "doc_id": 0}, 0.9),
            ({"text": "doc B", "doc_id": 1}, 0.5),
            ({"text": "doc C", "doc_id": 2}, 0.3),
        ]
        # Embeddings are arbitrary — diversity term is zeroed by lambda=1.0
        rng = np.random.default_rng(42)
        fixed_embs = {
            "doc A": rng.random(4).astype(np.float32),
            "doc B": rng.random(4).astype(np.float32),
            "doc C": rng.random(4).astype(np.float32),
        }
        # L2-normalise
        for k in fixed_embs:
            fixed_embs[k] /= np.linalg.norm(fixed_embs[k])

        result = mmr_rerank(
            "query", candidates, top_k=3, lambda_param=1.0,
            embed_fn=_unit_embed_fn(fixed_embs),
        )

        assert len(result) == 3
        result_ids = [doc["doc_id"] for doc, _ in result]
        assert result_ids == [0, 1, 2], (
            f"lambda=1.0 must order by relevance: expected [0,1,2] got {result_ids}"
        )

    def test_order_with_non_monotone_scores(self):
        # Scores not in natural index order
        candidates = [
            ({"text": "low",  "doc_id": 0}, 0.2),
            ({"text": "high", "doc_id": 1}, 0.9),
            ({"text": "mid",  "doc_id": 2}, 0.5),
        ]
        embs = {
            "low":  np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "high": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "mid":  np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        result = mmr_rerank(
            "q", candidates, top_k=3, lambda_param=1.0,
            embed_fn=_unit_embed_fn(embs),
        )
        result_ids = [doc["doc_id"] for doc, _ in result]
        # Expected: high(1), mid(2), low(0)
        assert result_ids == [1, 2, 0], (
            f"lambda=1.0 must order by relevance desc: expected [1,2,0] got {result_ids}"
        )

    def test_top_k_truncation(self):
        candidates = _make_candidates(5)
        embs = {f"doc_{i}": np.eye(5, dtype=np.float32)[i] for i in range(5)}
        result = mmr_rerank(
            "q", candidates, top_k=3, lambda_param=1.0,
            embed_fn=_unit_embed_fn(embs),
        )
        assert len(result) == 3
        # First 3 by relevance: 0, 1, 2
        assert [doc["doc_id"] for doc, _ in result] == [0, 1, 2]

    def test_unsorted_negative_scores_follow_relevance(self):
        candidates = _make_candidates(3, scores=[-5.0, -1.0, -3.0])
        embs = {f"doc_{i}": np.eye(3, dtype=np.float32)[i] for i in range(3)}
        result = mmr_rerank(
            "q", candidates, top_k=3, lambda_param=1.0,
            embed_fn=_unit_embed_fn(embs),
        )
        assert [doc["doc_id"] for doc, _ in result] == [1, 2, 0]


# ---------------------------------------------------------------------------
# lambda=0.0 — pure diversity maximisation
# ---------------------------------------------------------------------------

class TestLambdaZero:
    """
    lambda=0.0 must maximise pairwise diversity.

    Setup: A and C are nearly collinear (high cosine sim); B is orthogonal to A.
    Expected selection order:
      1. A  (index 0, all MMR scores tie at 0 for first pick)
      2. B  (sim to A ≈ 0  →  MMR score = -0 = 0)  vs C (sim to A ≈ high → score negative)
      3. C  (only candidate left)
    """

    @pytest.fixture
    def setup(self):
        candidates = [
            ({"text": "A", "doc_id": 0}, 1.0),
            ({"text": "B", "doc_id": 1}, 1.0),
            ({"text": "C", "doc_id": 2}, 1.0),
        ]
        # A and C nearly identical direction; B orthogonal to A
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        c_raw = np.array([0.99, 0.0, 0.141, 0.0], dtype=np.float32)
        c = c_raw / np.linalg.norm(c_raw)
        embs = {"A": a, "B": b, "C": c}
        return candidates, embs

    def test_first_pick_is_index_zero(self, setup):
        candidates, embs = setup
        result = mmr_rerank(
            "q", candidates, top_k=1, lambda_param=0.0,
            embed_fn=_unit_embed_fn(embs),
        )
        assert len(result) == 1
        assert result[0][0]["doc_id"] == 0, (
            "lambda=0.0 first pick must be index 0 (deterministic tie-break)"
        )

    def test_second_pick_is_most_diverse(self, setup):
        candidates, embs = setup
        result = mmr_rerank(
            "q", candidates, top_k=2, lambda_param=0.0,
            embed_fn=_unit_embed_fn(embs),
        )
        assert len(result) == 2
        assert result[0][0]["doc_id"] == 0  # A first
        assert result[1][0]["doc_id"] == 1  # B second (most different from A)

    def test_full_diversity_order(self, setup):
        candidates, embs = setup
        result = mmr_rerank(
            "q", candidates, top_k=3, lambda_param=0.0,
            embed_fn=_unit_embed_fn(embs),
        )
        result_ids = [doc["doc_id"] for doc, _ in result]
        assert result_ids == [0, 1, 2], (
            f"lambda=0.0 full order expected [0,1,2] (A,B,C) got {result_ids}"
        )

    def test_unsorted_candidates_are_relevance_seeded(self):
        candidates = _make_candidates(3, scores=[0.1, 0.5, 1.0])
        embs = {
            "doc_0": np.array([1.0, 0.0]),
            "doc_1": np.array([0.0, 1.0]),
            "doc_2": np.array([1.0, 0.0]),
        }
        result = mmr_rerank(
            "q", candidates, top_k=3, lambda_param=0.0,
            embed_fn=_unit_embed_fn(embs),
        )
        assert [doc["doc_id"] for doc, _ in result] == [2, 1, 0]


class TestKnownObjective:
    @pytest.mark.parametrize("lambda_param,expected", [
        (0.25, [0, 2, 1]),
        (0.5, [0, 2, 1]),
        (0.75, [0, 1, 2]),
    ])
    def test_hand_calculated_selection(self, lambda_param, expected):
        # Normalised relevance is exactly [1, 0.5, 0].  After selecting doc 0,
        # similarities are sim(1, 0)=0.8 and sim(2, 0)=0.
        candidates = _make_candidates(3, scores=[3.0, 2.0, 1.0])
        embs = {
            "doc_0": np.array([1.0, 0.0]),
            "doc_1": np.array([0.8, 0.6]),
            "doc_2": np.array([0.0, 1.0]),
        }
        result = mmr_rerank(
            "q", candidates, top_k=3, lambda_param=lambda_param,
            embed_fn=_unit_embed_fn(embs),
        )
        assert [doc["doc_id"] for doc, _ in result] == expected


# ---------------------------------------------------------------------------
# Precomputed embeddings path
# ---------------------------------------------------------------------------

class TestPrecomputedEmbs:
    """mmr_rerank should accept precomputed_embs and not call embed_fn."""

    def test_precomputed_embs_used_not_embed_fn(self):
        candidates = [
            ({"text": "d0", "doc_id": 10}, 0.8),
            ({"text": "d1", "doc_id": 11}, 0.6),
            ({"text": "d2", "doc_id": 12}, 0.4),
        ]
        precomputed = {
            10: np.array([1.0, 0.0], dtype=np.float32),
            11: np.array([0.0, 1.0], dtype=np.float32),
            12: np.array([-1.0, 0.0], dtype=np.float32),
        }

        def should_not_call(texts):
            raise AssertionError("embed_fn must NOT be called when precomputed_embs is provided")

        # Should complete without raising
        result = mmr_rerank(
            "q", candidates, top_k=3, lambda_param=0.5,
            embed_fn=should_not_call,
            precomputed_embs=precomputed,
        )
        assert len(result) == 3

    def test_precomputed_and_on_demand_are_equivalent(self):
        candidates = _make_candidates(3, scores=[0.9, 0.6, 0.2])
        vectors = {
            0: np.array([2.0, 0.0], dtype=np.float32),
            1: np.array([1.0, 1.0], dtype=np.float32),
            2: np.array([0.0, 3.0], dtype=np.float32),
        }
        on_demand = {f"doc_{i}": vectors[i] for i in range(3)}
        from_fn = mmr_rerank(
            "q", candidates, 3, 0.5, _unit_embed_fn(on_demand)
        )
        from_precomputed = mmr_rerank(
            "q", candidates, 3, 0.5, lambda texts: None,
            precomputed_embs=vectors,
        )
        assert [doc["doc_id"] for doc, _ in from_fn] == [
            doc["doc_id"] for doc, _ in from_precomputed
        ]

    def test_duplicate_candidates_remain_deterministic(self):
        candidates = _make_candidates(3, scores=[3.0, 2.0, 1.0])
        candidates[1][0]["doc_id"] = 0
        precomputed = {
            0: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
        }
        result = mmr_rerank(
            "q", candidates, 3, 0.5, lambda texts: None,
            precomputed_embs=precomputed,
        )
        assert [doc["text"] for doc, _ in result] == ["doc_0", "doc_2", "doc_1"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_candidates(self):
        result = mmr_rerank("q", [], top_k=5, lambda_param=0.5,
                            embed_fn=lambda t: np.array([]))
        assert result == []

    def test_top_k_larger_than_candidates(self):
        candidates = _make_candidates(2)
        embs = {f"doc_{i}": np.eye(2, dtype=np.float32)[i] for i in range(2)}
        result = mmr_rerank(
            "q", candidates, top_k=10, lambda_param=0.5,
            embed_fn=_unit_embed_fn(embs),
        )
        assert len(result) == 2  # capped at len(candidates)

    def test_single_candidate(self):
        candidates = [({"text": "only", "doc_id": 0}, 1.0)]
        embs = {"only": np.array([1.0, 0.0], dtype=np.float32)}
        result = mmr_rerank(
            "q", candidates, top_k=5, lambda_param=0.5,
            embed_fn=_unit_embed_fn(embs),
        )
        assert len(result) == 1
        assert result[0][0]["doc_id"] == 0

    def test_all_equal_scores_lambda_one(self):
        # All relevance scores equal → normalized to 1.0 → order is by index (first picked first)
        candidates = [({"text": f"d{i}", "doc_id": i}, 0.5) for i in range(3)]
        embs = {f"d{i}": np.eye(3, dtype=np.float32)[i] for i in range(3)}
        result = mmr_rerank(
            "q", candidates, top_k=3, lambda_param=1.0,
            embed_fn=_unit_embed_fn(embs),
        )
        # All norm scores are 1.0 (max==min edge case), tie-break by first argmax = index 0
        assert len(result) == 3

    def test_negative_scores_are_minmax_normalised(self):
        candidates = _make_candidates(3, scores=[-5.0, -3.0, -1.0])
        embs = {f"doc_{i}": np.eye(3, dtype=np.float32)[i] for i in range(3)}
        result = mmr_rerank("q", candidates, 3, 1.0, _unit_embed_fn(embs))
        assert [score for _, score in result] == [1.0, 0.5, 0.0]

    def test_equal_scores_remain_equal_after_normalisation(self):
        candidates = _make_candidates(3, scores=[-2.0, -2.0, -2.0])
        embs = {f"doc_{i}": np.eye(3, dtype=np.float32)[i] for i in range(3)}
        result = mmr_rerank("q", candidates, 3, 1.0, _unit_embed_fn(embs))
        assert [score for _, score in result] == [1.0, 1.0, 1.0]

    def test_positive_affine_score_transform_preserves_selection(self):
        scores = [-4.0, 2.0, 1.0]
        embs = {f"doc_{i}": np.eye(3, dtype=np.float32)[i] for i in range(3)}
        original = mmr_rerank(
            "q", _make_candidates(3, scores), 3, 0.6, _unit_embed_fn(embs)
        )
        transformed = mmr_rerank(
            "q", _make_candidates(3, [3.0 * x + 7.0 for x in scores]),
            3, 0.6, _unit_embed_fn(embs),
        )
        assert [doc["doc_id"] for doc, _ in original] == [
            doc["doc_id"] for doc, _ in transformed
        ]

    @pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
    def test_nonfinite_relevance_score_raises(self, score):
        candidates = _make_candidates(2, scores=[1.0, score])
        with pytest.raises(ValueError, match="relevance scores"):
            mmr_rerank("q", candidates, 1, 0.5, lambda texts: np.ones((2, 2)))

    @pytest.mark.parametrize("lambda_param", [-0.1, 1.1, np.nan, np.inf, -np.inf])
    def test_invalid_lambda_raises(self, lambda_param):
        with pytest.raises(ValueError, match="lambda_param"):
            mmr_rerank("q", [], 0, lambda_param, lambda texts: np.empty((0, 1)))

    def test_dispatch_rejects_invalid_lambda(self):
        with pytest.raises(ValueError, match="MMR lambda"):
            parse_condition("mmr_1.5")

    @pytest.mark.parametrize("top_k", [-1, 1.5, True])
    def test_invalid_top_k_raises(self, top_k):
        with pytest.raises(ValueError, match="top_k"):
            mmr_rerank("q", [], top_k, 0.5, lambda texts: np.empty((0, 1)))

    def test_top_k_zero_validates_scores_without_embedding_call(self):
        candidates = _make_candidates(2, scores=[1.0, np.nan])
        with pytest.raises(ValueError, match="relevance scores"):
            mmr_rerank(
                "q", candidates, 0, 0.5,
                lambda texts: pytest.fail("embedding function should not be called"),
            )

    def test_top_k_zero_returns_empty_without_embedding_call(self):
        candidates = _make_candidates(2, scores=[1.0, 0.0])
        result = mmr_rerank(
            "q", candidates, 0, 0.5,
            lambda texts: pytest.fail("embedding function should not be called"),
        )
        assert result == []

    @pytest.mark.parametrize("embeddings,error", [
        (np.array([1.0, 2.0]), "2-D"),
        (np.ones((1, 2)), "one row per candidate"),
        (np.empty((2, 0)), "positive feature dimension"),
        (np.array([[1.0, 0.0], [np.nan, 1.0]]), "finite"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), "nonzero row norms"),
    ])
    def test_invalid_embeddings_raise(self, embeddings, error):
        candidates = _make_candidates(2)
        with pytest.raises(ValueError, match=error):
            mmr_rerank("q", candidates, 1, 0.5, lambda texts: embeddings)
