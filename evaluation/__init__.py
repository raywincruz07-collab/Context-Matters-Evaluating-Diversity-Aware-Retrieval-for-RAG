"""
Evaluation metrics for RAG pipeline.

Phase 1 baseline metrics:
- Exact Match (EM)
- Token-level F1
- ROUGE-L
- Retrieval metrics: Recall@K, MRR
"""

import re
import string
from typing import List, Dict, Tuple
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, articles, extra whitespace."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    """Binary exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 score using longest common subsequence."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    # LCS length
    m, n = len(pred_tokens), len(truth_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == truth_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def retrieval_recall_at_k(retrieved_doc_ids: List[int], gold_doc_ids: List[int]) -> float:
    """What fraction of gold documents are in the retrieved set?"""
    if not gold_doc_ids:
        return 0.0
    retrieved_set = set(retrieved_doc_ids)
    hits = sum(1 for gid in gold_doc_ids if gid in retrieved_set)
    return hits / len(gold_doc_ids)


def mrr(retrieved_doc_ids: List[int], gold_doc_ids: List[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant document."""
    gold_set = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


def evaluate_single(
    prediction: str,
    ground_truth: str,
    retrieved_doc_ids: List[int],
    gold_doc_ids: List[int],
) -> Dict[str, float]:
    """Evaluate a single QA example."""
    return {
        "exact_match": exact_match(prediction, ground_truth),
        "f1": token_f1(prediction, ground_truth),
        "rouge_l": rouge_l(prediction, ground_truth),
        "recall_at_k": retrieval_recall_at_k(retrieved_doc_ids, gold_doc_ids),
        "mrr": mrr(retrieved_doc_ids, gold_doc_ids),
    }


def evaluate_batch(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregate evaluation over a batch.

    Args:
        results: List of dicts, each with keys from evaluate_single()

    Returns:
        Dict of averaged metric values
    """
    if not results:
        return {}

    metrics = {}
    for key in results[0]:
        values = [r[key] for r in results]
        metrics[f"avg_{key}"] = sum(values) / len(values)

    return metrics
