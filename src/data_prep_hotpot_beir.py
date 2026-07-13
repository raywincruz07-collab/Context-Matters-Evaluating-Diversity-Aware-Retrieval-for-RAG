"""
Data preparation for HotpotQA via BEIR (BeIR/hotpotqa on HuggingFace) — Sprint 2.

Option B subsampling strategy: the full BEIR HotpotQA corpus contains 5.23M passages,
which is too large to index on Colab. We sample 500 test queries (seed=42), collect ALL
gold passages for those queries from the full corpus (fetched by BEIR _id), then add
exactly 50 random negative passages per query drawn from the full corpus excluding any
passage that is a gold doc for any of the 500 sampled queries. This produces a corpus of
~26–30 k documents that is manageable on Colab while preserving realistic retrieval
difficulty (negatives are drawn from 5.23M, not the 10-passage distractor pool).

This replaces data_prep_hotpot.py for Sprint 2. It does NOT overwrite any Sprint 1 or
distractor-version files (corpus.json, qa_pairs.json, hotpot_corpus.json, hotpot_qa_pairs.json).
"""

import json
import os
import random
from typing import Dict, List, Set, Tuple

from config import (
    DATA_DIR,
    HOTPOT_BEIR_CORPUS_PATH,
    HOTPOT_BEIR_NEGATIVES_PER_QUERY,
    HOTPOT_BEIR_QA_PAIRS_PATH,
    HOTPOT_BEIR_SAMPLE_SIZE,
    HOTPOT_BEIR_SEED,
)

# Test split has 7,405 queries per BEIR benchmark.
_BEIR_TEST_SPLIT_SIZE = 7405


def build_hotpot_beir_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Build corpus and qa_pairs from the BEIR version of HotpotQA.

    Loads corpus, queries, and qrels directly from HuggingFace (no local download needed).
    Uses fixed seed HOTPOT_BEIR_SEED everywhere randomness is involved.

    Returns:
        corpus   — list of document dicts matching the pipeline schema
        qa_pairs — list of QA pair dicts matching the pipeline schema

    Raises:
        AssertionError  if any gold_doc_id is absent from the built corpus
        RuntimeError    if a gold BEIR passage _id cannot be found in the corpus dataset
    """
    from datasets import load_dataset

    # ── 1. Load all three BEIR components from HuggingFace ──────────────────
    print("Loading BeIR/hotpotqa corpus from HuggingFace …")
    corpus_ds = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
    print(f"  Full corpus size: {len(corpus_ds):,} passages")

    print("Loading BeIR/hotpotqa queries from HuggingFace …")
    queries_ds = load_dataset("BeIR/hotpotqa", "queries", split="queries")
    print(f"  Total queries: {len(queries_ds):,}")

    print("Loading BeIR/hotpotqa-qrels from HuggingFace …")
    qrels_ds = load_dataset("BeIR/hotpotqa-qrels", split="test")
    print(f"  Qrels (test split): {len(qrels_ds):,} rows")

    # ── 2. Build lookup maps ─────────────────────────────────────────────────
    # query_id → query text
    qid_to_text: Dict[str, str] = {row["_id"]: row["text"] for row in queries_ds}

    # corpus _id → {title, text} (needed for gold lookup and negative sampling)
    print("Indexing corpus by _id … (this may take ~30 s for 5.23 M passages)")
    beir_id_to_doc: Dict[str, Dict] = {}
    for row in corpus_ds:
        beir_id_to_doc[str(row["_id"])] = {"title": row["title"], "text": row["text"]}
    print(f"  Corpus index built: {len(beir_id_to_doc):,} entries")

    # query_id → list of gold corpus _ids (score >= 1)
    qid_to_gold_beir_ids: Dict[str, List[str]] = {}
    for row in qrels_ds:
        if int(row["score"]) >= 1:
            qid_to_gold_beir_ids.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))

    # ── 3. Sample 500 queries from the test split (seed=42) ──────────────────
    # qrels test split gives us the test query ids
    test_qids: List[str] = sorted(set(str(row["query-id"]) for row in qrels_ds))
    print(f"Unique test query ids in qrels: {len(test_qids)}")

    rng = random.Random(HOTPOT_BEIR_SEED)
    print(f"[seed={HOTPOT_BEIR_SEED}] Sampling {HOTPOT_BEIR_SAMPLE_SIZE} queries …")
    sampled_qids: List[str] = sorted(
        rng.sample(test_qids, min(HOTPOT_BEIR_SAMPLE_SIZE, len(test_qids)))
    )
    print(f"  Sampled {len(sampled_qids)} queries")

    # ── 4. Collect ALL gold BEIR passage ids across sampled queries ──────────
    all_gold_beir_ids: Set[str] = set()
    for qid in sampled_qids:
        for bid in qid_to_gold_beir_ids.get(qid, []):
            all_gold_beir_ids.add(bid)
    print(f"  Unique gold passages across all sampled queries: {len(all_gold_beir_ids)}")

    # Verify every gold passage exists in the corpus; fail loudly if not
    missing_gold = all_gold_beir_ids - beir_id_to_doc.keys()
    if missing_gold:
        raise RuntimeError(
            f"{len(missing_gold)} gold BEIR passage id(s) not found in the corpus dataset: "
            f"{sorted(missing_gold)[:10]} … (showing first 10)"
        )

    # ── 5. Sample negatives: 50 per query from full corpus excluding all golds ─
    non_gold_beir_ids: List[str] = sorted(
        bid for bid in beir_id_to_doc.keys() if bid not in all_gold_beir_ids
    )
    total_negatives_needed = HOTPOT_BEIR_NEGATIVES_PER_QUERY * len(sampled_qids)
    print(
        f"[seed={HOTPOT_BEIR_SEED}] Sampling {total_negatives_needed:,} negatives "
        f"({HOTPOT_BEIR_NEGATIVES_PER_QUERY} × {len(sampled_qids)} queries) "
        f"from {len(non_gold_beir_ids):,} eligible passages …"
    )

    # Sample a flat pool then assign HOTPOT_BEIR_NEGATIVES_PER_QUERY per query.
    # We sample (negatives_per_query × n_queries) unique passages, then slice.
    # If the pool is smaller than needed, take everything available.
    flat_neg_pool_size = min(total_negatives_needed, len(non_gold_beir_ids))
    flat_neg_pool: List[str] = rng.sample(non_gold_beir_ids, flat_neg_pool_size)

    # Assign negatives round-robin style so each query gets a distinct slice
    query_negatives: Dict[str, List[str]] = {}
    for q_idx, qid in enumerate(sampled_qids):
        start = q_idx * HOTPOT_BEIR_NEGATIVES_PER_QUERY
        end = start + HOTPOT_BEIR_NEGATIVES_PER_QUERY
        query_negatives[qid] = flat_neg_pool[start:end]

    # ── 6. Build corpus: gold passages + per-query negatives ─────────────────
    # Collect the unique set of passages needed
    corpus_beir_ids: Set[str] = set(all_gold_beir_ids)
    for neg_list in query_negatives.values():
        corpus_beir_ids.update(neg_list)

    # Assign sequential doc_ids (sorted for determinism)
    beir_id_to_doc_id: Dict[str, int] = {
        bid: idx for idx, bid in enumerate(sorted(corpus_beir_ids))
    }

    corpus: List[Dict] = []
    for bid, doc_id in sorted(beir_id_to_doc_id.items(), key=lambda x: x[1]):
        raw = beir_id_to_doc[bid]
        corpus.append(
            {
                "doc_id": doc_id,
                "title": raw["title"],
                "text": raw["text"],
                "beir_id": bid,   # kept for traceability
            }
        )

    # ── 7. Build QA pairs ────────────────────────────────────────────────────
    # answer is intentionally empty string — BEIR HotpotQA qrels do not include
    # free-text answers.  Downstream evaluation must use recall@k / diversity
    # metrics only.  Do NOT compute F1 or ROUGE-L against this empty reference.
    qa_pairs: List[Dict] = []
    for qa_idx, qid in enumerate(sampled_qids):
        gold_beir_ids = qid_to_gold_beir_ids.get(qid, [])

        # Map gold BEIR _ids to integer doc_ids in our corpus
        gold_doc_ids: List[int] = []
        missing_for_query: List[str] = []
        for bid in gold_beir_ids:
            if bid in beir_id_to_doc_id:
                gold_doc_ids.append(beir_id_to_doc_id[bid])
            else:
                missing_for_query.append(bid)

        if missing_for_query:
            # We already checked all_gold_beir_ids above, so this path should
            # never trigger — but if the per-query qrels reference an id not in
            # all_gold_beir_ids it means the qrels data is inconsistent.
            raise AssertionError(
                f"qa_id={qa_idx} (beir_query_id={qid}): gold BEIR passage ids "
                f"{missing_for_query} not mapped to any corpus doc_id. "
                "This indicates a qrels/corpus inconsistency."
            )

        qa_pairs.append(
            {
                "qa_id": qa_idx,
                "question": qid_to_text.get(qid, ""),
                "answer": "",   # see comment above — do not use for F1/ROUGE-L
                "gold_doc_ids": gold_doc_ids,
                "beir_query_id": qid,   # kept for traceability
            }
        )

    # ── 8. Integrity assertion: every gold_doc_id must exist in corpus ────────
    corpus_id_set: Set[int] = {doc["doc_id"] for doc in corpus}
    for qa in qa_pairs:
        for gid in qa["gold_doc_ids"]:
            if gid not in corpus_id_set:
                raise AssertionError(
                    f"Integrity violation: gold_doc_id={gid} in qa_id={qa['qa_id']} "
                    f"(beir_query_id={qa['beir_query_id']}) is not present in the "
                    f"built corpus (corpus size={len(corpus)}). "
                    "This should never happen — investigate corpus build logic."
                )

    # ── 9. Print summary ─────────────────────────────────────────────────────
    avg_gold = (
        sum(len(qa["gold_doc_ids"]) for qa in qa_pairs) / len(qa_pairs)
        if qa_pairs else 0.0
    )
    subsampling_summary = (
        f"BEIR HotpotQA (Option B): {len(sampled_qids)} test queries sampled "
        f"(seed={HOTPOT_BEIR_SEED}) from 7,405 test queries; corpus built from "
        f"{len(all_gold_beir_ids)} unique gold passages + "
        f"{HOTPOT_BEIR_NEGATIVES_PER_QUERY} random negatives per query "
        f"(excluded from gold set, drawn from 5.23 M-passage BEIR corpus); "
        f"total corpus size = {len(corpus):,} documents."
    )

    print()
    print("=" * 70)
    print(f"Total corpus size            : {len(corpus):,} docs")
    print(f"Number of QA pairs           : {len(qa_pairs)}")
    print(f"Average gold docs per query  : {avg_gold:.2f}")
    print(f"Negatives per query          : {HOTPOT_BEIR_NEGATIVES_PER_QUERY}")
    print(f"Seed used                    : {HOTPOT_BEIR_SEED}")
    print()
    print("Subsampling summary (copy into Sprint 2 report methods section):")
    print(subsampling_summary)
    print("=" * 70)

    return corpus, qa_pairs


def save_hotpot_beir_data(corpus: List[Dict], qa_pairs: List[Dict]) -> None:
    """Save corpus and qa_pairs to the BEIR-specific JSON files."""
    with open(HOTPOT_BEIR_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved corpus to   {HOTPOT_BEIR_CORPUS_PATH}")

    with open(HOTPOT_BEIR_QA_PAIRS_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"Saved qa_pairs to {HOTPOT_BEIR_QA_PAIRS_PATH}")


def load_hotpot_beir_data() -> Tuple[List[Dict], List[Dict]]:
    """Load previously saved BEIR corpus and qa_pairs from disk."""
    for path in (HOTPOT_BEIR_CORPUS_PATH, HOTPOT_BEIR_QA_PAIRS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BEIR HotpotQA data not found at {path}. "
                "Run prepare_hotpot_beir_data() or build_hotpot_beir_data() first."
            )
    with open(HOTPOT_BEIR_CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(HOTPOT_BEIR_QA_PAIRS_PATH, encoding="utf-8") as f:
        qa_pairs = json.load(f)
    return corpus, qa_pairs


def prepare_hotpot_beir_data() -> Tuple[List[Dict], List[Dict]]:
    """Load cached BEIR data if it exists, otherwise build and cache it."""
    if os.path.exists(HOTPOT_BEIR_CORPUS_PATH) and os.path.exists(HOTPOT_BEIR_QA_PAIRS_PATH):
        print("Loading cached BEIR HotpotQA data from disk …")
        return load_hotpot_beir_data()

    corpus, qa_pairs = build_hotpot_beir_data()
    save_hotpot_beir_data(corpus, qa_pairs)
    return corpus, qa_pairs


if __name__ == "__main__":
    prepare_hotpot_beir_data()
