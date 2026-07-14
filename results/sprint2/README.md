# Sprint 2 Results — BEIR HotpotQA + MMR Diversification

## The authoritative results table

`summary/sprint2_beir_master_summary.csv` is **the** results table.  
It has **24 rows** = 4 retrievers × 6 MMR conditions, 500 questions each, 0 errors.

## Raw per-condition CSVs

`raw/` holds the 24 per-condition CSVs (`sprint2_beir_<retriever>_<condition>_top5.csv`),
each with **500 rows** — one per question.

## ⚠ Do not report exact_match, f1, or rouge_l

**`exact_match`, `f1`, and `rouge_l` are all exactly 0.0 in every file and must
never be reported.**

BEIR HotpotQA qrels carry no free-text reference answers. These columns are filled
with zeros rather than nulls, so they silently pass any "present / non-null" check.
Reporting them would read as total system failure rather than "not applicable."

**The meaningful metrics are:** `recall_at_k`, `mrr`, `retrieval_diversity`, `faithfulness_nli`.

## Stale artifact

`raw/PARTIAL_colbertv2_only_summary_DO_NOT_USE.csv` is kept for provenance only.
It contains the 6-row ColBERTv2-only summary produced during the final resume run —
not the full 24-combination grid. Use `summary/sprint2_beir_master_summary.csv` instead.

## Experiment metadata

`EXPERIMENT_METADATA_sprint2_beir.json` records the full grid parameters:
seed=42, sample_size=500, negatives_per_query=50, top_k=5, mmr_candidate_pool=20,
generator=qwen3.5-122b, temperature=0.0, true_error_count=0.
