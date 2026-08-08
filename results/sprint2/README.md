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

## Clustering and DPP (added after the initial MMR grid)

Deck Sub-tasks 2.2 and 2.3 -- both required for Sprint 2.

- **Stage 1 (retrieval-only, full grid):** 36 combinations, 0 errors.
- **Stage 2 (generation + faithfulness, representative slice):** 8 of those 36
  combinations (kmeans_k3, dpp_map, x4 retrievers), picked FROM Stage 1 results.
  See `conditions_with_generation` / `conditions_retrieval_only` in
  `EXPERIMENT_METADATA_sprint2_beir.json` for the exact split.

## Master summary now includes real comparison

`summary/sprint2_beir_master_summary.csv` (60 rows): `recall_delta_vs_baseline`,
`diversity_delta_vs_baseline`, `diversity_gained_per_recall_lost`, rankings,
best-condition flags. Baseline = that retriever's own `none` condition within
THIS SAME Sprint 2 BEIR grid.

## Sprint 1 is the project baseline -- for cross-dataset retriever comparison

`summary/sprint1_vs_sprint2_retriever_comparison.csv`. Sprint 1 (PubMedQA) is
the baseline against which retriever performance is measured across datasets.
DPR shows the largest jump (+0.38 recall) moving to BEIR HotpotQA, consistent
with DPR being NQ-trained and better domain-matched to open-domain content.
Sprint 1 is NOT used as the baseline in the diversity-tradeoff table above,
since PubMedQA has no diversity metric -- that baseline is each retriever's
own Sprint-2 `none` condition instead.

## Sprint 2 research question -- answered

"Is diversifying retrieved context actually helpful for LLM reasoning, or does
it introduce noise and hallucinations?" -- **It introduces noise. It did not
help in any of the 60 tested conditions.** See
`EXPERIMENT_METADATA_sprint2_beir.json`'s `research_question_answer` field.

## Graphs

`graphs/mmr/` -- 5 original MMR-lambda-sweep graphs.
`graphs/clustering_dpp/` -- 4 new graphs. Both folders contain a file named
`diversity_accuracy_tradeoff.png` -- different graphs, same filename. Check
the folder, not just the filename, when pulling figures into the report.
