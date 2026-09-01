# Sprint 3 Project State — 2026-09-01

## Project
Context Matters — Evaluating Diversity-Aware Retrieval for RAG

Branch: `sprint3`

Checkpoint source HEAD:
`bba0be298d7209aa709607b2102469b9dfb46956`

## Research Question
Does diversifying retrieved context improve LLM reasoning, or does it introduce noise and hallucinations?

## Canonical Setup
Datasets: PubMedQA, HotpotQA, ASQA

Retrievers:
- BM25
- DPR
- Contriever
- ColBERTv2

Retrieval:
- candidate pool = Top-20
- selected context = Top-5

Generators:
- Llama 3.3 70B
- Gemma 4 26B
- Qwen3.6 36B

Generation:
- temperature = 0
- direct / non-thinking
- one canonical replica
- max 3 infrastructure attempts
- no semantic/content retries

## PubMedQA Baseline — COMPLETE
Evidence role:
`HISTORICAL_OBSERVED_CONTROL_REPLICATION`

Population:
`N = 1000`

Generation matrix:
- WITHOUT_CONTEXT
- BM25
- DPR
- Contriever
- ColBERTv2
- 3 generators
- 15 conditions
- 15,000 generated answers

Generation status:
- OK = 14,966
- PARSE_FAILURE = 33
- TRUNCATED = 1
- REFUSAL = 0
- ERROR = 0

Semantic failures are preserved and are not regenerated.

## PubMedQA Retrieval
Historical Sprint 1 full-1000 results:

| Retriever | Recall@5 | MRR@5 | Recall@20 |
|---|---:|---:|---:|
| BM25 | 0.5992 | 0.9115 | 0.7024 |
| DPR | 0.3253 | 0.5787 | 0.4653 |
| Contriever | 0.7404 | 0.9721 | 0.8367 |
| ColBERTv2 | 0.7382 | 0.9762 | 0.8253 |

Historical Sprint 1 evidence remains separate and immutable.

## PubMedQA Answer Accuracy
Llama:
- no context 0.571
- BM25 0.656
- DPR 0.579
- Contriever 0.691
- ColBERTv2 0.708

Gemma:
- no context 0.4599
- BM25 0.4038
- DPR 0.3193
- Contriever 0.461
- ColBERTv2 0.496

Qwen3.6:
- no context 0.559
- BM25 0.596
- DPR 0.4975
- Contriever 0.653
- ColBERTv2 0.673

Main finding:
Retrieved context is not automatically beneficial. RAG behavior depends on retrieval quality × generator behaviour.

## Derived Analysis
Directory:
`results/sprint3/analysis/pubmedqa_baseline/`

Files:
- decision_accuracy_summary.csv
- decision_class_recall.csv
- generation_correctness_per_query.parquet
- generation_status_summary.csv
- paired_context_effects.csv
- retrieval_answer_summary.csv

## Professor Notebook
`notebooks/01_pubmedqa_baseline.ipynb`

Restart Kernel → Run All completed successfully.

The notebook:
- uses saved evidence
- makes zero LLM/API generation calls
- reproduces retrieval, correctness, paired effects, diagnostics and final audit
- passed all final audit checks

## Physical Backups

Generation checkpoint:
`/workspace/pubmedqa-final-baseline-15blocks-2026-08-31.tar.gz`

SHA256:
`d22c65cc4e7d32d781f6dcade7a7f2d66dd3fd726d35375405522ba1cdc2ebd7`

Final notebook backup:
`/workspace/notebook-backups/01_pubmedqa_baseline-final-runall-2026-09-01.ipynb`

SHA256:
`d918091fa006754a290637f76648d4c2634528a73ed256fba64ab0cce0dd6f1a`

Complete checkpoint:
`/workspace/context-matters-pubmedqa-complete-2026-09-01.tar.gz`

SHA256:
`00dcfc995fa2e0732c290a0e225ddfe29afa8edd309fe1de5c80a82d22ea4421`

## Remaining Sprint 3 Work
Still required:
- retrieval diversity
- faithfulness / hallucination
- diversified generation
- ACC_bi
- diversity–accuracy analysis
- diversity–hallucination analysis
- context-to-output change analysis

## Frozen Diversification Conditions
Reference:
`relevance_baseline`

Treatments:
- mmr_lambda_0
- mmr_lambda_0_25
- mmr_lambda_0_50
- mmr_lambda_0_75
- kmeans_selected
- agglo_selected
- dpp_map

Do not add Spectral, xQuAD or stochastic DPP seeds to the canonical confirmatory set.

## ASQA Next
Evidence roles:
- DEVELOPMENT = 3482
- SELECTION = 871
- PROJECT_PROTECTED_FINAL = dev948

Protected dev948 must remain unopened until the applicable gates are frozen.

ASQA retrieval metrics:
- SRecall@5
- alpha-nDCG@5
- corpus coverability
- c*

ASQA primary answer correctness:
- official Disambig-F1 / QA-F1

Execution:
- Terminal 1: control / Git / provenance
- Terminal 2: ASQA data / corpus
- Terminal 3: ASQA implementation / tests

Avoid simultaneous GPU-heavy jobs on the same RTX 5090.

ASQA professor-facing notebook follows after ASQA baseline execution.
HotpotQA follows after ASQA.
