# Context Matters RAG — Project State and Continuation Record

Last updated: 2026-08-24. This file is the permanent handoff record for
future project sessions. Update it when a governed checkpoint changes; do not
rewrite historical evidence.

## Project objective

Determine when diversity-aware retrieval improves RAG and when it harms RAG by
introducing irrelevant evidence, noise, or hallucination. The project compares
focused evidence, multi-hop/complementary evidence, and ambiguous/multi-aspect
information needs. Retrieval diversity is not itself success: retrieval
effectiveness, answer correctness, faithfulness, and answer-content change are
separate constructs.

## Authority and document order

Apply authority in this order:

1. later explicit professor or supervisor instructions, prospectively recorded;
2. kickoff hard requirements;
3. the latest frozen Sprint-3 protocols and amendments;
4. older compatible blueprint/protocol language;
5. implementation choices.

Within level 3, later specific amendments supersede only the stated scope of
older documents. Current controlling authorities include:

- stage and exposure governance:
  `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`;
- final conditions and analyses:
  `docs/sprint3/FINAL_EXPERIMENT_MATRIX_PROTOCOL.md`,
  `docs/sprint3/CONFIRMATORY_CONTRAST_REGISTRY.md`, and
  `docs/sprint3/SELECTION_STATISTICS_PROTOCOL.md`;
- data/corpus authority:
  `docs/sprint3/ASQA_INTERNAL_PARTITION_PROTOCOL.md`,
  `docs/sprint3/ASQA_CORPUS_AUTHORITY_NOTE.md`,
  `docs/sprint3/HOTPOTQA_FINAL_EVALUATION_PROTOCOL.md`,
  `docs/sprint3/HOTPOTQA_FULL_CORPUS_AUTHORITY_AMENDMENT.md`, and
  `docs/sprint3/HOTPOTQA_RETRIEVAL_TEXT_CONTRACT_AMENDMENT.md`;
- retrieval design:
  `docs/sprint3/CANDIDATE_POOL_TOPK_PROTOCOL.md`,
  `docs/sprint3/DIVERSIFICATION_CONFIGURATION_PROTOCOL.md`, and
  `docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md`;
- generation/evaluation:
  `docs/sprint3/GENERATION_PROTOCOL.md`,
  `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`,
  `docs/sprint3/ACC_PROTOCOL.md`,
  `docs/sprint3/FAITHFULNESS_PROTOCOL.md`, and
  `docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md` plus Amendment 01;
- decomposer selection:
  `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md` plus Amendments 01–03;
- secondary robustness:
  `docs/sprint3/PARAPHRASE_ROBUSTNESS_PROTOCOL.md`.

`docs/sprint3/PROJECT_BLUEPRINT.md`, `RESEARCH_QUESTIONS.md`,
`DATASET_PROTOCOL.md`, and `DECISION_LOG.md` remain useful where compatible.
They do not override later specific protocols. Historical Sprint-1/Sprint-2
code and artifacts describe those historical runs, not canonical new runs.

## Kickoff and professor requirements

- Phase 0: literature/background understanding and traceability.
- Sprint-1 Phase 1: corpus preparation/indexing; BM25, DPR, Contriever, and
  ColBERTv2; a fixed generator within each controlled comparison; and a
  relevance-only baseline evaluation.
- Later project requirements: all three datasets in the overall three-sprint
  study; three generator LLMs; each LLM with and without context; and a
  defensible, prospectively governed method-selection procedure.
- ASQA dev948 is supervisor-directed `PROJECT_PROTECTED_FINAL` evidence.
- The kickoff paraphrase item is a suggestion, not a hard requirement; the
  frozen bounded development-only pilot satisfies it.
- The original kickoff deck/source is not tracked in the current repository.
  Its exact source-level traceability therefore remains incomplete.

## Research questions

The primary question is: when does diversity-aware retrieval improve RAG, and
when does it introduce noise or hallucination? The frozen matrix supports:

- relevance-only `WITH_CONTEXT` versus `WITHOUT_CONTEXT`;
- each diversification treatment versus the same-retriever relevance baseline;
- the relevance/diversity and coverage trade-off;
- whether retrieval changes propagate into substantive answer content
  (`ACC_bi`);
- effects on dataset-native correctness and atomic context faithfulness;
- heterogeneity by dataset, retriever, and generator LLM.

Claims must remain specific to the dataset's evidence regime. No method is
superior merely because it produces more diverse retrieval or higher ACC.

## Historical Sprint 1 status

Historical Sprint 1 is a complete PubMedQA relevance-only top-5 experiment,
not the complete final three-dataset/three-LLM design.

- Originating commit:
  `20c94b3e528dae2ac7cf364319a0578ead9e6a1f`
  (`Sprint 1 Final Baseline Project Cleanup`).
- Dataset: complete expert-labelled PQA-L1000, 1,000 questions.
- Historical corpus: 3,358 PubMedQA context-section documents.
- Retrievers: BM25, original DPR, Contriever, ColBERTv2.
- Generator: Mannheim Maki API, `ministral-3-14b`.
- Generation: `temperature=0.0`, `max_tokens=512`, one context-only output per
  question and retriever, top-k 5.
- Exact historical prompt is recoverable from `config.py` at the originating
  commit. It was one user message beginning `You are a helpful medical
  assistant...`, instructed use of only provided context, and used historical
  document scaffolding that could include PubMed metadata.
- Stored retrieval metrics: Recall@5 and MRR.
- Stored historical answer metrics: normalized EM, token F1, and ROUGE-L
  against `long_answer`. These are historical lexical metrics, not canonical
  Sprint-3 PubMedQA decision correctness.
- Raw result files contain 1,000 unique QA rows each and 4,000 nonblank
  predictions total; every row is labelled `OK`.
- One BM25 row contains only one retrieved document despite `row_status=OK`;
  the other 3,999 rows contain five. Preserve and disclose this anomaly.
- No `WITHOUT_CONTEXT`, llama, or Gemma generations exist.
- The current copies of the metadata, four raw result files, and two summary
  files are byte-identical to their originating-commit versions after their
  later move into `results/sprint1/`.

Historical aggregate values in `results/sprint1/raw/fullrag_summary_top5.csv`:

| Retriever | N | Recall@5 | MRR | EM | token F1 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 1,000 | 0.599211 | 0.911517 | 0.0 | 0.224472 | 0.151236 |
| DPR | 1,000 | 0.267880 | 0.474600 | 0.0 | 0.168399 | 0.115600 |
| Contriever | 1,000 | 0.605600 | 0.952733 | 0.0 | 0.230435 | 0.154575 |
| ColBERTv2 | 1,000 | 0.750586 | 0.979450 | 0.0 | 0.233590 | 0.154276 |

These values are historical observations, not current selection evidence.

## Sprint 1 evidence/gap audit

Status vocabulary:

- **A — VERIFIED COMPLETE**
- **B — PARTIALLY COMPLETE**
- **C — NOT FOUND / NOT DONE**
- **D — CANNOT VERIFY**

| # | Audit item | Status | Repository evidence and verified facts | New work required |
|---:|---|---|---|---|
| 1 | Phase-0 literature review/traceability | **B** | `README.md` retains primary references; `reports/sprint2/methods_notes.md` and `docs/sprint3/DPP_IMPLEMENTATION_AUDIT.md` retain later method references. A historical `reports/DEEP_UNDERSTANDING_BRIEF.md` exists in Git history at `6051485` but was later removed. No dedicated current Phase-0 review, source matrix, or kickoff artifact was found. | Add concise source-to-decision traceability to the future Sprint-1 notebook/report; do not invent a missing historical review. |
| 2 | PubMedQA corpus preparation | **A** | Historical `data_prep.py` at `20c94b3` creates one document per supplied context section. Current `src/data_prep.py`, `scripts/build_sample_manifests.py`, `scripts/build_corpus_manifests.py`, `artifacts/sample_manifests/pubmedqa_sample_manifest_v2.json` (1,000), and `artifacts/corpus_manifests/pubmedqa_corpus_manifest_v1.json` (3,358) provide reconstructable current identity at source revision `9001f2853fb87cab8d220904e0de81ac6973b318`. | Only physical rematerialization/validation before a new run; no corpus-method redesign. |
| 3 | HotpotQA baseline corpus preparation suitable for prospective Sprint-1 backfill | **C** | `artifacts/sample_manifests/hotpotqa_sample_manifest_v2.json` binds only the historical 500 IDs. The historical 25,997-document Sprint-2 pool is gold/query informed and explicitly prohibited for canonical new evidence. No complete 5,233,329-document manifest/corpus is present. | Acquire and validate the exact full BEIR corpus, implement its frozen text serialization, build its manifest, and pass the full-corpus resource gate before canonical retrieval. |
| 4 | ASQA baseline corpus preparation suitable for prospective Sprint-1 backfill | **C** | No ASQA data, sample/corpus manifest, or index is present. Later authority requires the full DPR-2018 corpus, expected 21,015,324 passages, with no canonical fallback. | Acquire and validate ASQA plus the full DPR-2018 corpus; materialize the frozen 3,482/871 partition and corpus manifest before use. |
| 5 | BM25 implementation | **A** | Historical `retrievers/bm25_retriever.py` at `20c94b3` executed in Sprint 1. Current `src/retrievers/bm25_retriever.py` is a hardened corpus-bound implementation; `scripts/run_pubmedqa_bm25_candidates.py` and 1,000 local top-20 candidate artifacts exist. | Dataset-specific full-corpus execution remains; no missing historical implementation. |
| 6 | DPR implementation | **A** | Historical original Facebook dual encoder executed in all 1,000 Sprint-1 rows. Current `src/retrievers/dpr_original_retriever.py` pins the question/context encoders and manifest-bound FAISS identity; 1,000 local PubMedQA top-20 candidate artifacts exist. | Full-corpus HotpotQA/ASQA indexing and candidate production remain. |
| 7 | Contriever implementation | **A** | Historical `facebook/contriever` execution produced 1,000 Sprint-1 rows. Current frozen config/runtime/provenance and candidate runner exist under `src/retrievers/` and `scripts/`; a validated 3,358-by-768 local index exists. | Current PubMedQA candidate materialization is only 3/1,000; full HotpotQA/ASQA execution remains. |
| 8 | ColBERTv2 implementation | **A** | Historical RAGatouille ColBERTv2 produced 1,000 Sprint-1 rows. Current canonical direct Stanford ColBERT implementation, immutable checkpoint policy, runtime adapter, candidate runner, environment locks, and tests are tracked. | No local current PubMedQA candidate artifacts exist; RunPod GPU validation and all required index/candidate executions remain. |
| 9 | Common retriever interface/comparable outputs | **B** | Historical `BaseRetriever`/factory and uniform CSV schema enabled four-way Sprint-1 execution. Current `src/retrieval_artifacts/` defines a stronger common top-20 artifact contract. The PubMedQA Contriever producer now has governed exact missing-ID planning, registry lifecycle hooks, resume-safe artifact preservation, output inventory, and complete-set finalization. Only PubMedQA producers exist and current local completion remains BM25 1,000, DPR 1,000, Contriever 3, ColBERTv2 0. The legacy general pipeline is not the canonical Sprint-3 runner. | Integrate the same governed layer into remaining canonical runners, complete candidate artifacts, and quarantine legacy paths from canonical execution. |
| 10 | Historical PubMedQA relevance-only top-5 runs | **A** | Four files in `results/sprint1/raw/fullrag_*_top5.csv`; 1,000 unique questions per retriever, 4,000 parsed rows total. | No rerun to rewrite history. Prospective comparison backfills remain separate. |
| 11 | Historical generator/model actually used | **A** | `results/sprint1/raw/EXPERIMENT_METADATA.json`, every raw row, notebook outputs, and the originating commit agree on Mannheim Maki `ministral-3-14b`. | None for historical identity. Physical provider revision is not recorded and cannot be retroactively invented. |
| 12 | Historical prompt/decoding actually used | **A** | Exact prompt code is in `config.py`/`generator.py` at `20c94b3`; metadata and notebook bind `temperature=0.0`, `max_tokens=512`, `top_k=5`, and `MAKI_DEFAULT_CTX=7680`. The request contained one user message and no separate system message. | Preserve as historical. New runs must use `GENERATION_PROTOCOL.md`, not this prompt. |
| 13 | Historical PubMedQA generation outputs | **A** | Four raw CSVs contain 4,000 nonblank predictions, all labelled `OK`. `results/sprint1/final_csv_outputs/01_individual_results_top5.csv` contains 4,000 parsed derived rows. | Do not overwrite. Backfill outputs require new run IDs/directories. |
| 14 | Historical retrieval metrics | **A** | Per-row Recall@5 and MRR plus four-row summary are present in `results/sprint1/raw/`. | Recompute only for a separately identified canonical replication if required; never alter historical values. |
| 15 | Historical answer-correctness metrics | **A** | Per-row normalized EM, token F1, and ROUGE-L against `long_answer` are present. | Treat them only as historical lexical comparisons. Prospective PubMedQA correctness must parse `Decision` and score `final_decision`. |
| 16 | Existing `WITHOUT_CONTEXT` outputs | **C** | No no-context field, condition, result file, or generation artifact was found in Sprint-1 or Sprint-2 results. | Generate once per dataset × sample × LLM × frozen prompt/decoding identity; never per retriever. |
| 17 | Existing HotpotQA baseline runs | **B** | Four historical `results/sprint2/raw/sprint2_beir_*_none_top5.csv` files contain 500 rows each, all `OK`, with `qwen3.5-122b`. They use the historical 25,997-document pooled corpus and are `HISTORICAL_OBSERVED`, not a suitable canonical backfill. | Canonical comparable baselines require the complete BEIR corpus and the three primary LLMs under current prompt/status/provenance contracts. |
| 18 | Existing ASQA baseline runs | **C** | No ASQA result artifact was found. | Full prospective baseline pipeline required after permitted corpus/data preparation. |
| 19 | Existing llama-3.3-70b generations | **C** | No historical result rows with this model were found. | Prospective backfill required. |
| 20 | Existing gemma4-26b generations | **C** | No historical result rows with this model were found. | Prospective backfill required. |
| 21 | Existing ministral-3-14b generations | **A** | Exactly 4,000 historical PubMedQA context generations are present, 1,000 per retriever. | Current-protocol PubMedQA `WITH_CONTEXT`, all `WITHOUT_CONTEXT`, and HotpotQA/ASQA cells remain prospective backfill. The historical PubMedQA cells use the historical prompt and 512-token limit and do not satisfy the current controlled three-LLM generation contract. |
| 22 | Existing run manifests | **B** | Sprint-1 has a small `EXPERIMENT_METADATA.json`; current PubMedQA sample/corpus manifests and the historical HotpotQA sample manifest exist. Sprint-1 metadata omits Git commit, source revision, exact prompt hash, context artifact/hash, attempts, hardware, and raw inventory hash. The new registry intentionally contains no fabricated retrospective Sprint-1 records. | Register prospective backfills before execution; do not retrofit invented provenance into historical files. |
| 23 | Existing run/provenance registry | **A — HARDENED AND VERIFIED** | The implementation passed two adversarial review cycles plus final focused verification before any experiment launched: 46 registry tests and 410 retrieval-artifact/manifest/candidate tests pass. It has a fail-closed evidence-manifest authority, order-independent aggregate candidate-set identity, content-addressed index/selected-context IDs, selected-context lineage, retry-aware append-only snapshots, and strict run-type/sprint validation. `artifacts/run_registry/run_registry_v1.jsonl` remains header-only with zero experiment records; no historical evidence changed and no scientific result was compromised. | Integrate mandatory pre-run registration into each future canonical/backfill runner before prospective execution. |
| 24 | Existing Sprint-1 notebook(s) | **B** | `notebooks/Sprint1_Baseline_RAG_Evaluation.ipynb` exists: 33 cells (27 code, 6 markdown), 20 code cells with saved outputs, including smoke and full runs. It is an executed historical Colab/reproduction notebook, not the required principal `01_sprint1_analysis.ipynb`; it embeds setup/workflow code and lacks the complete prospective-backfill story. | Preserve it. Build the new principal notebook later from immutable artifacts, with reusable logic in `src/`. |
| 25 | Historical results immutable/preserved separately | **A** | Sprint-1 raw and derived files are tracked under `results/sprint1/`; verified raw/summary hashes match their `20c94b3` versions exactly. Sprint-2 artifacts are separately under `results/sprint2/`. Frozen protocols prohibit relabelling or overwriting. | Enforce distinct run IDs/directories and `PROSPECTIVE_BACKFILL` labels for all new work. |

No audit item is classified **D**: every requested existence/configuration claim
could be resolved from the working tree or Git history. Unknown historical
physical provider revision/hardware fields are explicitly missing provenance,
not reasons to deny that the stored run exists.

## Minimal Sprint 1 comparable-condition gap matrix

This matrix records evidence presence only. It is not execution authorization.
`PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED` means a comparable condition
under the frozen current generation contract is absent. A cell may contain both
that label and `HISTORICAL EXISTING`: this preserves the original evidence while
making clear that it does not satisfy the current controlled three-LLM baseline.
Any new run must use its permitted evidence role and the frozen current
contracts.

| Dataset | Context mode | Retriever | llama-3.3-70b | gemma4-26b | ministral-3-14b |
|---|---|---|---|---|---|
| PubMedQA | `WITH_CONTEXT` relevance baseline | BM25 | PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED | PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED | HISTORICAL EXISTING; PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED |
| PubMedQA | `WITH_CONTEXT` relevance baseline | DPR | PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED | PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED | HISTORICAL EXISTING; PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED |
| PubMedQA | `WITH_CONTEXT` relevance baseline | Contriever | PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED | PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED | HISTORICAL EXISTING; PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED |
| PubMedQA | `WITH_CONTEXT` relevance baseline | ColBERTv2 | PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED | PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED | HISTORICAL EXISTING; PROSPECTIVE CURRENT-PROTOCOL BACKFILL REQUIRED |
| PubMedQA | `WITHOUT_CONTEXT` | NOT APPLICABLE — shared across retrievers | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| HotpotQA | `WITH_CONTEXT` relevance baseline | BM25 | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| HotpotQA | `WITH_CONTEXT` relevance baseline | DPR | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| HotpotQA | `WITH_CONTEXT` relevance baseline | Contriever | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| HotpotQA | `WITH_CONTEXT` relevance baseline | ColBERTv2 | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| HotpotQA | `WITHOUT_CONTEXT` | NOT APPLICABLE — shared across retrievers | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| ASQA | `WITH_CONTEXT` relevance baseline | BM25 | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| ASQA | `WITH_CONTEXT` relevance baseline | DPR | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| ASQA | `WITH_CONTEXT` relevance baseline | Contriever | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| ASQA | `WITH_CONTEXT` relevance baseline | ColBERTv2 | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |
| ASQA | `WITHOUT_CONTEXT` | NOT APPLICABLE — shared across retrievers | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED | PROSPECTIVE BACKFILL REQUIRED |

The four historical HotpotQA relevance-baseline runs with `qwen3.5-122b` are
real historical Sprint-2 evidence but sit outside this three-primary-LLM matrix
and use a prohibited corpus for new canonical evidence. The four historical
PubMedQA `ministral-3-14b` cells also use the historical prompt, 512-token limit,
and lexical metrics. They remain valid historical cells, not substitutes for a
canonical Sprint-3 replication.

## Historical Sprint 2 status

- Dataset: historical seed-42 sample of 500 BEIR-test HotpotQA queries.
- Corpus: 25,997 passages formed from 997 unique gold passages plus 50 sampled
  negatives per query; it is query/gold informed and may not be used for new
  canonical HotpotQA evidence.
- Four retrievers and 15 stored conditions per retriever: baseline, five MMR
  lambdas, KMeans 2/3/5, Agglomerative 3/5, deterministic DPP, and three
  historical stochastic DPP seeds.
- `results/sprint2/raw/` contains 60 condition CSVs × 500 rows = 30,000
  retrieval rows, all labelled `OK`.
- Thirty-two files contain `qwen3.5-122b` outputs (16,000 generations); 28 are
  retrieval-only. Historical generation settings are temperature 0 and 512
  maximum tokens.
- Summaries/figures exist. The progress report and individual-contribution
  files are empty, and there is no principal `02_sprint2_analysis.ipynb`.
- All Sprint-2 claims remain historical and must be recomputed under current
  instruments before being presented as canonical new evidence.

## Current Sprint 3 and final-design status

Scientific methodology is broadly frozen. Implementation, provenance,
calibration, corpus construction, resource validation, bounded clustering
selection, and execution remain.

- Final datasets: PubMedQA control replication, HotpotQA protected final 500,
  ASQA protected dev948.
- Final per-retriever context treatments: relevance baseline; MMR lambdas
  0/0.25/0.50/0.75; one selected global KMeans k; one selected global
  Agglomerative k; deterministic `dpp_map`.
- KMeans selection candidates are `{2,3,5}`; Agglomerative candidates are
  `{3,5}`. Only these two hyperparameters are selected in Stage 3.
- Final structural counts are 32 retrieval/context cells and 96 with-context
  generations per sample, plus three shared no-context generations.
- The confirmatory contrast registry is frozen. PubMedQA analyses are
  registered descriptive control analyses; HotpotQA/ASQA protected contrasts
  are confirmatory.
- `SELECTION` has not opened. No protected-final execution has started.
- `results/sprint3/` and `configs/sprint3/` contain no files.

## Datasets and evidence roles

| Dataset | DEVELOPMENT | SELECTION | PROJECT_PROTECTED_FINAL | Historical/control |
|---|---|---|---|---|
| PubMedQA | Exposed/control use only | No numeric selection vote | None | Complete PQA-L1000, `HISTORICAL_OBSERVED_CONTROL_REPLICATION`, N=1,000 |
| HotpotQA | BEIR train | BEIR dev, N=5,447 | Deterministic N=500 from the 6,905-query unexposed complement of BEIR test | Historical seed-42 BEIR-test 500 and 25,997-doc pool |
| ASQA | 3,482 IDs from official train | 871 disjoint IDs from official train | All official dev, N=948 | No historical run found |

Canonical corpora:

- PubMedQA: shared 3,358-section corpus from PQA-L1000.
- HotpotQA: only the complete BEIR corpus, expected 5,233,329 documents;
  `retrieval_content = (title + " " + text).strip()`; no canonical fallback.
- ASQA: only the full DPR Wikipedia 2018-12-20 corpus, expected 21,015,324
  passages; no canonical fallback.

All four retrievers must use the same logical corpus, IDs, boundaries, and
frozen pre-tokenizer document surface within each dataset. No query-specific
corpus or post-retrieval gold injection is permitted.

## Retrievers

- **BM25:** `rank_bm25.BM25Okapi`; current hardened PubMedQA implementation
  binds runtime cache to corpus/config and uses deterministic score-tie order.
- **DPR:** Facebook single-NQ question/context encoders; 768-dimensional
  float32 `IndexFlatIP`; canonical adapter supplies complete shared passage
  content as text with an empty separate-title field.
- **Contriever:** `facebook/contriever`; pinned revision/config, 768-dimensional
  float32 exact inner-product index.
- **ColBERTv2:** direct Stanford ColBERT (`colbert-ai==0.2.22`) with
  `colbert-ir/colbertv2.0` at immutable revision
  `0855eac81381e0323a846f1ed7d8452d4c648b50`; isolated environment required.

Retriever identity remains an experimental factor; do not select one winner
and discard the others.

## LLMs and generation

Primary generators:

- `llama-3.3-70b`;
- `gemma4-26b`;
- `ministral-3-14b`.

Reserve: `qwen3.5-122b`, usable only through a prospective amendment before
Stage 2 closes for unavailability, operability, or repeatability failure. It is
not a fourth canonical LLM and may never be substituted for quality reasons.

Canonical generation uses the exact system/user templates in
`GENERATION_PROTOCOL.md`, temperature 0, one replica, maximum output tokens
256/256/512 for PubMedQA/HotpotQA/ASQA, exactly five passage bodies for
`WITH_CONTEXT`, and no context placeholder for `WITHOUT_CONTEXT`. Statuses are
`OK`, `REFUSAL`, `PARSE_FAILURE`, `TRUNCATED`, and `ERROR`; only infrastructure
errors may retry, with three total attempts maximum.

## Diversification methods

- relevance-only baseline: first five of the shared top-20;
- MMR: canonical 0/0.25/0.50/0.75 curve; lambda 1 is development equivalence
  validation only;
- KMeans: development/selection candidates 2/3/5, one global winner;
- Agglomerative: candidates 3/5, one global winner;
- deterministic fixed-cardinality `dpp_map`: canonical DPP treatment;
- exact stochastic fixed-k DPP seeds 1/2/3: development sensitivity only.

Every method consumes the identical retriever-specific top-20 artifact for a
dataset/query. No method may reretrieve.

## Evaluation metrics

Retrieval effectiveness:

- PubMedQA: positive-gold-section Recall@5 primary descriptive; MRR@5
  secondary.
- HotpotQA: official qrel Recall@5 lead; MRR@5 secondary/descriptive.
- ASQA: corpus-relative SRecall@5 lead; alpha-nDCG@5 at alpha 0.5 mandatory
  secondary; alpha 0.3/0.7 sensitivities; coverability and exact `c*`
  diagnostics. All reuse the frozen BODY-only alias matcher.

Answer correctness:

- PubMedQA: parsed categorical `final_decision` accuracy.
- HotpotQA: official answer token F1 primary and official EM secondary.
- ASQA: official Disambig-F1/QA-F1 primary; deterministic answer-side alias
  coverage secondary diagnostic.

Other constructs:

- `Faithfulness@5`: atomic-claim support against the exact five supplied
  passage bodies; `WITHOUT_CONTEXT` is not applicable.
- `ACC_bi`: bidirectional substantive asserted-content change between
  relevance-only and diversified `WITH_CONTEXT` answers. Higher means more
  change, not better quality.
- retrieval diversity is a manipulation/interpretive diagnostic, not a
  quality objective.
- paraphrase robustness is a separate development-only descriptive retrieval
  pilot using Jaccard overlap at 20 and at 5.

The current code metric registry predates several frozen definitions and still
contains legacy generic entries. Registry synchronization and exact scorer
implementations are required before canonical evaluation.

## Frozen constants

- `candidate_pool = 20`;
- `final_top_k = 5`;
- one canonical generation replica;
- temperature 0;
- primary LLM count 3;
- maximum generation tokens: PubMedQA 256, HotpotQA 256, ASQA 512;
- retrieval/context treatments per retriever: 8;
- bootstrap resamples: 10,000; base seed 20260823; paired 95% percentile CI;
- HotpotQA corpus: full only, expected 5,233,329;
- ASQA corpus: full only, expected 21,015,324;
- HotpotQA resource projections: multiplier 1.25, 20% reserve, pass when
  `1.25 * R <= 0.80 * L`;
- decomposer winner procedure and terminal tie-break are frozen, but the winner
  has not been executed/selected;
- NLI verifier family is frozen; empirical ACC/faithfulness thresholds have
  not been fitted.

## Completed results and artifacts

- Immutable tracked Sprint-1 raw and derived CSV artifacts under
  `results/sprint1/`.
- Completed read-only historical Sprint-1 artifact validation, with machine-
  readable inventory at
  `artifacts/audit_inventories/sprint1_historical_artifact_inventory_v1.json`.
  The inventory binds all five source hashes, four 1,000-row retriever files,
  the common QA identity, historical generation configuration, metric fields,
  4,000 `OK` statuses, and the preserved BM25 `qa_id=60` anomaly.
- Executed historical Sprint-1 Colab notebook at
  `notebooks/Sprint1_Baseline_RAG_Evaluation.ipynb`.
- Immutable tracked Sprint-2 raw, summary, and figure artifacts under
  `results/sprint2/`.
- Frozen PubMedQA sample manifest (1,000) and corpus manifest (3,358).
- Frozen historical HotpotQA sample manifest (500).
- Hardened and independently verified run/provenance registry at
  `src/run_registry.py`, with schema `sprint3.run-registry-record.v1`, 46
  focused passing tests after two adversarial review cycles plus final focused
  verification, 410 passing retrieval-artifact/manifest/candidate tests, a
  one-binding prospective evidence authority at
  `artifacts/run_registry/evidence_manifest_authority_v1.json`, and a
  header-only registry containing zero experiment records at
  `artifacts/run_registry/run_registry_v1.jsonl`. No historical run record was
  fabricated and no experiment has used the registry yet.
- Local ignored Sprint-3 PubMedQA candidate artifacts: BM25 1,000; DPR 1,000;
  Contriever 3; ColBERTv2 0.
- Local ignored manifest-bound DPR and Contriever PubMedQA embedding/FAISS
  caches, each 3,358 × 768.
- Frozen Sprint-3 scientific protocols and amendments through current HEAD.

## Missing work

- run-registry integration into remaining canonical runners before their
  prospective execution; the PubMedQA Contriever missing-only path is ready
  for a final clean-HEAD confirmation after the prospective three-attempt
  retrieval infrastructure ceiling was documented and centrally enforced;
- physical generation model bindings and evaluator output schemas; the
  canonical PubMedQA prompt, selected-context, generation-row, Maki adapter,
  repeatability-gate, missing-only runner, and governed block interfaces are
  now implemented but have made no model calls;
- local co-location and exact identity validation of the already-completed
  RunPod PubMedQA Contriever and ColBERTv2 candidate sets and ColBERT index;
- HotpotQA and ASQA source acquisition, manifests, full corpora, indexes, and
  candidate producers;
- HotpotQA full-corpus resource pilot and full-build validation;
- canonical diversification artifact production;
- physical Maki model identities, local co-location/validation of all four
  candidate sets, materialization of relevance-only selected contexts, and
  execution of the frozen generator repeatability gate;
- decomposer bake-off, immutable winner snapshot, and claim artifacts;
- NLI snapshot, human annotation, threshold fitting/validation, and evaluator
  bundle;
- PubMedQA/HotpotQA parsers and official correctness scorers; ASQA official
  scorer snapshot/parity;
- ASQA matcher implementation, 150-case development validation, J matrix,
  SRecall/alpha-nDCG/coverability/`c*`;
- ACC and faithfulness implementations;
- materialized development/selection/protected manifests at their governed
  stages;
- prospective comparable baseline backfills in the gap matrix;
- the four principal notebooks and final report/presentation.

## External blockers

- Professor/supervisor adjudication is required for seven inactive proposals:
  `DELTA_HOTPOT_RECALL`, `DELTA_ASQA_SRECALL`, `DELTA_CORRECTNESS`,
  `DELTA_FAITHFULNESS`, `EPSILON_PRACTICAL_EQUIVALENCE`,
  `STRUCTURAL_RETRIEVAL_FAILURE_RULE`, and
  `DELTA_DOWNSTREAM_TECH_FAILURE`. Proposed values/rules in the statistics
  protocol are not active.
- Full-corpus HotpotQA and ASQA work requires sufficient storage, RAM/VRAM,
  time, quota, and budget. Neither dataset has a canonical reduced fallback.
- Maki access and immutable physical identities/availability for the three
  primary LLMs must be established before their gates.
- Human annotators/reviewers are needed for decomposer selection, NLI
  calibration, ASQA matcher validation, and the optional paraphrase pilot.
- Required immutable dataset/model acquisitions depend on external sources and
  licenses; none may float silently to a new revision.

## Things that must not change

- Never alter, overwrite, delete, relabel, or silently repair historical
  Sprint-1/Sprint-2 raw results.
- Never convert missing, failed, truncated, or uncomputed values to zero unless
  a frozen metric explicitly defines a semantic zero (for example, correctness
  of a refusal).
- Never inspect selection/protected outcomes before their stage gate.
- Never tune with protected-final data or use poor outcomes to change methods,
  prompts, thresholds, corpora, metrics, or statistics.
- Never use the historical HotpotQA 25,997-document pool as canonical new
  evidence; HotpotQA and ASQA have no authorized canonical fallback.
- Never inject gold/qrel/alias documents into retrieval candidates.
- Never duplicate `WITHOUT_CONTEXT` by retriever/diversifier.
- Never let retriever/LLM-specific exceptions change the shared logical corpus,
  prompt, parser, threshold, or metric semantics.
- Never commit credentials, private data, or floating unverified model state.

## Four-notebook plan

Exactly four principal research notebooks will be produced:

1. `notebooks/01_sprint1_analysis.ipynb`
2. `notebooks/02_sprint2_analysis.ipynb`
3. `notebooks/03_sprint3_analysis.ipynb`
4. `notebooks/04_final_research_analysis.ipynb`

Each sprint notebook must cover objective/RQ, datasets/configuration, workflow,
small real demonstrations, provenance, validation/loading of complete saved
artifacts, metrics, tables, figures, interpretation, and conclusions. Core
implementations remain in `src/`; notebooks consume frozen artifacts and do
not contain manually typed result numbers.

## Result storage and provenance

Raw experimental outputs live outside notebooks in structured artifacts. Keep
the trace:

```text
sample/question
-> first-stage top-20 candidates
-> selected top-5
-> exact context and prompt
-> raw LLM output and status
-> evaluator outputs/statuses
-> aggregate analysis
```

Every new run must bind its run ID, stage/evidence role, Git commit and clean
state, dataset/sample/corpus manifests, retriever/index/configuration,
diversifier, candidate pool/top-k, LLM physical identity, prompt/decoding
hashes, context artifacts, evaluator bundle, environment/hardware, attempts,
timestamps, row counts, failure reasons, output inventory, and hashes. Reuse
artifacts only by exact identity. Keep historical outputs and prospective
backfills in separate, explicitly labelled run directories.

## Execution plan

1. **Sprint 1:** verify historical evidence; implement only missing comparable
   baseline backfills under current contracts; create/validate
   `01_sprint1_analysis.ipynb`; freeze Sprint 1.
2. **Sprint 2:** verify historical diversification evidence; backfill only
   missing comparable conditions; create/validate `02_sprint2_analysis.ipynb`;
   freeze Sprint 2.
3. **Sprint 3:** close Stage-1/2 implementation, calibration, provenance, and
   professor-dependent gates; bounded selection; lock winners; execute frozen
   final/protected matrix; predefined Phase-5 analyses; create/validate
   `03_sprint3_analysis.ipynb`; freeze Sprint 3.
4. **Final:** create `04_final_research_analysis.ipynb`; prepare only
   evidence-supported report and presentation conclusions.

## Current checkpoint

- Mode: EXECUTION STARTED
- Current sprint: SPRINT 1 COMPLETION
- Last completed: canonical governed PubMedQA generation subsystem
  implemented without retrieval or model/API execution
- Current task: generation pre-execution closure: physical Maki model-ID
  resolution, candidate co-location/validation, and repeatability execution
- Current candidate status:
  - BM25: 1,000/1,000 reusable
  - DPR: 1,000/1,000 reusable
  - Contriever: 3/1,000 currently co-located in WSL; independently validated
    RunPod set is 1,000/1,000 and must be co-located/identity-validated, not rerun
  - ColBERTv2: 0/1,000 currently co-located in WSL; RunPod contains 1,000
    candidates and the completed canonical index, which must be
    co-located/identity-validated, not rerun
- Production registry: 0 experiment records
- Generation-infrastructure verification: 21 focused generation tests, 48
  existing registry tests, 320 relevant retrieval-artifact/manifest/candidate
  tests, and 1,308 full-repository tests passed
- SELECTION: BLOCKED pending professor numeric-rule adjudication
- Protected-final execution: NOT STARTED
- Experiments launched: none; the production registry contains zero experiment
  records, historical Sprint-1/Sprint-2 evidence remains unchanged, and no
  scientific result is compromised
- Canonical generation status: subsystem implemented; zero model/API calls;
  repeatability gate pending; all 15 governed blocks and 15,000 canonical rows
  remain pending
- Pre-execution dependency: resolve and provenance-bind the three physical
  Maki model IDs; co-locate and validate all four candidate sets and required
  neural index identities; materialize the 4,000 missing-only selected-context
  artifacts; freeze the exact 20-prompt DEVELOPMENT repeatability manifest;
  pass the 180-call gate before opening any 1,000-row generation block
- Next action: perform the physical-model and candidate/context preflight;
  do not register or execute a generation block until every prerequisite and
  the matching repeatability gate pass
- Branch: `sprint3`
- Implementation base HEAD: `ac18535d19ab49a5b2a9a345e9602c6d5631c089`
- Registry implementation audit-start working tree: clean
- Current implementation files: `src/generation/`, the three governed
  PubMedQA generation/context/repeatability CLIs, two focused generation test
  modules, and this continuity update; historical inputs unchanged

## Exact next action

Resolve the three physical Maki model identities and co-locate/validate the
four current PubMedQA candidate sets and neural index identities. Then
materialize exact relevance-only selected contexts and freeze/execute the
20-prompt repeatability gate. Do not register or execute any 1,000-row
generation block before those prerequisites pass.

## Known contradictions and superseded statements

These do not reopen frozen methodology:

- `PROJECT_BLUEPRINT.md` still mentions conditional HotpotQA and ASQA reduced
  fallbacks and lists several now-frozen items as unresolved. The HotpotQA
  full-corpus amendment, ASQA corpus authority note, and later metric/generation
  protocols control.
- `FINAL_EXPERIMENT_MATRIX_PROTOCOL.md` Section 22 retains old HotpotQA
  fallback/`M_min` wording. The later full-corpus amendment retires `M_min` and
  prohibits canonical fallback.
- `EXPERIMENT_STAGE_GATE_PROTOCOL.md` lists HotpotQA retrieval text as open; the
  later retrieval-text amendment freezes it.
- `DATASET_PROTOCOL.md` contains the older ASQA expected count 21,015,300 and
  reduced-final language. The later ASQA authorities freeze expected count
  21,015,324, dev948, and no fallback.
- `README.md` describes a two-sprint project, legacy generator defaults and
  commands, and a removed `reports/DEEP_UNDERSTANDING_BRIEF.md`. It is stale
  orientation, not current execution authority.
- `src/config.py`, `src/generator.py`, `src/pipeline.py`, and the legacy
  evaluation path retain historical prompt/model/default behavior. Canonical
  new execution must fail fast rather than inherit it.
- `src/evaluation/metric_registry.py` does not yet reflect all frozen ASQA,
  correctness, ACC, and faithfulness definitions. This is an implementation
  synchronization gap.
- Sprint-1 derived files call one retriever “best” and include estimated timing
  fields. Those are historical derived outputs, not current method-selection
  authority.
