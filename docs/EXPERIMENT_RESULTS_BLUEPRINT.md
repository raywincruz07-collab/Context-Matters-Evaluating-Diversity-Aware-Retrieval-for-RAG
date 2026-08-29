# Context Matters: Permanent Experiment and Results Blueprint

Status: active project-wide execution map. Last synthesized: 2026-08-29.

This document is the single practical map for completing the project. It does
not replace frozen scientific protocols or amendments. Where details conflict,
authority is: later supervisor instruction, kickoff hard requirements, the
latest specific frozen protocol/amendment, then this synthesis. Historical
Sprint-1 and Sprint-2 evidence remains immutable and retains its original
labels. New work is stored as separately identified prospective controlled
evidence.

The active statistical plan contains **no non-inferiority margin, acceptable-
loss margin, equivalence margin, or numerical tolerance rule**. All legacy
numerical-margin logic and the inactive margin proposals in
`SELECTION_STATISTICS_PROTOCOL.md` are superseded for future execution by the
supervisor's correction. This does not alter historical artifacts.

## 1. Central research question

**When does diversifying retrieved context improve RAG, and when does it
introduce noise, reduce answer quality, or increase hallucination?**

Concise frozen questions/hypotheses:

- **RQ1 — context value:** compare relevance-only `WITH_CONTEXT` with the same
  LLM's shared `WITHOUT_CONTEXT` answer.
- **RQ2 — diversification effect:** compare every frozen diversification
  treatment with the same retriever's relevance-only top-5 baseline.
- **RQ3 — trade-off:** relate diversity and coverage gains to changes in
  retrieval effectiveness, correctness, faithfulness, and answer content.
- **RQ4 — heterogeneity:** determine how effects vary by focused-evidence
  PubMedQA, complementary-evidence HotpotQA, ambiguous/multi-aspect ASQA,
  retriever, and LLM.
- **RQ5 — groundedness:** determine whether diversification changes
  `Faithfulness@5` and unsupported-claim behavior.
- **Mechanistic hypothesis:** diversification can help when distinct evidence
  or aspects are needed, but excessive diversity pressure can displace relevant
  evidence and add noise. This is a hypothesis, not a conclusion.
- **Control hypothesis:** PubMedQA may benefit less from diversification because
  it is a focused-evidence control. Its results remain descriptive because the
  complete PQA-L1000 has already been observed.

## 2. Kickoff phase checklist

The original kickoff source is not tracked. This checklist uses the repository's
audited reconstruction in `PROJECT_STATE.md`; source-level traceability to the
original deck remains incomplete.

| Phase | Requirement | Classification | Execution interpretation |
|---|---|---|---|
| 0 — Literature/background | Understand RAG, retrieval, diversity, MMR, clustering, DPP, datasets, and evaluation; preserve source-to-decision traceability | REQUIRED | Consolidate existing references and a concise decision/source matrix in professor-facing reporting. Do not invent a missing historical review. |
| 1 — Baseline RAG | Prepare/index the corpus; run BM25, DPR, Contriever, ColBERTv2; relevance-only top-5; fixed generator within each comparison; evaluate | REQUIRED | Complete controlled baselines for all datasets and three primary LLMs, while preserving historical runs separately. |
| 1 — Baseline RAG | `WITH_CONTEXT` and `WITHOUT_CONTEXT` for each LLM | REQUIRED by later professor direction | Generate no-context once per dataset × sample × LLM, not once per retriever. |
| 1 — Baseline RAG | Streamlit/demo work | OPTIONAL / NOT PLANNED | Existing demo is not scientific evidence and needs no expansion. |
| 2 — Diversification | Implement and compare diversity-aware selection against relevance-only retrieval | REQUIRED | Same top-20 candidates and final top-5; apples-to-apples pairs. |
| 2 — Diversification | MMR, KMeans, Agglomerative, DPP | SELECTED IMPLEMENTATION | Use only frozen grids and semantics below. |
| 2 — Diversification | Spectral clustering, xQuAD, additional methods | OPTIONAL / NOT PLANNED | Explicitly excluded by the frozen diversification protocol. |
| 2 — Diversification | Three or more seeds for stochastic DPP | SELECTED IMPLEMENTATION | Exact fixed-k DPP seeds 1/2/3, development sensitivity only. Deterministic `dpp_map` needs no seed. |
| 3 — Experimental design | Fixed datasets/splits, samples, corpora, retrievers, top-20/top-5, LLMs, prompts, decoding, method grids, and provenance before outcomes | REQUIRED | Govern through manifests, registry, stage gates, and immutable configuration IDs. |
| 3 — Experimental design | Tune context size/top-k | OPTIONAL / NOT PLANNED | Top-20 and top-5 are frozen; only bounded development sensitivity already authorized may be diagnostic. |
| 4 — Evaluation framework | Retrieval, correctness, diversity/coverage, faithfulness/hallucination, paired uncertainty, failures | REQUIRED | Use the metric master table and per-query evidence; never substitute aggregate-only results. |
| 4 — Evaluation framework | Generic geometric diversity | SELECTED IMPLEMENTATION | Mean pairwise cosine distance among normalized top-5 Contriever-space document embeddings; diagnostic/manipulation check only. |
| 4 — Evaluation framework | Every possible IR/NLP metric | OPTIONAL / NOT PLANNED | Use only the frozen/project-selected metrics below. |
| 5 — Final analysis | Diversity/accuracy, hallucination, output change, dataset/LLM/retriever dependence | REQUIRED | Predefined analyses A–G below. |
| 5 — Final analysis | Paraphrase robustness | OPTIONAL kickoff suggestion; SELECTED bounded implementation | Development-only pilot using frozen Jaccard measures; no protected-final claim or new Holm family. |

## 3. Sprint 1 — controlled relevance-only baseline

### Goal and separation from history

Establish comparable relevance-only RAG baselines before diversification. The
historical Sprint 1 is the immutable PubMedQA/`ministral-3-14b` top-5 run and
the historical Sprint 2 includes HotpotQA baseline rows on a prohibited
query-informed 25,997-document pool. Neither substitutes for the current
three-dataset, three-LLM controlled baseline.

Each planned dataset uses 3 LLMs × 4 retrievers × relevance-only
`WITH_CONTEXT` top-5, plus 3 retriever-independent `WITHOUT_CONTEXT`
conditions. No diversification is run in Sprint 1. All top-5 contexts are ranks
1–5 of the same retriever-specific top-20 artifact later used in Sprint 2.

| Dataset | Permitted Sprint-1 evidence | Planned N | Corpus authority | Retrieval metrics | Answer and context metrics |
|---|---|---:|---|---|---|
| PubMedQA | Complete exposed PQA-L1000; label `HISTORICAL_OBSERVED_CONTROL_REPLICATION` for canonical replication | **1,000** | Shared 3,358 section corpus from PQA-L1000; verified manifest; no outside/gold injection | Recall@5 primary descriptive; MRR@5 secondary; candidate Recall@20 diagnostic | Parsed verdict accuracy primary; `Faithfulness@5` for `WITH_CONTEXT`; focused-evidence section coverage diagnostic where useful; no ASQA-style aspect claim |
| HotpotQA | `DEVELOPMENT` only (BEIR train); do not expose dev selection or protected test IDs for baseline development | **PRE-EXECUTION DECISION:** exact fixed development sample N/manifest | Complete BEIR HotpotQA corpus, expected 5,233,329 docs; `(title + " " + text).strip()`; no fallback | qrel Recall@5 primary; MRR@5 secondary; NDCG@5 secondary; candidate Recall@20 diagnostic; supporting-fact/evidence coverage where derivable | Official answer F1 primary, EM secondary; `Faithfulness@5`; supporting-evidence coverage |
| ASQA | Frozen train-derived `DEVELOPMENT` partition only | **3,482** | Full DPR Wikipedia 2018-12-20 corpus, expected 21,015,324 passages; no fallback | SRecall@5 primary; alpha-nDCG@5 alpha=.5 mandatory secondary; alpha=.3/.7 sensitivity; candidate SRecall@20, coverability, absolute gold coverage, exact `c*` diagnostics | Official Disambig-F1/QA-F1 primary; answer-side alias coverage diagnostic; `Faithfulness@5`; aspect coverage |

All three primary logical LLM slots are `llama-3.3-70b`, `gemma4-26b`, and
`ministral-3-14b`. Use the frozen physical binding, prompt, parsing, status,
retry, and decoding contracts: temperature 0, one replica, and maximum output
tokens 256/256/512 for PubMedQA/HotpotQA/ASQA. A reserve substitution requires
the already-governed prospective amendment and never changes the logical slot.

Minimum per-query baseline outputs: sample identity and evidence role; gold;
top-20 IDs, scores, ranks and text hashes; ordered selected top-5; retrieval
metrics; exact context and prompt; LLM binding/decoding; raw response, finish
reason and status; parsed answer; correctness; faithfulness and its component
states; applicable coverage/diversity diagnostics; all evaluator statuses; and
complete provenance. `WITHOUT_CONTEXT` has no retrieval or faithfulness value,
and these fields remain missing/not applicable rather than zero.

## 4. Sprint 2 — controlled diversification and selection

Every method consumes exactly the Sprint-1 retriever-specific top-20 for the
same sample. The baseline is explicitly `method=none`, ranks 1–5. Never use MMR
lambda 1 as the baseline and never reretrieve per method.

| Family | Development/selection configurations | Canonical role and determinism |
|---|---|---|
| Relevance baseline | `none` | Deterministic reference retained everywhere |
| MMR | lambda 0, .25, .50, .75, 1.0 | 0/.25/.50/.75 are frozen treatments; 1.0 is development-only order/ID equivalence validation |
| KMeans | k={2,3,5}; normalized Contriever embeddings; k-means++; `n_init=1`, Lloyd, max_iter=300, tol=1e-4, seed=42; relevance-ranked round robin | Stochastic initialization controlled by frozen seed; select one **global** k, never per dataset/retriever/LLM |
| Agglomerative | k={3,5}; normalized Contriever embeddings; Euclidean/Ward; relevance-ranked round robin | Deterministic subject to pinned library/tie behavior; select one **global** k |
| DPP | deterministic `dpp_map`, greedy MAP-style approximation, cardinality 5, theta=1 | Canonical deterministic treatment; not an exact global MAP solver |
| Stochastic k-DPP | exact fixed-k seeds {1,2,3} | Development sensitivity only; retain every seed, report seed variability and within-query seed average; never select the best seed; no required downstream generation |
| Spectral/xQuAD | none | OPTIONAL / NOT PLANNED |

### Evidence roles and configuration choice

- HotpotQA selection uses the complete BEIR dev split, **N=5,447**.
- ASQA selection uses the frozen disjoint train-derived `SELECTION` partition,
  **N=871**.
- PubMedQA may be shown as exposed control/development evidence but casts no
  numeric clustering-selection vote.
- MMR values are not selected from outcomes. Only global KMeans k and global
  Agglomerative k are selected; the winners remain unresolved until selection
  is validly executed.
- Protected HotpotQA test and ASQA dev948 stay closed. No selection result may
  change prompts, metrics, grids, corpora, retrievers, LLMs, or the final matrix.

Before selecting configurations, save for every selection query and candidate:
the lead retrieval metric (Hotpot Recall@5 or ASQA SRecall@5), mandatory
secondary retrieval metrics, generic diversity, dataset-specific evidence or
aspect coverage, structural failures and missingness. For Stage-B conditions,
also save all three LLMs' primary correctness, secondary correctness,
`Faithfulness@5`, status/failure rates, and diagnostic `ACC_bi`. Preserve raw
paired vectors, mean effects, 95% paired CIs, paired tests, and multiplicity
fields. Under the supervisor correction there is no margin gate: a concise
outcome-independent rule for resolving the global clustering winners without
the removed margins is a **PRE-EXECUTION DECISION** that must be frozen before
`SELECTION` opens.

## 5. Sprint 3 — final controlled experiment

Protected-final evidence remains closed until all stage gates, implementations,
physical identities, scorer calibrations, manifests, selected clustering IDs,
and statistical decisions pass. After it opens: no tuning, replacement based
on quality, metric change, prompt change, threshold refit, method reduction, or
new confirmatory contrast.

| Dataset | Final evidence role | N |
|---|---|---:|
| PubMedQA | Exposed descriptive control replication | 1,000 |
| HotpotQA | `PROJECT_PROTECTED_FINAL`; deterministic 500 from the unexposed 6,905-query complement of BEIR test | 500 |
| ASQA | `PROJECT_PROTECTED_FINAL`; complete official dev | 948 |

For each sample: 4 retrievers × 8 `WITH_CONTEXT` conditions = 32 contexts;
32 × 3 LLMs = 96 with-context generations; plus 3 shared no-context
generations = **99 generations**. The eight contexts are relevance baseline,
MMR 0/.25/.50/.75, selected global KMeans, selected global Agglomerative, and
deterministic `dpp_map`. Total planned generation rows are PubMedQA 99,000,
HotpotQA 49,500, and ASQA 93,852, subject to governed status rows rather than
silent deletion. Retrieval has 32 configuration rows per sample. Historical
rows are never renamed, merged into, or overwritten by these counts.

The protected contrast registry remains structurally applicable: RQ1
correctness (24), RQ2 retrieval (56), RQ2 correctness (168), and RQ5
faithfulness (168): **416 contrasts in 62 predefined Holm families** across
HotpotQA and ASQA. PubMedQA uses the same effect/CI/test reporting descriptively
but is not added to protected confirmatory families.

## 6. Phase-5 / final research analyses

| Analysis | Required inputs and metrics | Produced by |
|---|---|---|
| A. Diversity vs accuracy / diversity ceiling | Per-query top-5 embeddings/IDs; mean pairwise cosine distance; method intensity; Recall@5 or SRecall@5; alpha-nDCG/coverage where applicable; primary correctness; paired deltas. Plot quality against diversity without declaring diversity intrinsically good. | Sprint 1 baselines; Sprint 2 method curve/selection; Sprint 3 final confirmation |
| B. Diversity vs faithfulness/hallucination | Exact contexts, atomic claims, passage-support decisions, `Faithfulness@5`, unsupported/contradicted/unverifiable component counts, correctness and diversity deltas | Sprint 1 baseline instrumentation; Sprint 2 development; Sprint 3 final |
| C. Context diversity vs output change | Paired relevance/diversified responses, exact pairing identity, directional ACC components, `ACC_bi`, correctness and faithfulness deltas | Sprint 2 and Sprint 3; baseline answers originate in Sprint 1/reused final baseline cells |
| D. Paraphrase robustness | Fixed development-only original/paraphrase manifest; separate retrieval for each surface; Jaccard top-20 and top-5 for baseline and MMR .50; ordered-top-5 identity and dataset-native retrieval delta diagnostics | Bounded development pilot in Sprint 2; summarized in final analysis |
| E. Dataset dependence | Same configuration IDs and effect definitions across all datasets; evidence-role labels; per-dataset paired deltas/CIs; dataset-native metrics, never invalidly pooled | All sprints; Sprint 3/final synthesis |
| F. LLM dependence | Logical and physical LLM IDs; same samples/contexts/prompts; per-LLM correctness, faithfulness, ACC, failures, paired treatment effects | Sprint 1 generation baseline; Sprint 2 Stage B; Sprint 3 final |
| G. Retriever dependence | Shared corpus/sample and method artifacts; retriever-specific top-20/top-5; retrieval, diversity, correctness, faithfulness and ACC effects | All sprints; no “winner” test or retriever elimination |

## 7. Metric master table

“Primary” is dataset/construct specific; it does not authorize cross-dataset
pooling of unlike metrics.

| Metric | Purpose | Dataset(s) | Sprint(s) | Per-query required? | Role | Origin |
|---|---|---|---|---|---|---|
| Recall@5 | Relevant-document/section recovery | PubMedQA, HotpotQA | 1–3 | Yes | Primary retrieval (PubMed descriptive) | Kickoff retrieval evaluation; frozen project definition |
| MRR@5 | Rank of first relevant result | PubMedQA, HotpotQA | 1–3 | Yes | Secondary | Project-selected/frozen |
| NDCG@5 | Ranked relevance diagnostic where qrels permit | HotpotQA | 1–3 | Yes | Secondary | Kickoff/project-selected; do not make primary |
| Candidate Recall@20 / SRecall@20 | Diagnose whether reranking is candidate-limited | All as applicable | 1–3 | Yes | Diagnostic | Project-selected |
| SRecall@5 | Coverable ASQA aspect recall | ASQA | 1–3 | Yes | Primary retrieval | Frozen project metric |
| alpha-nDCG@5, alpha=.5 | Novelty/rank-aware aspect coverage | ASQA | 1–3 | Yes | Mandatory secondary | Frozen project metric |
| alpha-nDCG@5, alpha=.3/.7 | Alpha sensitivity | ASQA | 1–3 | Yes | Diagnostic | Frozen project metric |
| Absolute gold coverage@5, coverability, exact c* | Separate corpus limitations and aspect coverage ceiling | ASQA | 1–3 | Yes | Diagnostic | Frozen project metric |
| PubMedQA verdict accuracy | Categorical final-decision correctness | PubMedQA | 1–3 | Yes | Primary correctness | Frozen project metric |
| HotpotQA token F1 | Official answer correctness | HotpotQA | 1–3 | Yes | Primary correctness | Frozen project metric |
| HotpotQA EM | Exact normalized answer match | HotpotQA | 1–3 | Yes | Secondary correctness | Frozen project metric |
| ASQA Disambig-F1 / official QA-F1 output | Official disambiguated answer correctness; two names for the same frozen primary output, not two selectable metrics | ASQA | 1–3 | Yes | Primary correctness | Frozen project metric |
| ASQA answer-side alias coverage | Explicit answer-aspect coverage | ASQA | 1–3 | Yes | Diagnostic | Project-selected/frozen |
| Faithfulness@5 | Atomic-claim support by exact five supplied passages | All `WITH_CONTEXT` | 1–3 | Yes | Primary groundedness construct | Kickoff requirement/project-selected instrument |
| Faithfulness component states and verification coverage | Diagnose unsupported, contradicted, unverifiable, and evaluator-missing claims | All `WITH_CONTEXT` | 1–3 | Yes | Diagnostic | Frozen project instrument |
| Supporting-evidence/qrel coverage | Complementary evidence coverage | HotpotQA | 1–3 | Yes | Secondary/diagnostic | Dataset-specific project metric |
| Mean pairwise cosine distance, top-5 | Generic retrieval-diversity manipulation check using normalized Contriever-space embeddings | All | 1–3 | Yes | Diagnostic | Project-selected; historically used in Sprint 2 |
| ACC_bi and directional components | Bidirectional substantive output change, not answer quality | All paired baseline/diversified `WITH_CONTEXT` | 2–3/final | Yes | Registered secondary/mechanistic | Frozen project metric |
| Jaccard@20 and Jaccard@5 | Original/paraphrase retrieval robustness | Development pilot, all datasets | 2/final | Yes | Diagnostic | Optional kickoff suggestion; frozen pilot |
| Ordered top-5 identity | Strict paraphrase robustness diagnostic | Development pilot | 2/final | Yes | Diagnostic | Frozen pilot |
| Generation/evaluation status | Preserve `OK`, `REFUSAL`, `PARSE_FAILURE`, `TRUNCATED`, `ERROR` and missingness | All | 1–3 | Yes | Required diagnostic | Frozen generation/evaluation contract |

Historical PubMedQA long-answer EM/token-F1/ROUGE-L and historical Sprint-2
NLI proxy metrics remain historical only; they are not canonical replacements
for the metrics above.

## 8. Statistical analysis plan

The analysis unit is the question/sample ID. Every comparison fixes dataset,
evidence role, sample manifest, corpus, retriever, LLM, prompt, decoding,
candidate top-20, evaluation version, and all other factors except the tested
factor. Use exact same-ID pairing and pairwise complete cases; never impute or
turn missing evaluator outputs into zero. Also report expected N, measured N,
paired N, and status/failure counts against the full manifest.

For every important comparison report:

1. baseline/reference mean;
2. method/treatment mean;
3. mean paired difference (`treatment - reference`);
4. per-query paired values and variation (SD plus distribution/quantiles);
5. ordinary 95% paired percentile-bootstrap CI, 10,000 resamples, base seed
   20260823 and the frozen deterministic per-contrast seed derivation;
6. two-sided paired sign-flip/permutation test: exact when the complete sign
   space is at most 10,000 patterns, otherwise 10,000 Monte Carlo sign flips,
   with the frozen SHA-256 per-contrast seed;
7. raw p-value; and
8. Holm-adjusted p-value within the predefined family.

The paired test and Holm method are already authoritatively frozen; they are
not pre-execution decisions. Ordinary CIs are not simultaneous. A protected
formal multiplicity-controlled significance statement requires adjusted
`p < .05`; effect magnitude, CI, and failures remain visible. P-values do not
select methods. No numerical margin or tolerance is used.

### Comparison families

- **Sprint 1 WITH vs WITHOUT:** one family per dataset × LLM, containing the
  four retriever-specific relevance-baseline contrasts. For protected
  HotpotQA/ASQA this is the frozen RQ1 family; Sprint-1 development and PubMedQA
  reporting mirrors it descriptively.
- **Sprint 1 retriever comparisons:** no retriever-superiority confirmatory
  family is frozen and no winner is selected. If inferential pairwise retriever
  tests are still desired, exact hypotheses and Holm membership are a
  **PRE-EXECUTION DECISION**; otherwise report means/effects/CIs descriptively.
- **Sprint 2 diversification vs relevance baseline:** keep retrieval families
  per dataset × retriever and downstream correctness/faithfulness families per
  dataset × retriever × LLM, with all frozen treatment-versus-baseline tests in
  their family. Development/selection inference must be labelled as such and
  cannot expose protected outcomes.
- **Sprint 3 final:** use the frozen 62 Holm families: 6 RQ1 correctness, 8 RQ2
  retrieval, 24 RQ2 correctness, and 24 RQ5 faithfulness. Do not create one
  416-test family, cross-dataset pooled tests, treatment-vs-treatment
  confirmatory tests, or new families after viewing outcomes.

## 9. Required raw evidence chain

For every generated row, the scientific artifacts must reconstruct:

```text
dataset; split/evidence role; sample ID; original question;
gold answer and gold evidence/aspects where available;
retriever and immutable index/corpus identity;
ordered top-20 candidates, raw scores, exact document IDs and exact text;
selection/diversification method, parameters, seed and implementation identity;
ordered selected top-5 and complete lineage to the top-20;
exact rendered context; exact system/user prompt and hashes;
logical LLM slot; physical LLM/provider/revision binding; decoding;
raw response; finish reason; attempts; generation status;
parsed correctness outputs and evaluator identity/status;
faithfulness outputs and claim/passage decisions;
coverage and retrieval-diversity outputs;
ACC/output-change outputs where relevant;
run ID, Git commit/dirty state, manifests, environment/hardware,
timestamps, hashes, row counts and failure provenance
```

**NEVER STORE ONLY AGGREGATE MEANS.** Aggregates, tables, plots, notebooks, and
claims must be reproducible from immutable row-level evidence. Preserve exact
text in governed artifacts where licensing permits; otherwise preserve the
authoritative reconstructable identity plus hashes and access provenance.

## 10. Result storage structure

New controlled work prospectively follows this simple structure. Existing
historical evidence stays in place and is not renamed merely for consistency.

```text
artifacts/candidates/<dataset>/
artifacts/selected_contexts/<dataset>/
artifacts/generations/<dataset>/
artifacts/evaluations/<dataset>/

results/sprint1/<dataset>/
results/sprint2/<dataset>/
results/sprint3/<dataset>/
results/final/
```

Within each new dataset/sprint run directory, write as applicable:

```text
retrieval_per_query.parquet
generation_master.parquet
evaluation_per_query.parquet
paired_statistics.parquet
summary_metrics.csv
figures/
run_manifest.json
```

Parquet is the analysis surface, not a substitute for immutable raw response,
candidate, context, evaluator, and registry artifacts. Use stable IDs and
configuration/run keys across files. Never mix evidence roles in an unlabeled
table or silently overwrite a prior run.

## 11. Notebook plan

Create separate, professor-facing dataset notebooks that load saved evidence:

```text
notebooks/sprint1/
  01_pubmedqa_baseline.ipynb
  02_hotpotqa_baseline.ipynb
  03_asqa_baseline.ipynb
  04_sprint1_summary.ipynb
notebooks/sprint2/
  01_pubmedqa_diversification.ipynb
  02_hotpotqa_diversification.ipynb
  03_asqa_diversification.ipynb
  04_sprint2_summary.ipynb
notebooks/sprint3/
  01_pubmedqa_final.ipynb
  02_hotpotqa_final.ipynb
  03_asqa_final.ipynb
  04_sprint3_summary.ipynb
notebooks/final/
  01_cross_sprint_analysis.ipynb
```

This prospective plan supersedes the older four-principal-notebook plan as an
organization choice, without deleting or changing existing notebooks. Keep
Markdown short and non-technical, tables clear, matplotlib figures readable,
and plain-English interpretation directly below important results. Keep
implementation/provenance concise, show no giant raw outputs, and never type
result numbers manually. Scientific evidence lives outside notebooks.

## 12. Standard notebook result sections

Each dataset notebook should use, where applicable:

1. Research question
2. Dataset/setup and evidence role
3. Experimental design
4. Retrieval results
5. Generation/correctness results
6. `WITH_CONTEXT` vs `WITHOUT_CONTEXT`, or baseline vs diversification
7. LLM comparison
8. Retriever comparison
9. Faithfulness, coverage, and diversity
10. Paired CI/significance results
11. Failure/status analysis
12. A small number of preselected useful qualitative examples
13. Plain-English conclusion with claim limitations

Qualitative examples must not be cherry-picked to replace the quantitative
result. Define their selection rule before inspecting protected responses.

## 13. Sprint completion gates

### Sprint 1 complete only when

- for every planned dataset, the permitted sample/corpus manifests and all four
  retrieval inputs are complete, identity-validated, and provenance-bound;
- every relevance-only top-20 and ordered top-5 is complete or has an explicit
  governed failure row;
- all planned three-LLM `WITH_CONTEXT` and shared `WITHOUT_CONTEXT` generations
  are complete with statuses;
- correctness and required baseline faithfulness/coverage are complete;
- row-level retrieval, generation, evaluation and status tables are saved;
- predefined paired CIs/tests and multiplicity fields are complete;
- figures, three dataset notebooks, and Sprint-1 summary notebook are complete;
- counts/hashes/recomputations are validated and results are frozen separately
  from historical evidence.

### Sprint 2 complete only when

- identical top-20 lineage is verified for baseline and all methods;
- all frozen MMR, clustering, deterministic DPP, and stochastic sensitivity
  retrieval artifacts are complete with seed-level evidence;
- lambda-1 equivalence and implementation-validity gates pass;
- development/selection metrics, diversity, coverage and failure reports are
  complete per query without protected-final exposure;
- required Stage-B three-LLM correctness, faithfulness, ACC and status evidence
  is complete;
- the supervisor-corrected global clustering selection rule was frozen before
  outcomes and produces exactly one KMeans k and one Agglomerative k;
- paired statistics, figures, dataset notebooks and summary notebook are
  complete; selected configuration IDs and all Sprint-2 evidence are frozen.

### Sprint 3 complete only when

- every protected-entry gate passed before exposure and protected manifests
  remained closed until authorization;
- the exact 3-dataset × 4-retriever × 8-context × 3-LLM matrix plus shared
  no-context cells is complete or explicitly status-accounted;
- no tuning or protocol change occurred after protected evidence opened;
- all correctness, retrieval, diversity/coverage, `Faithfulness@5`, ACC and
  failure outputs are complete at row level;
- all 416 registered protected contrasts, 10,000-bootstrap CIs, paired tests,
  raw p-values and 62-family Holm adjustments are complete where measurable;
- expected/measured/paired N and all failures reconcile to manifests;
- dataset notebooks, summary notebook, figures and plain-English conclusions
  are complete; results are audited and frozen;
- the final cross-sprint notebook completes analyses A–G while maintaining
  historical/control/development/selection/protected evidence labels.

## 14. Current project state

Repository authority as of the latest `PROJECT_STATE.md` checkpoint:

| Status | Current state |
|---|---|
| DONE | Frozen scientific protocols/amendments; immutable historical Sprint-1 and Sprint-2 artifacts; PubMedQA sample/corpus manifests; historical Hotpot sample manifest; hardened run registry; current PubMedQA BM25 and DPR top-20 sets locally complete; generation subsystem implemented without model calls |
| IN PROGRESS | Sprint-1 completion; physical Maki model-ID resolution; co-location/identity validation of RunPod PubMedQA Contriever and ColBERTv2 candidates/index; relevance top-5 context materialization and generation preflight |
| NOT STARTED | Canonical model generations/evaluations; HotpotQA/ASQA full-corpus acquisition/indexing and controlled runs; diversification artifacts under current contracts; selection execution; protected final; requested dataset notebook suite and final synthesis |
| PRE-EXECUTION DECISIONS | Exact HotpotQA `DEVELOPMENT` baseline sample N/manifest; supervisor-corrected outcome-independent global KMeans/Agglomerative selection rule without margins; whether to run Sprint-1 inferential retriever comparisons and, if yes, their hypotheses/Holm families; qualitative-example selection rule before protected inspection |
| BLOCKERS | Full HotpotQA/ASQA corpora and compute/storage/resource gates; external dataset/model access and immutable revisions; physical Maki availability; human work for decomposer selection, NLI calibration, ASQA matcher validation and optional paraphrase pilot; scorer/instrument implementations and calibrations; protected stage gates |

### Immediate execution order

Finish Sprint 1 first. The single next recommended task is the current governed
PubMedQA preflight: resolve and provenance-bind the three physical Maki model
identities, co-locate and identity-validate all four existing PubMedQA candidate
sets and neural indexes, then materialize relevance-only selected contexts.
Only after that should the already-frozen 20-prompt repeatability gate run;
no 1,000-row generation block opens before it passes.

## Authority conflicts and resolutions recorded by this blueprint

- The user's later supervisor correction removes every active margin/tolerance
  rule. Older selection documents retain inactive margin architecture and
  blocker language; future execution must not use it.
- Historical sprint identity is dataset-specific (Sprint 1 PubMedQA, Sprint 2
  HotpotQA), while the completion direction requires all three datasets in the
  baseline/diversification/final story. Historical evidence remains unchanged;
  new cross-dataset work is explicitly prospective.
- The older project blueprint and matrix contain reduced HotpotQA/ASQA fallback
  language. Later amendments require full corpora and authorize no canonical
  fallback.
- Older ASQA notes use 21,015,300; current authority uses 21,015,324 and full
  dev948.
- The earlier four-notebook plan conflicts with the requested dataset-specific
  notebook plan. This blueprint adopts the latter prospectively and preserves
  existing notebooks.
- README/config/legacy pipeline defaults describe historical prompts, models,
  metrics, and two-sprint scope; they are not canonical new-run authority.
