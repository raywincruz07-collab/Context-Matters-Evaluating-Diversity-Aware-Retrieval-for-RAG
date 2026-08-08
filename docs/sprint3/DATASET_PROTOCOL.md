# Sprint 3 Dataset Protocol

## Purpose

This protocol defines how datasets, splits, samples, corpora, identifiers, and
dataset-derived artifacts are handled during Sprint 3.

The goals are:

- reproducibility,
- prevention of evaluation leakage,
- consistent comparison across retrievers,
- preservation of dataset provenance,
- clear separation between development and final evaluation.

No dataset role, split assignment, sampling rule, or final-evaluation sample
may be changed after final evaluation has begun unless the change is documented
as a protocol deviation.

---

## 1. Dataset Roles Across Sprints

Sprint 3 interprets results across three information-need regimes.

### PubMedQA

Role:

- Sprint 1 single-source/focused-evidence control.

Existing validated Sprint 1 results should be reused where appropriate.

Sprint 3 should not rerun PubMedQA simply to obtain a more favorable result.

---

### BEIR HotpotQA

Role:

- Sprint 2 multi-hop/complementary-evidence benchmark.

Existing Sprint 2 setup:

- test queries available in BEIR: 7,405,
- sampled queries: 500,
- sampling seed: 42,
- negatives per sampled query: 50,
- final pooled corpus: 25,997 passages,
- top-k: 5,
- candidate pool: 20 for diversification experiments.

Historical Sprint 2 artifacts remain immutable.

Sprint 3 extensions must clearly distinguish new runs from historical Sprint 2
runs.

---

### ASQA

Role:

- Sprint 3 ambiguous/multi-aspect information-need benchmark.

Dataset source:

`din0s/asqa`

Observed Hugging Face splits:

- train: 4,353 questions,
- dev: 948 questions.

Observed structural properties include:

- 2–6 disambiguated QA pairs per question,
- unique sample IDs,
- no questions with zero QA pairs,
- no duplicate sample IDs in either split.

---

## 2. ASQA Split Assignment

Sprint 3 assigns:

### Train split

Purpose:

- implementation development,
- debugging,
- smoke experiments,
- method development,
- hyperparameter selection,
- method selection,
- context-size selection,
- threshold selection.

ASQA train data may be inspected during development.

---

### Dev split

Purpose:

- protected final evaluation.

The ASQA dev split must not be used to choose:

- diversification family,
- diversification configuration,
- MMR lambda,
- clustering k,
- DPP configuration,
- top-k,
- candidate-pool size,
- prompt wording,
- generator settings,
- selection thresholds,
- statistical thresholds,
- acceptable-loss margins,
- primary metric definitions,
- metric weights.

Final evaluation results must not trigger retrospective tuning.

---

## 3. ASQA Final-Evaluation Unit

The preferred final evaluation unit is the official ASQA dev split.

Target size:

- 948 questions.

If resource constraints make full evaluation infeasible, any reduced final
sample must be defined before result inspection.

A reduced final sample must use:

- a fixed random seed,
- deterministic sample generation,
- stable `sample_id` values,
- a saved sample manifest,
- no result-dependent filtering.

A reduced sample must not be created by selecting easier questions or questions
with more complete annotations.

The reason for reducing the sample must be documented.

---

## 4. ASQA Development Sampling

Small ASQA train subsets may be created for:

- unit/integration development,
- smoke tests,
- runtime estimation,
- retrieval debugging,
- generator debugging.

Development subsets must never be described as final ASQA benchmark results.

Every sampled development subset should record:

- source split,
- number of examples,
- sampling seed,
- ordered sample IDs,
- creation timestamp,
- Git commit.

Where practical, development samples should preserve a reasonable spread of
questions with different numbers of disambiguated QA pairs.

---

## 5. ASQA Retrieval Corpus

Final ASQA retrieval will use the shared DPR Wikipedia passage corpus.

Corpus definition:

- Wikipedia snapshot date: 2018-12-20,
- total passages: 21,015,300,
- passage segmentation: blocks of up to 100 words.

The same corpus must be used for:

- BM25,
- DPR,
- Contriever,
- ColBERTv2.

This ensures that retriever comparisons are not confounded by different
corpora.

---

## 6. Oracle Information Prohibition

ASQA provides reference information including:

- disambiguated QA pairs,
- short answers,
- supplied contexts for some aspects,
- Wikipedia page information for some aspects,
- long-answer annotations.

These annotations may be used for:

- reference-answer evaluation,
- aspect-definition/evaluation where methodologically valid,
- analysis,
- error inspection after permitted evaluation.

They must not be used to create an oracle retrieval corpus for the final
open-domain retrieval experiment.

In particular, the final corpus must not be restricted to:

- known relevant pages,
- ASQA-provided pages,
- gold contexts,
- pages selected from answer annotations.

---

## 7. Missing ASQA Context Encoding

The value:

`No context provided`

is a missing-context placeholder.

It must not be interpreted as:

- evidence,
- a valid passage,
- a relevant document,
- a reference context.

Observed audit:

- approximately 45% of QA/aspect pairs have supplied page/context grounding,
- therefore ASQA contexts are incomplete document-level relevance judgments.

Metrics that require complete document-level judgments must account for this
limitation explicitly.

---

## 8. Reduced-Corpus Experiments

Reduced corpora are permitted only for:

- code development,
- integration testing,
- smoke runs,
- profiling,
- performance estimation.

Reduced-corpus results must be labeled clearly.

They must not be presented as equivalent to full-corpus ASQA retrieval.

A reduced corpus must not be created by injecting gold pages for final
evaluation questions unless the run is explicitly labeled as an oracle or
diagnostic experiment.

Oracle/diagnostic experiments must remain separate from the primary benchmark.

---

## 9. Corpus Integrity

Before full ASQA experiments, the corpus preparation pipeline should verify:

- expected corpus identity,
- expected snapshot/version,
- expected passage count where applicable,
- stable document identifiers,
- no accidental duplication introduced by preprocessing,
- consistent corpus content across retrievers.

Where feasible, record:

- source dataset/repository,
- dataset revision or commit,
- file hashes or manifest hashes,
- passage count,
- preparation script version,
- Git commit.

---

## 10. Stable Identifiers

Every ASQA question must retain its original:

`sample_id`

Experiment outputs must contain enough information to map every row back to the
original ASQA example.

Do not rely only on sequential dataframe row numbers.

Where applicable, preserve:

- ASQA sample ID,
- original split,
- internal question index,
- retrieved document IDs.

---

## 11. Sampling Reproducibility

Every random sampling operation must use an explicitly recorded seed.

Do not use implicit global randomness for dataset construction.

A sampling operation should be reproducible from:

- dataset source,
- dataset version,
- split,
- seed,
- requested sample size,
- sampling algorithm.

Sample manifests should be generated once and reused rather than resampling
independently for each retriever.

---

## 12. Retriever Fairness

All retrievers compared within the same experiment must receive:

- the same query set,
- the same corpus,
- the same final evaluation sample,
- the same top-k definition,
- the same candidate-pool definition where applicable.

Retriever-specific model preprocessing is allowed where required by the model,
but dataset membership must remain unchanged.

---

## 13. No Result-Dependent Filtering

Questions must not be removed because:

- one retriever performs poorly,
- generation fails for a particular method,
- an answer is difficult,
- an annotation is inconvenient,
- a result weakens the expected hypothesis.

Any exclusion must have a pre-defined methodological or technical reason.

All exclusions must be counted and documented.

---

## 14. Error Handling

Failed rows must not silently disappear.

Experiment outputs should preserve:

- `row_status`,
- `row_error` or equivalent error information.

Failures must be distinguishable from genuine metric values.

A missing or uncomputed metric must be represented as missing/NA, not `0.0`.

---

## 15. Historical Sprint 2 Data

Finalized Sprint 2 raw outputs are immutable.

They must not be rewritten to:

- change missing-value encoding,
- alter generated answers,
- repair historical implementation issues,
- update metrics retrospectively.

If a Sprint 2 issue is discovered:

1. document the issue,
2. preserve historical files,
3. create a new corrected Sprint 3 run if necessary,
4. clearly distinguish historical and corrected results.

---

## 16. Dataset Provenance Per Run

Each experiment run should record, where applicable:

- run ID,
- Git commit,
- dataset name,
- dataset revision/version,
- split,
- sample size,
- sampling seed,
- sample-manifest identifier,
- corpus name,
- corpus version/snapshot,
- corpus passage count,
- corpus manifest/hash where available,
- retriever,
- diversification configuration,
- top-k,
- candidate-pool size,
- timestamp,
- hardware/environment.

---

## 17. Dataset Manifests

Sprint 3 should maintain machine-readable manifests for important fixed samples
and corpora.

Suggested location:

`configs/sprint3/`

Examples may include:

- ASQA development sample manifest,
- ASQA final sample manifest if subsampling becomes necessary,
- corpus metadata manifest.

Manifests should contain identifiers and metadata, not large raw datasets.

---

## 18. Data Storage

Large or regenerable data should remain outside Git history.

The repository may store:

- configuration,
- manifests,
- hashes,
- metadata,
- preparation code.

Large corpora, embeddings, model indexes, and caches should remain ignored or
stored in the compute environment.

---

## 19. Local vs RunPod Data

Local WSL may be used for:

- small development samples,
- metadata inspection,
- unit/integration testing.

RunPod will be used for:

- full Wikipedia corpus preparation where required,
- large embedding generation,
- retriever indexes,
- GPU-heavy retrieval,
- large Sprint 3 experiments.

Both environments must use the same committed experiment configuration.

---

## 20. Protocol Deviations

Any change to this protocol after final evaluation begins must be recorded in:

`docs/sprint3/DECISION_LOG.md`

The record must include:

- what changed,
- why it changed,
- when it changed,
- whether final results had already been inspected,
- which experiments are affected.

---

## 21. Decisions Still Open

This protocol does not yet finalize:

- whether all 948 ASQA dev questions will be generated through the LLM,
- final ASQA aspect-level metric definitions,
- exact development subset sizes,
- final top-k values,
- final candidate-pool values,
- final statistical analysis procedure.

These will be fixed in the metrics and experiment protocols before final
evaluation.

---

## Status

Dataset protocol established.

ASQA train is reserved for development/method selection.

ASQA dev is protected for final evaluation.

The full DPR Wikipedia 2018-12-20 passage corpus is the target shared retrieval
corpus for the primary ASQA benchmark.
