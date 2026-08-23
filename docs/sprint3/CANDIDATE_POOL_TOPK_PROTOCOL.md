# Sprint 3 Candidate-Pool and Final Top-K Protocol

## 1. Purpose

This document freezes the canonical first-stage candidate-pool size and final
retrieval/context cardinality for Sprint 3 before the diversification matrix and
HotpotQA resource benchmark are frozen.

## 2. Canonical Values

The following values are frozen globally:

```text
candidate_pool = 20
final_top_k    = 5
```

They apply to all three datasets:

- PubMedQA;
- HotpotQA;
- ASQA.

They apply to all four retrievers:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

They apply to all diversification families:

- relevance-only baseline;
- MMR;
- KMeans;
- Agglomerative;
- DPP.

Candidate-pool or final-top-k values must not be selected separately by dataset
or retriever.

## 3. Final Top-K Rationale

`final_top_k = 5` is frozen because it is the coherent cardinality across the
already-frozen methodology:

- canonical `WITH_CONTEXT` generation uses exactly five passages;
- the ASQA lead retrieval constructs are S-recall@5 and alpha-nDCG@5;
- faithfulness evaluates claims against the actual five supplied passages;
- clustering-configuration interpretation is defined relative to output
  cardinality five;
- DPP selects fixed cardinality five;
- Sprint 1 and historical Sprint 2 both used five final passages.

Changing top-k would reopen multiple already-frozen constructs without a
substantive scientific requirement. Project conclusions must be scoped
explicitly to five-passage RAG.

## 4. No Top-K Sweep

No top-k or context-window-size sweep will be run.

The kickoff treated context size as a possible experimental factor, not a
mandatory full-factorial dimension. A top-k sweep would:

- alter the generator evidence surface;
- alter the @5 retrieval constructs;
- change the faithfulness evidence;
- change the clustering interpretation;
- multiply generation and evaluation conditions.

Therefore top-k is a controlled constant, not a treatment variable.

## 5. Candidate-Pool Rationale

`candidate_pool = 20` is frozen prospectively.

The primary rationale is:

- the final output is five passages, so pool 20 provides a four-times-larger
  choice set;
- every method has 15 alternatives beyond the final selected five;
- this provides meaningful opportunity to diversify without making the
  candidate layer unnecessarily large;
- KMeans and Agglomerative `k <= 5` and fixed-cardinality-five DPP are naturally
  supported;
- artifact size and embedding/reranking work remain tractable;
- DPP cost grows substantially as pool size grows.

Historical Sprint 2 use of pool 20 and existing PubMedQA implementations are
supporting reproducibility and comparability considerations only.

`candidate_pool = 20` is **not** chosen because historical retrieval or answer
performance was best at 20.

## 6. Shared Candidate Contract

For every `(dataset, evidence role/split, sample, retriever)`, candidate
production must produce exactly one ranked candidate artifact containing 20
unique, valid documents, unless candidate production fails.

That exact ordered artifact feeds:

- the baseline;
- every MMR lambda;
- every KMeans k;
- every Agglomerative k;
- deterministic DPP;
- any stochastic DPP development sensitivity.

The baseline selects candidate ranks 1 through 5 from that exact artifact.

Every diversifier selects exactly five unique documents from that exact
artifact.

Diversifiers must never:

- re-query the retriever;
- request a deeper candidate pool;
- request a different candidate pool;
- reorder the first-stage artifact before diversification;
- receive gold or query-specific supplemental documents.

Record at minimum:

- `candidate_pool = 20`;
- `final_top_k = 5`;
- ordered document IDs;
- native retrieval scores;
- ranks;
- candidate-artifact identity and hash;
- query and sample identity;
- corpus identity;
- retriever configuration and provenance.

## 7. Short-Candidate Failure Policy

For canonical pool-20 experiments, if candidate production returns fewer than
20 **unique, valid, finite-score documents**, classify that query/retriever as a
retrieval-production failure.

The recommended diagnostic reason is:

```text
SHORT_CANDIDATE_LIST
```

This is not a successful analyzable candidate artifact.

Do not:

- diversify a pool of 5 through 19 documents anyway;
- duplicate documents;
- inject gold documents;
- borrow another retriever's candidates;
- replace the query;
- convert the failure into metric zero.

Retrieval-dependent generations for that query/retriever are missing or failed
according to the later analysis protocol. Failures remain recorded for
failure-rate reporting.

## 8. Development-Only Candidate-Pool Sensitivity

One bounded interpretive sensitivity is frozen:

```text
candidate_pool in {10, 20, 50}
final_top_k = 5
```

Its purpose is to assess whether qualitative retrieval and diversification
behavior depends strongly on the amount of first-stage choice.

This sensitivity is not:

- candidate-pool tuning;
- method selection;
- a protected-final treatment;
- a reason to select whichever pool scores highest.

The canonical main-study pool remains 20 regardless of ordinary sensitivity
performance. Only a genuine validity defect could reopen that value, and doing
so would require a prospective methodology amendment before protected-final
work.

## 9. Sensitivity Evidence

The sensitivity uses development or exposed evidence only:

### PubMedQA

- historical/exposed PQA-L control evidence.

### HotpotQA

- BEIR train / `DEVELOPMENT` only.

### ASQA

- internal `DEVELOPMENT` only.

The sensitivity must never use:

- the HotpotQA protected-final sample;
- ASQA dev948;
- selection outcomes.

## 10. Sensitivity Query Sample

Use exactly 100 fixed development/exposed queries per dataset.

The same 100 IDs must be reused across:

- all candidate pools;
- all four retrievers;
- all sensitivity diversification methods.

The prospective hash priority is:

```text
priority(source_sample_id) =
SHA256(
  UTF8(
    "context-matters-rag|sprint3|CANDIDATE_POOL_SENSITIVITY|20260823|"
    + canonical_dataset_id
    + "|"
    + exact_source_sample_id
  )
).hexdigest()
```

`canonical_dataset_id` must be exactly one of:

```text
pubmedqa
hotpotqa
asqa
```

Sort by `(lowercase_hex_digest, UTF-8 source-ID bytes)` and select the first 100
eligible IDs.

Materialize and hash this manifest only later, before sensitivity execution.

## 11. Sensitivity Methods

At each pool in `{10, 20, 50}`, run these retrieval-only representative
conditions:

- relevance-only baseline;
- MMR `lambda = 0.50`;
- KMeans `k = 3`;
- Agglomerative `k = 3`;
- deterministic DPP-MAP-style selection.

Do not include:

- all MMR lambda values;
- all clustering k values;
- stochastic DPP repetitions;
- generation;
- answer correctness;
- claim decomposition;
- NLI;
- faithfulness.

This sensitivity is intentionally bounded. Any retrieval or diversity summaries
used for interpretation must later use already-frozen canonical metric
definitions or be explicitly labelled descriptive diagnostics. No post-hoc
threshold may be created to decide which pool "wins."

## 12. Resource-Gate Consequence

The canonical HotpotQA full-corpus feasibility and resource benchmark must
measure canonical retrieval/search behavior using:

```text
candidate_pool = 20
```

and downstream output cardinality:

```text
final_top_k = 5
```

Pool-10 and pool-50 sensitivity measurements may be reported separately but
must not replace pool 20 in the canonical feasibility decision.

## 13. Diversification Consequence

Because `final_top_k = 5` is now frozen, the currently proposed diversification
interpretation is valid:

### KMeans development

```text
k = {2, 3, 5}
```

### Agglomerative development

```text
k = {3, 5}
```

### MMR

Every lambda selects exactly five documents from the same 20.

### DPP

Canonical cardinality is five.

### Baseline

The baseline selects candidate ranks 1 through 5 from the same 20.

This document does not itself freeze the final diversification matrix. That is
a separate, immediately downstream methodology decision.

## 14. Implementation Contracts Still Required

This document freezes methodology, not the following engineering work, which is
still required later:

- a HotpotQA canonical candidate producer;
- an ASQA canonical candidate producer;
- a diversified-output artifact schema;
- explicit `final_top_k` provenance;
- generation validation enforcing exactly five passages for successful
  `WITH_CONTEXT` runs;
- sensitivity-manifest materialization.

None of those items is implemented by this document.

## 15. Frozen Before Observation

`candidate_pool = 20` and `final_top_k = 5` are frozen prospectively before
canonical selection or protected-final outcomes are observed.

The `{10, 20, 50}` development-only sensitivity is interpretive and cannot act
as hidden hyperparameter selection.

No sensitivity results or protected-final results are observed while writing
this document.
