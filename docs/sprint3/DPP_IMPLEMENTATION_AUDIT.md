# Sprint 3 DPP Implementation Audit

## Status

A correctness issue was identified in the historical Sprint 2 stochastic DPP
implementation.

No historical Sprint 2 result files have been modified.

---

## 1. Affected Implementation

File:

`src/diversification/dpp.py`

Function:

`_sample_kdpp`

Historical documentation described this function as an exact k-DPP sampler.

The implementation performs ordinary DPP-style independent eigenvector
selection using probabilities:

`lambda / (1 + lambda)`

and then forces the selected eigenvector set to cardinality `k` using retry,
truncation, or additional random selection.

This is not the general exact fixed-cardinality k-DPP eigenvector-selection
procedure.

---

## 2. Empirical Validation

The historical sampler was compared against exact k-DPP subset probabilities
computed by exhaustive determinant enumeration on small PSD kernels.

A first small test showed only a small discrepancy:

- total variation distance: 0.0093
- errors: 0
- wrong-size outputs: 0

A broader stress test showed material discrepancies.

Observed total variation distances included:

- LOW: 0.0141
- BALANCED: 0.0055
- MIXED: 0.0387
- HIGH: 0.1277
- EXTREME: 0.2749
- SIX_K3: 0.2149

The stress test also produced sampling exceptions in multiple scenarios.

Therefore the historical stochastic implementation cannot be treated as an
exact k-DPP sampler in general.

---

## 3. Sprint 2 Exposure

The affected Sprint 2 conditions are:

- `dpp_seed1`
- `dpp_seed2`
- `dpp_seed3`

Affected retrievers:

- BM25
- DPR
- Contriever
- ColBERTv2

Artifacts:

- 12 raw CSV files
- 500 questions per file
- 6,000 total retrieval rows

Audit confirmed:

- generated answers: 0
- non-zero faithfulness values: 0

Therefore the implementation issue affects historical stochastic-DPP retrieval
results only.

It does not require regeneration of historical LLM answers because none were
generated for these conditions.

---

## 4. DPP MAP

`dpp_map` uses the separate deterministic `_greedy_map` implementation.

The stochastic sampling defect identified in `_sample_kdpp` does not by itself
invalidate `dpp_map`.

The deterministic MAP implementation should still receive its own correctness
tests, but it must not be grouped with the stochastic sampler defect.

---

## 5. Historical Result Policy

Historical Sprint 2 raw files are immutable.

They must not be overwritten or silently corrected.

The existing `dpp_seed1`, `dpp_seed2`, and `dpp_seed3` artifacts should be
described as results from the historical stochastic DPP implementation rather
than as validated exact k-DPP samples.

Corrected results must be written as new Sprint 3 artifacts with explicit
provenance.

---

## 6. Required Sprint 3 Actions

Before relying on stochastic k-DPP in Sprint 3:

1. implement a correct fixed-cardinality k-DPP sampler,
2. add distribution-level correctness tests against exact enumeration,
3. add deterministic seed/reproducibility tests,
4. preserve the historical implementation results,
5. rerun the affected 6,000 retrieval evaluations using the corrected sampler,
6. compare historical and corrected retrieval metrics,
7. determine whether Sprint 2 scientific conclusions change.

If corrected results materially change the project interpretation, the change
must be documented and raised with the supervisor before final reporting.

---

## 7. Claim Boundary

Until the corrected sampler is implemented and validated, Sprint 3 must not
claim that historical `dpp_seed1`, `dpp_seed2`, or `dpp_seed3` results represent
exact k-DPP sampling.


---

## 8. Reference and Audit Provenance

### Primary theoretical reference

Kulesza, A. and Taskar, B. (2012).
*Determinantal Point Processes for Machine Learning*.
Foundations and Trends in Machine Learning.
arXiv:1207.6083.

The k-DPP treatment conditions the model on fixed cardinality and requires
dedicated inference/sampling machinery based on elementary symmetric
polynomials of the eigenvalues.

This audit therefore distinguishes:

- ordinary DPP eigenvector selection,
- exact fixed-cardinality k-DPP sampling,
- greedy MAP subset selection.

These procedures must not be described interchangeably.

### Repository provenance

Audit date: 2026-08-08

Repository HEAD before any corrective DPP implementation:

`a94315eed1d364d894327f7234dc555822e92837`

The audit was performed on the frozen historical Sprint 2 implementation
present at this repository state.

No corrective changes to `src/diversification/dpp.py` had been made when the
audit evidence above was collected.
