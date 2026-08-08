# Sprint 3 Method Selection Protocol

## Purpose

This protocol defines how diversification methods and hyperparameter
configurations will be selected in Sprint 3.

The purpose is to ensure that method selection is:

- evidence-based,
- reproducible,
- transparent,
- resistant to cherry-picking,
- separated from final evaluation.

A method will not be selected merely because it produces the highest observed
value on one metric.

---

## 1. General Principle

Sprint 3 separates three activities:

1. **Method development**
2. **Method selection**
3. **Final evaluation**

The final evaluation data must not be used to choose:

- diversification family,
- hyperparameters,
- top-k,
- candidate-pool size,
- thresholds,
- metric weights,
- statistical criteria.

Any decision that affects method selection must be fixed before inspecting
final evaluation results.

---

## 2. Baseline

The relevance-only retrieval condition is always retained as the primary
baseline.

The baseline is not removed even if a diversification method performs better
during method selection.

Every final diversification result must be interpretable relative to this
baseline.

---

## 3. Candidate Method Families

The currently approved candidate families are:

### Relevance-only baseline

- no diversification

### MMR

Existing candidate lambda values:

- 0.00
- 0.25
- 0.50
- 0.75
- 1.00

### Clustering

Existing candidate configurations:

- K-Means k=2
- K-Means k=3
- K-Means k=5
- Agglomerative k=3
- Agglomerative k=5

### DPP

Existing candidate configurations:

- deterministic DPP MAP
- stochastic DPP seed 1
- stochastic DPP seed 2
- stochastic DPP seed 3

These candidates originate from the existing Sprint 2 experimental design.

Adding new method families or hyperparameters requires an explicit documented
reason before final evaluation.

---

## 4. Implementation Validity Gate

Before any method or configuration is eligible for method selection, its
implementation must be validated against the intended algorithmic definition.

This validation should confirm, where applicable:

- the implemented algorithm matches the method described in the report,
- hyperparameters have the intended interpretation,
- deterministic and stochastic behavior is correctly represented,
- random seeds affect only intended stochastic components,
- candidate-set and top-k constraints are applied correctly,
- no implementation artifact is being interpreted as scientific evidence.

### DPP-specific requirement

Before stochastic DPP results are used for Sprint 3 method selection, the
sampling implementation must be audited against the intended k-DPP procedure.

The audit must establish whether the implementation actually samples a
fixed-size set according to the intended formulation.

If the implementation differs from the intended algorithm:

1. the existing results must be clearly labeled according to what was actually
   implemented,
2. the discrepancy must be documented,
3. affected configurations must either be corrected and rerun or excluded from
   method-selection evidence.

Historical Sprint 2 raw outputs must not be modified to conceal or repair an
implementation discrepancy.

Implementation validation evidence should be recorded before final method
selection.

---

## 5. Method Selection Unit

Selection must distinguish between:

- diversification **family**, and
- diversification **configuration**.

For example:

- MMR is a family,
- MMR lambda=0.75 is a configuration.

The analysis must not claim that an entire family is superior solely because
one configuration performs well.

---

## 6. Dataset Separation

Method selection must use designated development/selection data.

Final evaluation data must remain held out from selection decisions.

The exact dataset splits and sampling procedure will be defined in:

`docs/sprint3/DATASET_PROTOCOL.md`

Until that protocol is finalized, no ASQA final-evaluation run should be used
for method selection.

---

## 7. Two-Stage Selection Procedure

Method selection will use two stages.

### Stage A — Retrieval screening

Configurations are first evaluated on retrieval behavior.

Relevant dimensions include:

- retrieval relevance,
- retrieval diversity,
- aspect coverage where available.

Candidate metrics may include:

- Recall@k,
- MRR@k,
- NDCG@k,
- retrieval diversity,
- alpha-nDCG,
- subtopic/aspect coverage.

The exact metric definitions will be fixed in:

`docs/sprint3/METRICS_PROTOCOL.md`

A configuration that increases diversity but causes severe relevance collapse
must not automatically proceed simply because diversity increased.

---

### Stage B — Downstream RAG evaluation

Configurations that remain scientifically plausible after retrieval screening
are evaluated using downstream answer behavior.

Relevant dimensions may include:

- answer correctness,
- faithfulness,
- coverage,
- supportedness,
- output diversity where applicable.

Retrieval diversity alone cannot determine the final selected method.

---

## 8. Pareto Analysis

Sprint 3 will use Pareto analysis where objectives conflict.

Examples of competing objectives include:

- retrieval diversity vs. Recall@k,
- aspect coverage vs. relevance,
- answer correctness vs. hallucination/faithfulness,
- output diversity vs. correctness.

A configuration is Pareto-dominated if another configuration is at least as
good on all selected objectives and strictly better on at least one.

Dominated configurations should normally not be selected unless a documented
scientific reason exists.

The Pareto frontier will be reported rather than hiding trade-offs behind a
single aggregate score.

---

## 9. Statistical Uncertainty

Question-level comparisons should preserve pairing between methods.

Where appropriate, Sprint 3 should use:

- paired differences,
- paired bootstrap confidence intervals,
- confidence intervals around effect estimates.

The exact bootstrap procedure, number of resamples, random seed, and reported
interval will be fixed before final analysis.

Statistical uncertainty must be considered when differences are small.

A numerically larger mean is not sufficient evidence of meaningful
superiority.

---

## 10. Non-Inferiority / Acceptable-Loss Constraints

Where diversification improves one objective while reducing another,
selection may use pre-defined acceptable-loss constraints.

For example:

- improved aspect coverage while preserving retrieval relevance within an
  acceptable margin,
- improved answer quality without a meaningful faithfulness reduction.

Any non-inferiority margin or acceptable-loss threshold must:

1. be defined before final evaluation,
2. have a documented rationale,
3. not be chosen after viewing final results.

Exact margins are not fixed in this document yet.

They must be finalized in the metric/statistical protocol before final runs.

---

## 11. Selection Hierarchy

Unless later justified and documented before final evaluation, method selection
will follow this hierarchy:

1. Reject scientifically invalid or implementation-invalid configurations.
2. Reject configurations showing catastrophic relevance or quality collapse.
3. Identify non-dominated configurations using the pre-defined objectives.
4. Compare downstream answer quality among eligible configurations.
5. Consider statistical uncertainty for close comparisons.
6. Prefer the simpler configuration when evidence does not clearly distinguish
   two configurations.
7. Record the final selection decision and evidence.

This hierarchy prevents selection based on one convenient metric.

---

## 12. Simplicity Principle

If two configurations are practically indistinguishable within uncertainty,
prefer the simpler or more deterministic configuration.

Examples of relevant simplicity considerations include:

- deterministic behavior,
- fewer hyperparameters,
- lower computational cost,
- easier reproducibility.

Simplicity must not override a clear performance disadvantage.

---

## 13. Stochastic Methods

Stochastic methods require additional care.

A stochastic method must not be judged from one favorable random seed.

Where stochastic DPP or another stochastic method is considered:

- multiple seeds must be examined,
- variability must be reported,
- seed-specific results must not be presented as if they represent the method
  generally.

A deterministic alternative may be preferred when average performance is
similar but stochastic variance is substantial.

---

## 14. Existing Sprint 2 Evidence

Sprint 2 results may be used as prior empirical evidence and for identifying
questions that require further testing.

However:

- Sprint 2 observations must not be silently converted into Sprint 3 selection
  rules after the fact,
- Sprint 2 HotpotQA results represent a multi-hop/complementary-evidence regime,
- they do not establish performance on aspect-diversity tasks such as ASQA.

Sprint 2 evidence should therefore inform hypotheses, not predetermine the ASQA
conclusion.

---

## 15. Required Selection Evidence Package

For every method/configuration considered for final selection, the project
should preserve enough evidence to reconstruct the decision.

The evidence package should contain, where applicable:

- method family,
- configuration,
- dataset and split,
- sample definition,
- retrieval metrics,
- diversity metrics,
- downstream metrics,
- paired effect estimates,
- confidence intervals,
- Pareto status,
- runtime/cost information where relevant,
- implementation validity status,
- selection decision,
- reason for selection or rejection.

---

## 16. Decision Log

Every major selection decision must be recorded in:

`docs/sprint3/DECISION_LOG.md`

A decision entry should state:

- date,
- decision,
- evidence used,
- alternatives considered,
- reason,
- affected experiments,
- Git commit where applicable.

---

## 17. Prohibited Selection Practices

The following are not allowed:

- choosing the best configuration after inspecting final evaluation results,
- changing the primary metric because another method wins,
- reporting only favorable random seeds,
- dropping an unfavorable retriever without methodological justification,
- changing top-k or candidate-pool size after seeing final results,
- treating tiny numerical differences as meaningful without uncertainty,
- selecting a method solely because it maximizes retrieval diversity,
- hiding dominated or failed configurations from the analysis,
- retroactively describing an exploratory choice as pre-planned.

---

## 18. Final Reporting

The final report should show not only which method was selected, but also
**why**.

Where practical, report:

- baseline performance,
- candidate configurations,
- trade-off plots,
- Pareto frontier,
- uncertainty intervals,
- rejected alternatives,
- final selection rationale.

The goal is that an external reviewer can independently understand the
selection from the evidence shown.

---

## 19. Decisions Still To Be Finalized

Before final Sprint 3 evaluation, the following must still be fixed:

- exact development/selection/final dataset split,
- ASQA sampling procedure,
- final metric set,
- primary and secondary selection objectives,
- bootstrap procedure,
- bootstrap resample count,
- statistical seed,
- confidence level,
- acceptable-loss/non-inferiority margins,
- exact rule for context-size sensitivity experiments.

These decisions must be completed before final evaluation results are
inspected.

---

## Status

**Protocol framework established.**

This document defines the method-selection principles but intentionally leaves
dataset-specific thresholds and statistical parameters unresolved until the
corresponding dataset and metric protocols are finalized.
