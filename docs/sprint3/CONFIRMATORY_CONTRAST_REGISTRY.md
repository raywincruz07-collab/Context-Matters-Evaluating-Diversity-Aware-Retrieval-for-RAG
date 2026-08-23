# Sprint 3 Confirmatory Contrast Registry

## 1. Purpose

This registry prospectively freezes which Sprint-3 analyses are confirmatory
before protected-final outcomes are opened. It does not authorize protected
execution.

It freezes:

- exact confirmatory contrast classes;
- exact contrast-ID grammar;
- reference and treatment conditions;
- primary metric for each contrast class;
- evidence roles;
- dataset, retriever, and LLM strata;
- effect direction;
- confidence-interval policy;
- secondary p-value policy;
- Holm multiplicity families;
- predefined secondary and descriptive analyses;
- the confirmatory/exploratory boundary.

Presence of a condition in
`docs/sprint3/FINAL_EXPERIMENT_MATRIX_PROTOCOL.md` does not automatically make
every possible comparison confirmatory.

## 2. Confirmatory Evidence Roles

Only these datasets provide canonical protected-final confirmatory evidence:

```text
HotpotQA
evidence_role = PROJECT_PROTECTED_FINAL
N = 500

ASQA
evidence_role = PROJECT_PROTECTED_FINAL
N = 948
```

PubMedQA has:

```text
evidence_role = HISTORICAL_OBSERVED_CONTROL_REPLICATION
N = 1,000
```

It is not protected confirmatory evidence. PubMedQA receives the same
structural comparisons for replication and cross-dataset interpretation, but
those analyses are labelled:

```text
REGISTERED_DESCRIPTIVE_CONTROL
```

They are not `PRIMARY_CONFIRMATORY` analyses.

Do not include PubMedQA p-values in protected confirmatory multiplicity
families. Do not call PubMedQA an independent final test.

## 3. Common Treatment Set

For diversification comparisons, freeze this exact ordered treatment set:

```text
T1 = mmr_lambda_0
T2 = mmr_lambda_0_25
T3 = mmr_lambda_0_50
T4 = mmr_lambda_0_75
T5 = kmeans_selected
T6 = agglo_selected
T7 = dpp_map
```

The reference condition for all diversification contrasts is:

```text
relevance_baseline
```

The physical selected `k` values may later replace only the logical
`kmeans_selected` and `agglo_selected` placeholders through the frozen bounded
selection procedure. The contrast IDs do not change when the selected `k`
values become known.

Do not add:

- MMR `lambda=1`;
- non-selected clustering `k`;
- stochastic DPP seeds;
- Spectral;
- xQuAD;
- a newly discovered treatment.

## 4. Common Statistical Rules

All confirmatory comparisons use:

```text
fundamental unit: question / sample_id
pairing: exact same sample_id
effect: mean(treatment - reference)
CI procedure: paired question-level bootstrap
bootstrap resamples: 10,000
base seed: 20260823
CI: 95% percentile
```

For missing values:

- never impute;
- never convert evaluator or generation failure to zero;
- use the exact measurable pairwise intersection for the metric;
- always report expected `N`, measured `N`, paired `N`, and failure counts.

Failure rates remain reported against the full expected manifest. Use the
scientific status and failure semantics from the applicable frozen metric
protocols.

## 5. Deterministic Bootstrap Seed Per Contrast

The frozen base seed remains:

```text
20260823
```

To make analysis independent of contrast execution order, derive a
deterministic per-contrast bootstrap seed from:

```text
SHA256(
  "sprint3_boot_v1|20260823|" + contrast_id
)
```

Freeze conversion as:

- take the first 8 bytes of the raw SHA-256 digest;
- interpret them as an unsigned big-endian integer;
- use that integer as the RNG seed for that contrast.

This is a reproducibility elaboration of the already-frozen base seed. Do not
choose a seed after observing results.

The permutation-test seed remains governed by
`docs/sprint3/SELECTION_STATISTICS_PROTOCOL.md`.

## 6. Primary Interpretation of Paired Effects

All primary metrics here are higher-is-better.

Define:

```text
delta = treatment - reference
```

Interpret:

```text
delta > 0: direction favors treatment
delta < 0: direction favors reference
```

For effect-size interpretation:

- an ordinary 95% CI entirely above zero indicates that the unadjusted
  interval supports a positive effect direction;
- an ordinary 95% CI entirely below zero indicates that the unadjusted
  interval supports a negative effect direction;
- an ordinary 95% CI crossing or touching zero means the effect remains
  uncertain at that interval level.

These ordinary 95% CIs are not simultaneous or multiplicity-adjusted
confidence intervals.

Do not redefine this as a non-inferiority gate. Protected-final evaluation is
not another configuration-selection stage.

Do not apply the professor's `SELECTION` acceptable-loss margins as
superiority thresholds in protected-final scientific reporting.

## 7. P-Value Role

For the confirmatory contrast classes below, p-values are:

```text
SECONDARY_CONFIRMATORY_SUPPORT
```

They are not the primary decision language.

Use the paired sign-flip/permutation procedure already frozen in
`docs/sprint3/SELECTION_STATISTICS_PROTOCOL.md`. Use two-sided p-values. Do not
change to one-sided testing after seeing effect direction.

Apply Holm adjustment exactly within the multiplicity families defined below.
Report both:

- raw `p`;
- Holm-adjusted `p`.

Because every frozen Holm family contains multiple protected contrasts, do
not label an individual contrast a statistically significant confirmatory
improvement or statistically significant confirmatory degradation solely
because its ordinary 95% CI excludes zero or its raw `p < 0.05`.

A formal familywise multiplicity-controlled confirmatory significance
statement requires:

```text
Holm-adjusted p < 0.05
```

within that contrast's predeclared family. Report the signed raw effect and
ordinary 95% CI alongside it. Direction remains determined by the signed raw
effect `treatment - reference`.

The Holm-adjusted p-value remains supporting inferential evidence rather than
a replacement for effect magnitude or its ordinary CI.

Do not:

- invent multiplicity-adjusted confidence intervals;
- change CI construction;
- change Holm families;
- use p-values for configuration selection;
- introduce a global 416-test correction.

No p-value may replace:

- raw effect;
- confidence interval;
- paired `N`;
- failure reporting.

## 8. RQ1: Context Versus No Context

RQ1 asks:

> Does retrieval context improve generated-answer correctness relative to no
> context?

The canonical protected confirmatory comparison is:

```text
relevance_baseline WITH_CONTEXT
minus
WITHOUT_CONTEXT
```

The primary metric is the dataset-specific primary correctness metric:

```text
HotpotQA: official answer F1
ASQA: official ASQA Disambig-F1 (`QA-F1` in the frozen official scorer)
```

This is one ASQA primary correctness metric. `Disambig-F1` is the scientific
metric name, and `QA-F1` is the corresponding official scorer or output
identifier used by the frozen implementation. They are not two alternative
primary metrics.

Faithfulness is not part of RQ1 because `WITHOUT_CONTEXT` has no numeric
faithfulness value. ACC is not part of RQ1. Retrieval metrics are not defined
for `WITHOUT_CONTEXT`.

For each protected dataset, every:

```text
retriever x LLM
```

is one confirmatory contrast instance.

Exact contrast-ID grammar:

```text
RQ1_CORR__{dataset}__{retriever}__{llm}
```

Allowed `dataset` values:

```text
hotpotqa
asqa
```

Allowed `retriever` values:

```text
bm25
dpr
contriever
colbertv2
```

Allowed LLM logical IDs:

```text
llama-3.3-70b
gemma4-26b
ministral-3-14b
```

The `{llm}` token in every contrast ID and multiplicity-family ID must use one
of these frozen logical design-slot identities.

If reserve substitution was validly invoked before the frozen cutoff:

- the logical slot identity remains unchanged for contrast-ID stability;
- the physical model and runtime actually occupying that slot are recorded
  separately in contrast provenance;
- the physical replacement model name must not replace `{llm}` in a contrast
  ID;
- the report must disclose the logical-slot-to-physical-model mapping.

For example, the logical slot `gemma4-26b` may prospectively be fulfilled by a
valid reserve replacement, but the registered contrast identity remains tied
to `gemma4-26b`, while physical-model provenance records the substitution.

This keeps contrast IDs frozen, preserves the structural counts, and prevents
a runtime substitution from redefining multiplicity families. It does not
alter reserve-substitution eligibility.

Effect:

```text
correctness(relevance_baseline WITH_CONTEXT)
- correctness(WITHOUT_CONTEXT)
```

The exact same retrieval-independent `WITHOUT_CONTEXT` generation is reused
across the four retriever-specific RQ1 contrasts for a
`dataset x sample x LLM`. Do not generate another no-context answer per
retriever.

Expected protected confirmatory RQ1 contrast instances:

```text
2 datasets x 4 retrievers x 3 LLMs = 24
```

## 9. RQ1 Multiplicity Family

Freeze one RQ1 correctness Holm family for each:

```text
dataset x LLM
```

Family-ID grammar:

```text
F_RQ1_CORR__{dataset}__{llm}
```

Each family contains exactly four retriever-specific RQ1 contrasts:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

Therefore protected RQ1 contains:

```text
2 datasets x 3 LLMs = 6 Holm families
4 p-values per family
```

Do not combine HotpotQA and ASQA in one Holm family. Do not combine different
LLMs in one RQ1 family.

## 10. RQ2: Diversification Effect on Retrieval

RQ2 retrieval asks:

> Does each frozen diversification treatment change or improve retrieval
> effectiveness relative to relevance-only retrieval?

Primary metrics:

```text
HotpotQA: positive-qrel document Recall@5
ASQA: SRecall@5
```

For each protected:

```text
dataset x retriever x treatment
```

define one confirmatory retrieval contrast.

Contrast-ID grammar:

```text
RQ2_RET__{dataset}__{retriever}__{treatment}
```

Reference:

```text
relevance_baseline
```

Treatment is one of `T1` through `T7`.

Effect:

```text
primary_retrieval_metric(treatment)
- primary_retrieval_metric(relevance_baseline)
```

There is no LLM factor in retrieval contrasts because retrieval artifacts are
generator-independent.

Expected protected RQ2 retrieval contrast instances:

```text
2 datasets x 4 retrievers x 7 treatments = 56
```

## 11. RQ2 Retrieval Multiplicity Family

Freeze one Holm family for each:

```text
dataset x retriever
```

Family ID:

```text
F_RQ2_RET__{dataset}__{retriever}
```

Each contains the seven baseline-versus-treatment retrieval contrasts:

- `mmr_lambda_0`;
- `mmr_lambda_0_25`;
- `mmr_lambda_0_50`;
- `mmr_lambda_0_75`;
- `kmeans_selected`;
- `agglo_selected`;
- `dpp_map`.

Protected retrieval families:

```text
2 datasets x 4 retrievers = 8 Holm families
7 p-values per family
```

Do not create a cross-retriever or cross-dataset pooled p-value family.

## 12. RQ2: Diversification Effect on Correctness

For each protected:

```text
dataset x retriever x LLM x one of seven diversification treatments
```

compare:

```text
treatment WITH_CONTEXT
minus
relevance_baseline WITH_CONTEXT
```

Primary correctness metrics:

```text
HotpotQA: official answer F1
ASQA: official ASQA Disambig-F1 (`QA-F1` in the frozen official scorer)
```

Contrast-ID grammar:

```text
RQ2_CORR__{dataset}__{retriever}__{llm}__{treatment}
```

Effect:

```text
correctness(treatment) - correctness(relevance_baseline)
```

Expected protected correctness contrast instances:

```text
2 x 4 x 3 x 7 = 168
```

Do not substitute secondary correctness metrics into these contrast IDs.

## 13. RQ2 Correctness Multiplicity Family

Freeze one family for every:

```text
dataset x retriever x LLM
```

Family ID:

```text
F_RQ2_CORR__{dataset}__{retriever}__{llm}
```

Each family contains exactly seven treatment-versus-baseline p-values.

Protected families:

```text
2 x 4 x 3 = 24
7 p-values per family
```

## 14. RQ5: Diversification Effect on Faithfulness

RQ5 asks:

> How does diversification affect context-groundedness or hallucination risk?

Use canonical `Faithfulness@5` only.

For each protected:

```text
dataset x retriever x LLM x treatment
```

compare:

```text
treatment WITH_CONTEXT
minus
relevance_baseline WITH_CONTEXT
```

Contrast-ID grammar:

```text
RQ5_FAITH__{dataset}__{retriever}__{llm}__{treatment}
```

Effect:

```text
Faithfulness@5(treatment) - Faithfulness@5(relevance_baseline)
```

Use exact measurable pairs according to
`docs/sprint3/FAITHFULNESS_PROTOCOL.md`. Do not treat undefined faithfulness
as zero.

Expected protected faithfulness contrast instances:

```text
2 x 4 x 3 x 7 = 168
```

## 15. RQ5 Faithfulness Multiplicity Family

Freeze one family for every:

```text
dataset x retriever x LLM
```

Family ID:

```text
F_RQ5_FAITH__{dataset}__{retriever}__{llm}
```

Each family contains seven treatment-versus-baseline faithfulness p-values.

Protected families:

```text
24
```

with seven p-values each.

Do not combine correctness and faithfulness p-values in one Holm family.

## 16. Total Protected Confirmatory Contrast Count

Freeze the structural count:

```text
RQ1 correctness:   24
RQ2 retrieval:     56
RQ2 correctness:  168
RQ5 faithfulness: 168
TOTAL:             416 protected confirmatory contrast instances
```

Freeze the multiplicity-family count:

```text
RQ1 correctness families:   6
RQ2 retrieval families:     8
RQ2 correctness families:  24
RQ5 faithfulness families: 24
TOTAL:                      62 Holm families
```

These are structural registry assertions. If implementation expands to a
different number, stop. Do not open protected results until the discrepancy is
explained.

## 17. Secondary Retrieval Metrics

The following are predeclared secondary or supporting analyses and are not
additional primary confirmatory contrast families.

Their analysis role is `REGISTERED_SECONDARY`.

### HotpotQA

Official or frozen secondary retrieval rank-quality diagnostics where already
authorized.

### ASQA

- `alpha-nDCG@5` at `alpha=0.5`: mandatory secondary;
- `alpha=0.3`: sensitivity only;
- `alpha=0.7`: sensitivity only;
- corpus coverability: collection diagnostic;
- `c*`: collection diagnostic.

### PubMedQA

`MRR@5` is a secondary descriptive or control retrieval metric.

For protected HotpotQA and ASQA, paired treatment-versus-baseline raw effects
and 95% paired-bootstrap CIs may be reported for these secondary retrieval
metrics.

Do not report additional confirmatory p-values for them under this registry.
Do not use them to overturn the primary retrieval endpoint.

## 18. Secondary Correctness Outputs

Freeze as secondary or descriptive:

```text
HotpotQA: official EM
ASQA: answer_side_alias_coverage
```

Their analysis role is `REGISTERED_SECONDARY`.

Optional ASQA ROUGE-L, if emitted by the frozen official scorer, remains
descriptive reference-wording overlap only and is not a correctness
confirmatory endpoint.

PubMedQA has no additional canonical correctness metric beyond categorical
decision accuracy.

For secondary correctness outputs:

- report condition means;
- report paired treatment/reference effects where useful;
- report paired 95% CIs;
- report no p-values;
- create no Holm family;
- perform no configuration selection.

## 19. ACC Registered Role

`ACC_bi` is a predeclared secondary mechanistic endpoint. It is not a quality
objective, and there is no meaningful hypothesis that higher ACC means better.
Its analysis role is `REGISTERED_SECONDARY`.

For every protected:

```text
dataset x retriever x LLM x treatment
```

where treatment is one of the seven diversified treatments, report the
canonical ACC pair already defined by `docs/sprint3/ACC_PROTOCOL.md`, including
at minimum:

- mean `ACC_bi`;
- median `ACC_bi`;
- paired-measured `N`;
- `BOTH_EMPTY` count;
- `ONE_SIDE_EMPTY` count;
- `NA_FAILED` count;
- directional retained, inherited, added, dropped, contradicted, and conflict
  summaries.

Report a question-level bootstrap 95% CI for mean `ACC_bi` where defined.

Do not test `ACC_bi = 0` as a confirmatory scientific null. Do not produce ACC
p-values. Do not put ACC in a Holm family.

ACC answers:

> Did the retrieval-context intervention propagate into substantive answer
> content?

It does not answer:

> Was the method better?

## 20. PubMedQA Control-Replication Registry

Run the same structural contrast classes on PubMedQA for comparability.

Evidence label:

```text
REGISTERED_DESCRIPTIVE_CONTROL
```

For RQ1, compare:

```text
relevance_baseline WITH_CONTEXT
minus
WITHOUT_CONTEXT
```

for all:

```text
4 retrievers x 3 LLMs
```

For diversification, compare each of the seven treatments with
`relevance_baseline` for:

- retrieval `Recall@5`;
- primary correctness;
- `Faithfulness@5`;
- ACC mechanistic summaries.

Also report:

- PubMedQA `MRR@5` as secondary;
- verdict-flip diagnostics according to `ACC_PROTOCOL.md`.

Use:

- raw paired effects;
- 95% paired-bootstrap CIs;
- exact paired Ns;
- failure reporting.

Do not:

- report PubMedQA protected-confirmatory p-values;
- include PubMedQA in Holm families;
- combine PubMedQA with HotpotQA or ASQA as if evidence roles were identical.

Historical Sprint-1 values remain separate from new canonical Sprint-3
replication outputs.

## 21. RQ3 Trade-Off Analysis

RQ3 asks:

> What relevance/diversity trade-offs correspond to changes in coverage,
> correctness, and faithfulness?

Freeze RQ3 as a predeclared synthesis of the already registered treatment
effects rather than a new protected-confirmatory hypothesis-test family.
Its analysis role is `REGISTERED_SECONDARY`; it is not a separate
`PRIMARY_CONFIRMATORY` hypothesis-test family.

For every treatment, preserve side by side:

- retrieval-diversity manipulation diagnostics;
- primary retrieval-effectiveness delta;
- `ACC_bi`;
- correctness delta;
- `Faithfulness@5` delta;
- applicable ASQA coverage diagnostics.

Use:

- paired-effect tables;
- forest-style plots;
- condition trajectories;
- clearly labelled question-level exploratory plots where useful.

Do not call ACC a mediator. Do not claim causal mediation. Do not create a new
confirmatory correlation or regression after protected outcomes are seen.

Any Spearman correlation, mixed-effects association model, thresholded
quadrant analysis, or new composite trade-off score not separately frozen
before protected opening is exploratory.

The canonical confirmatory scientific evidence for RQ3 is the joint pattern
of the already registered treatment effects, not an extra post-hoc association
test.

## 22. RQ4 Heterogeneity

RQ4 asks how effects depend on:

- dataset;
- generator LLM.

Freeze heterogeneity analysis as predeclared descriptive replication evidence.
Its analysis role is `REGISTERED_DESCRIPTIVE_HETEROGENEITY`.

For generated-answer contrasts report:

- each LLM-specific raw effect and 95% CI;
- effect sign;
- minimum and maximum effect across the three LLMs;
- number of the three LLMs with positive effect;
- number with negative effect;
- number whose CI excludes zero.

For protected-dataset comparisons, report HotpotQA and ASQA separately.

Do not:

- pool the two datasets into one confirmatory p-value;
- treat three LLM outputs as independent retrieval interventions;
- create a post-hoc best LLM;
- require a formal interaction test to answer RQ4.

Any formal interaction or mixed-effects heterogeneity model not prospectively
added before protected opening is exploratory.

## 23. RQ0 Synthesis

RQ0 is the umbrella research question:

> When and under what conditions does diversity-aware retrieval improve RAG,
> and when does it introduce noise or hallucination?

RQ0 has no additional statistical test. It is answered by synthesis of:

- RQ1 context-effect correctness contrasts;
- RQ2 retrieval contrasts;
- RQ2 correctness contrasts;
- RQ5 faithfulness contrasts;
- ACC mechanistic summaries;
- RQ3 trade-off synthesis;
- RQ4 heterogeneity summaries.

Do not create a post-hoc single overall RAG score. Do not average HotpotQA
`Recall@5` and ASQA `SRecall@5` into one raw metric.

## 24. MMR Curve and Intensity Analysis

The MMR lambda values:

```text
0
0.25
0.50
0.75
```

are fixed treatments. Their four individual baseline contrasts are already
registered in RQ2 and RQ5.

Additionally freeze a secondary descriptive MMR intensity presentation:

- plot condition means or effects in lambda order;
- show paired 95% CIs;
- preserve dataset, retriever, and LLM stratification where applicable.

Do not:

- select a best lambda;
- fit a result-dependent polynomial;
- add `lambda=1` to protected-final analysis;
- declare a dose-response p-value under this registry.

A formal trend model not separately frozen before protected opening is
exploratory.

## 25. KMeans and Agglomerative Selected Conditions

The registry refers only to logical IDs:

```text
kmeans_selected
agglo_selected
```

The physical selected `k` values come from the independent bounded `SELECTION`
procedure.

Once substituted:

- do not change contrast IDs;
- do not add rejected `k` values to protected confirmatory analyses;
- preserve selected-`k` provenance in every affected contrast artifact.

The selected clustering values were tuned on `SELECTION` evidence but are
tested confirmatorily only on protected HotpotQA and ASQA evidence.

## 26. DPP Condition

The canonical protected DPP treatment is:

```text
dpp_map
```

only.

Do not register stochastic seed comparisons as protected confirmatory tests.
Stochastic DPP seeds 1, 2, and 3 remain `DEVELOPMENT` sensitivity evidence.

## 27. Failure and Missingness Reporting

Every registered analysis must report, as applicable:

- expected `N`;
- reference measured `N`;
- treatment measured `N`;
- exact paired `N`;
- `REFUSAL` count and rate;
- `PARSE_FAILURE` count and rate;
- `TRUNCATED` count and rate;
- `ERROR` count and rate;
- structural retrieval failures;
- evaluator failures;
- decomposer failures;
- verifier failures;
- metric-specific NA reasons.

Complete-case comparison does not erase failures. Do not convert failure to
zero except where a frozen metric protocol explicitly defines a semantic zero,
such as correctness for `REFUSAL`.

## 28. Technical Failure Outcomes in Protected Final

Technical failure rates are predeclared safety or robustness diagnostics.
Their analysis role is `REGISTERED_SECONDARY`.

For each treatment versus baseline, report:

- failure-rate difference;
- paired 95% bootstrap CI;

on the full expected manifest.

Do not use the `SELECTION` technical-failure margin as a protected-final
superiority threshold. Do not report protected technical-failure p-values
under this registry. Do not hide a method's low metric `N` when its failures
are elevated.

## 29. No Direct Diversified-Versus-Diversified Confirmatory Tests

The confirmatory reference for diversification is always:

```text
relevance_baseline
```

Therefore this registry does not make the following protected-confirmatory:

- MMR versus DPP;
- KMeans versus Agglomerative;
- one MMR lambda versus another;
- DPP versus selected clustering;
- one retriever versus another;
- one LLM versus another.

Those may be shown descriptively when scientifically useful. A new inferential
comparison among them after protected outcomes are observed is exploratory.

## 30. No Retriever Winner Test

Retriever is a scientific factor, not a selected competitor.

Do not:

- rank retrievers to choose one;
- register cross-retriever superiority as a primary hypothesis;
- average retriever effects into a hidden winner score.

Report retriever-stratified effects. Cross-retriever observations support
RQ4-style heterogeneity interpretation.

## 31. Confirmatory Label Rule

A protected-final analysis may be labelled `PRIMARY_CONFIRMATORY` only if:

- its contrast ID is generated by one of Sections 8, 10, 12, or 14;
- its dataset evidence role is `PROJECT_PROTECTED_FINAL`;
- its metric version is the frozen primary metric named in that contrast
  class;
- reference and treatment identity match this registry;
- exact pairing and CI procedure match this registry;
- the applicable Holm family was predefined here;
- no protected outcome influenced the analysis definition.

An individual `PRIMARY_CONFIRMATORY` contrast may receive a formal
multiplicity-controlled confirmatory significance statement only when its
Holm-adjusted `p < 0.05` within its predeclared family. An ordinary 95% CI
excluding zero or a raw `p < 0.05` is not sufficient for that formal
familywise statement.

Everything else receives the applicable exact label from this closed set:

- `REGISTERED_SECONDARY`;
- `REGISTERED_DESCRIPTIVE_CONTROL`;
- `REGISTERED_DESCRIPTIVE_HETEROGENEITY`;
- `EXPLORATORY`.

## 32. Exploratory Boundary

After protected-final outcomes are opened, the following are exploratory
unless already explicitly frozen elsewhere:

- new subgroup cuts;
- new demographic, topic, or entity subsets;
- new metric transformations;
- new thresholds;
- new correlation analyses;
- new regression models;
- new composite scores;
- new pairwise treatment comparisons;
- new retriever winner comparisons;
- new LLM winner comparisons;
- new alpha values;
- new MMR trend functions;
- new error categories created because of observed performance.

Exploratory findings may be useful and reported. They must be clearly labelled
and cannot be retroactively promoted to confirmatory evidence.

## 33. Secondary Metric Labels

Freeze these registry labels:

```text
PRIMARY_CONFIRMATORY
SECONDARY_CONFIRMATORY_SUPPORT
REGISTERED_SECONDARY
REGISTERED_DESCRIPTIVE_CONTROL
REGISTERED_DESCRIPTIVE_HETEROGENEITY
EXPLORATORY
```

Use `PRIMARY_CONFIRMATORY` for the metric, effect, and CI associated with
Sections 8, 10, 12, and 14.

Use `SECONDARY_CONFIRMATORY_SUPPORT` only for the corresponding Holm-adjusted
paired p-value.

Use `REGISTERED_SECONDARY` for predefined non-primary outputs such as:

- HotpotQA EM;
- ASQA `alpha-nDCG@5`;
- ASQA answer-side alias coverage;
- `ACC_bi`;
- failure-rate deltas;
- MMR intensity plots;
- RQ3 synthesis.

Use `REGISTERED_DESCRIPTIVE_CONTROL` for PubMedQA canonical Sprint-3
replication analyses.

Use `REGISTERED_DESCRIPTIVE_HETEROGENEITY` for RQ4 heterogeneity summaries.

## 34. Protected Result Interpretation

Do not make a universal statement that diversification works from one positive
cell.

Report treatment effects stratified by:

- dataset;
- retriever;
- LLM, where applicable.

A treatment can:

- improve retrieval but hurt correctness;
- change answer content without improving quality;
- improve correctness while decreasing faithfulness;
- have different effects across LLMs.

Those are scientifically meaningful outcomes, not inconsistencies to hide.

## 35. Machine-Readable Expansion

Before protected-final analysis, materialize a machine-readable contrast
registry from this document.

Every expanded contrast must contain at least:

- registry protocol version and hash;
- `contrast_id`;
- RQ;
- analysis role;
- evidence role;
- dataset;
- retriever or `NOT_APPLICABLE`;
- LLM or `NOT_APPLICABLE`;
- reference condition;
- treatment condition;
- metric ID and version;
- direction convention;
- expected manifest and hash;
- pairwise-complete rule;
- bootstrap method;
- bootstrap resamples;
- bootstrap seed;
- CI level and type;
- p-value status;
- permutation seed if applicable;
- multiplicity-family ID if applicable;
- Holm status;
- Git commit;
- run-registry identity.

Do not materialize the registry in this task.

## 36. Structural Assertions

Before protected-final analysis, assert:

```text
protected confirmatory contrast instances = 416
Holm families = 62

RQ1_CORR = 24
RQ2_RET = 56
RQ2_CORR = 168
RQ5_FAITH = 168

F_RQ1_CORR = 6
F_RQ2_RET = 8
F_RQ2_CORR = 24
F_RQ5_FAITH = 24
```

If expansion differs, stop. Do not inspect protected metrics while debugging
registry-definition discrepancies.

## 37. Relation to Other Protocols

This registry must be read with:

- `docs/sprint3/FINAL_EXPERIMENT_MATRIX_PROTOCOL.md`;
- `docs/sprint3/SELECTION_STATISTICS_PROTOCOL.md`;
- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`;
- `docs/sprint3/GENERATION_PROTOCOL.md`;
- `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`;
- `docs/sprint3/FAITHFULNESS_PROTOCOL.md`;
- `docs/sprint3/ACC_PROTOCOL.md`;
- `docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md`;
- `docs/sprint3/HOTPOTQA_FINAL_EVALUATION_PROTOCOL.md`.

The matrix protocol defines which cells exist. The selection protocol defines
how selected clustering configurations were chosen. This registry defines
which protected comparisons are confirmatory. The correctness, faithfulness,
and ACC protocols define measurement semantics. The stage-gate protocol
controls when protected outcomes may be opened.

## 38. Frozen Before Protected Observation

This registry is frozen before canonical protected-final outcomes are
inspected.

No protected HotpotQA or ASQA result was used to choose:

- contrast classes;
- treatment and reference pairs;
- metrics;
- evidence roles;
- LLM and retriever strata;
- p-value role;
- Holm families;
- ACC role;
- PubMedQA descriptive role;
- RQ3 synthesis rule;
- RQ4 heterogeneity rule;
- exploratory boundary.

## 39. After Creation

After creating this file:

1. show the complete new-file diff;
2. run `git status --short`;
3. confirm that no other file changed;
4. do not commit;
5. stop.

Do not run implementation or experiment work.
