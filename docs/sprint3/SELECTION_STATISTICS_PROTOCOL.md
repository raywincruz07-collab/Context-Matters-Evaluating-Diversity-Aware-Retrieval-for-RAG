# Sprint 3 Method-Selection and Statistical Protocol

## 1. Purpose

This protocol defines the prospective bounded selection procedure used to
choose:

- one global KMeans `k` from `{2,3,5}`; and
- one global Agglomerative `k` from `{3,5}`.

The selected `k` for each family must be shared across:

- HotpotQA;
- ASQA;
- BM25;
- DPR;
- Contriever;
- ColBERTv2;
- all three generator LLMs.

Do not tune `k` separately by:

- dataset;
- retriever;
- LLM.

PubMedQA does not numerically select clustering `k`.

Only KMeans and Agglomerative have performance-selected hyperparameters.

Do not select:

- MMR lambda;
- `dpp_map`;
- stochastic-DPP seed;
- retriever;
- generator LLM;

using `SELECTION` outcomes.

## 2. Evidence Roles

K-selection uses only non-protected `SELECTION` evidence.

### HotpotQA

`SELECTION` is the complete BEIR HotpotQA dev split:

```text
N = 5,447 queries
```

Do not subsample those 5,447 queries for canonical clustering selection. The
physical manifest is materialized later and must contain every canonical
BEIR-dev query ID exactly once, plus source revision and hash.

### ASQA

`SELECTION` is the frozen internal 871-question selection partition governed
by `docs/sprint3/ASQA_INTERNAL_PARTITION_PROTOCOL.md`.

ASQA dev948 remains `PROJECT_PROTECTED_FINAL`.

### PubMedQA

PubMedQA is `HISTORICAL_OBSERVED` control evidence only. It contributes:

- descriptive replication;
- robustness and context interpretation;

but no numeric vote in global clustering-k selection.

## 3. Two-Stage Selection Design

Selection has exactly two stages:

```text
STAGE A: retrieval admissibility / non-inferiority screening
STAGE B: downstream non-harm guardrail
```

Only Stage-A survivors receive Stage-B downstream generation.

Do not decide after seeing Stage-A results to generate all candidates. Do not
introduce new `k` values after Stage A opens. Do not weaken a gate because no
candidate passes.

If a family has no eligible candidate, stop that family's selection. Do not
choose the least bad candidate. Do not open protected-final evaluation for an
unresolved family. A prospective amendment with exposure accounting would be
required.

## 4. Stage-A Candidates

Evaluate every registered clustering candidate:

```text
KMeans:
k = 2
k = 3
k = 5

Agglomerative:
k = 3
k = 5
```

against the same-retriever relevance-only baseline.

Use all four retrievers.

Every candidate receives the exact same first-stage top-20 candidate artifact
for a fixed:

```text
dataset x sample x retriever
```

The following remain governed by
`docs/sprint3/CANDIDATE_POOL_TOPK_PROTOCOL.md`:

```text
candidate_pool = 20
final_top_k = 5
```

Do not reretrieve separately for clustering candidates.

## 5. HotpotQA Lead Selection Metric

Freeze the Stage-A HotpotQA lead metric as positive-qrel document `Recall@5`.

For question `q`:

```text
Rel_q =
the set of positive-qrel document IDs in the canonical BEIR qrels

Retrieved_q^5 =
the five selected document IDs
```

Define:

```text
Recall@5(q) =
|Retrieved_q^5 intersection Rel_q|
/
|Rel_q|
```

when `|Rel_q| > 0`.

The Stage-A stratum statistic is the question-level mean `Recall@5`.

Safe terminology is:

- positive-qrel document `Recall@5`;
- relevant-document coverage.

Do not claim this metric directly measures:

- both reasoning hops;
- complete multi-hop reasoning;
- supporting-fact sentence retrieval.

Secondary HotpotQA retrieval metrics may be reported where already defined,
including rank-quality diagnostics, but they do not select `k`.

## 6. ASQA Lead Selection Metric

Freeze the Stage-A ASQA lead metric as `SRecall@5`, defined exactly in
`docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md`.

`alpha-nDCG@5` with `alpha=0.5` is mandatory secondary reporting.

`alpha=0.3` and `alpha=0.7` are sensitivity analyses only. `c*` and corpus
coverability are diagnostics.

Do not use:

- `alpha=0.3` or `alpha=0.7`;
- `c*`;
- embedding diversity;

to select `k`.

## 7. Eight Retrieval Strata

For each clustering family, global selection is based on exactly eight
retrieval strata:

```text
HotpotQA x BM25
HotpotQA x DPR
HotpotQA x Contriever
HotpotQA x ColBERTv2

ASQA x BM25
ASQA x DPR
ASQA x Contriever
ASQA x ColBERTv2
```

HotpotQA uses `Recall@5`. ASQA uses `SRecall@5`.

Do not average the raw HotpotQA and ASQA metric values together.

## 8. Fundamental Analysis Unit

The fundamental statistical unit is the question or sample ID.

All paired comparisons require the exact same source sample ID.

For candidate `c` and baseline `b`, define:

```text
d_q = metric_c(q) - metric_b(q)
```

only where that metric is validly measured in both conditions.

Never:

- pair different questions;
- replace missing values with zero;
- impute failed outcomes;
- substitute another question.

## 9. Pairwise Complete-Case Rule

For a candidate-versus-baseline effect estimate or non-inferiority gate, use
the exact pairwise complete-case intersection for that contrast.

Report:

- expected `N`;
- candidate measured `N`;
- baseline measured `N`;
- pairwise complete `N`;
- excluded or missing `N`;
- failure reasons.

The non-inferiority estimate is based on the pairwise complete intersection.

Failure rates themselves are always analyzed against the full expected
manifest, not the successful metric intersection.

## 10. Family-Common Complete-Case Rule

For comparing or ranking multiple `k` values within one clustering family and
one dataset-by-retriever stratum, freeze one family-common intersection.

The KMeans common set contains questions where the lead metric is validly
measured for:

```text
baseline
k=2
k=3
k=5
```

The Agglomerative common set contains questions where the lead metric is
validly measured for:

```text
baseline
k=3
k=5
```

This common set is frozen from the original registered family. Do not
recompute a larger common set merely because a candidate later fails Stage A
or Stage B.

Use this fixed family-common set for:

- candidate mean lead-metric values used in rank aggregation;
- candidate-to-candidate practical-equivalence checks.

Also report all ordinary pairwise `N` values. This prevents candidate-specific
missingness from changing the ranking sample.

## 11. Primary Effect Estimate

For every registered paired comparison, report:

- candidate mean on the paired intersection;
- baseline or reference mean on the same paired intersection;
- raw paired mean difference, `mean(candidate - reference)`;
- confidence interval;
- expected `N`;
- measured `N`;
- paired `N`;
- failure and missingness counts.

Raw metric-scale differences are primary.

Do not replace them with:

- standardized effects;
- relative percentage changes;
- pooled cross-dataset scores.

Standardized effects may later be exploratory or sensitivity analyses only.

## 12. Bootstrap Procedure

Freeze the canonical confidence-interval procedure:

```text
resampling unit:         question/sample ID
bootstrap resamples:     10,000
base seed:               20260823
interval:                percentile
primary confidence level: 95%
```

Procedure for one paired contrast:

1. Construct the exact paired question-level difference vector.
2. Sample `N` question indices with replacement.
3. Retain candidate and reference values as a pair.
4. Calculate the resampled mean paired difference.
5. Repeat 10,000 times.
6. Take the 2.5th and 97.5th empirical percentiles.

Do not:

- bootstrap candidate and baseline independently;
- bootstrap LLM answers as if they were independent questions;
- bootstrap retrieved documents inside a question;
- silently switch to BCa after viewing results.

If the statistic cannot be validly computed, mark the comparison
`INDETERMINATE`. Do not manufacture a confidence interval.

## 13. Non-Inferiority Architecture

For a higher-is-better metric, define:

```text
delta = candidate - baseline
```

A candidate passes a non-inferiority gate only if:

```text
lower endpoint of paired 95% CI > -DELTA
```

where `DELTA` is the corresponding prospectively frozen absolute margin.

Equality to `-DELTA` does not pass.

Margins are absolute metric-scale differences.

Do not use:

- relative-percent loss;
- standardized-effect margins;
- margins fitted from `SELECTION` outcomes.

## 14. Pending Supervisor Margins

Every value and rule in the following table has status
`PENDING_SUPERVISOR_ADJUDICATION`.

| Parameter | Status | Proposal sent for supervisor review |
|---|---|---|
| `DELTA_HOTPOT_RECALL` | `PENDING_SUPERVISOR_ADJUDICATION` | `0.02` |
| `DELTA_ASQA_SRECALL` | `PENDING_SUPERVISOR_ADJUDICATION` | `0.02` |
| `DELTA_CORRECTNESS` | `PENDING_SUPERVISOR_ADJUDICATION` | `0.02` |
| `DELTA_FAITHFULNESS` | `PENDING_SUPERVISOR_ADJUDICATION` | `0.02` |
| `EPSILON_PRACTICAL_EQUIVALENCE` | `PENDING_SUPERVISOR_ADJUDICATION` | `0.01` |
| `STRUCTURAL_RETRIEVAL_FAILURE_RULE` | `PENDING_SUPERVISOR_ADJUDICATION` | No additional candidate-attributable structural failures; zero tolerance |
| `DELTA_DOWNSTREAM_TECH_FAILURE` | `PENDING_SUPERVISOR_ADJUDICATION` | `0.01` |

Every proposed value or rule above is:

```text
PROPOSAL_SENT_FOR_SUPERVISOR_REVIEW
NOT ACTIVE
NOT FROZEN
```

These values are project or substantive conventions, not universal IR or RAG
standards.

`SELECTION` may not open until these values and rules are resolved in tracked
authority. Do not silently activate the proposed numbers merely by writing
them here.

## 15. Stage-A Admission Rule

A clustering candidate survives Stage A only if all of the following hold:

1. Implementation and provenance validity passes.
2. Required top-5 output is structurally valid under the frozen candidate and
   diversification contracts.
3. The finally adjudicated structural-retrieval-failure rule passes.
4. HotpotQA `Recall@5` non-inferiority passes in each of the four HotpotQA
   retriever strata.
5. ASQA `SRecall@5` non-inferiority passes in each of the four ASQA retriever
   strata.

Therefore, a candidate must pass the lead-metric admissibility gate in all
eight retrieval strata.

Do not:

- allow excellent performance in one retriever to compensate for unacceptable
  loss in another;
- average lead metrics before the gate;
- let ASQA gains compensate for HotpotQA loss;
- let HotpotQA gains compensate for ASQA loss.

Secondary retrieval metrics and diversity metrics are reported but do not
rescue or reject an otherwise valid candidate.

## 16. Diversity Manipulation Check

For Stage-A candidates, report the frozen retrieval-diversity diagnostics.

Their role is:

- manipulation check;
- mechanism description;
- trade-off interpretation.

Do not maximize diversity. Do not reject a candidate merely because another
candidate is more diverse, provided the candidate is otherwise valid. Do not
let high diversity compensate for failed relevance or coverage gates.

## 17. Stage-B Generation Policy

Generate downstream `SELECTION` answers only for:

- the relevance-only `WITH_CONTEXT` baseline;
- Stage-A-surviving KMeans candidates;
- Stage-A-surviving Agglomerative candidates.

Use all:

- two selection datasets: HotpotQA and ASQA;
- four retrievers;
- three primary generator LLMs.

Do not generate MMR or `dpp_map` merely for clustering-k selection. They
remain predeclared scientific conditions for the final experiment.

Do not multiply `WITHOUT_CONTEXT` by candidate. If canonical
`WITHOUT_CONTEXT` selection rows exist for another valid purpose, reuse their
retrieval-independent identity. They play no role in choosing clustering `k`.

## 18. Three-LLM Stage-B Rule

Use all three frozen primary LLMs:

```text
llama-3.3-70b
gemma4-26b
ministral-3-14b
```

Do not choose one reference LLM. Do not select `k` using a pooled raw LLM
score.

For every Stage-B candidate, assess downstream guardrails separately by:

```text
dataset x retriever x LLM
```

A candidate must pass the required downstream guardrails in every applicable
LLM stratum. No good result from one LLM may compensate for unacceptable harm
in another.

## 19. Stage-B Quality Metrics

Downstream quality guardrails are:

1. canonical dataset-specific answer correctness from
   `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`;
2. canonical `Faithfulness@5` from
   `docs/sprint3/FAITHFULNESS_PROTOCOL.md`.

`ACC_bi` is:

- descriptive and mechanistic;
- not a quality objective;
- not a Stage-B pass or fail metric.

ASQA answer-side alias coverage is diagnostic unless the correctness protocol
explicitly gives it another role.

Do not redefine correctness here.

## 20. Stage-B Correctness Gate

For each:

```text
candidate x dataset x retriever x LLM
```

compare the candidate with the same-retriever relevance-only baseline using
the dataset's frozen primary correctness metric.

Use exact question-level pairing.

A candidate passes that correctness stratum only if:

```text
lower endpoint of paired 95% CI(candidate - baseline)
>
-DELTA_CORRECTNESS
```

using the finally supervisor-adjudicated margin.

The candidate must pass all applicable correctness strata.

## 21. Stage-B Faithfulness Gate

For each:

```text
candidate x dataset x retriever x LLM
```

compare the candidate with the relevance-only baseline on defined
`Faithfulness@5` pairs.

Use exact pairwise complete cases according to
`docs/sprint3/FAITHFULNESS_PROTOCOL.md`.

A candidate passes that faithfulness stratum only if:

```text
lower endpoint of paired 95% CI(candidate - baseline)
>
-DELTA_FAITHFULNESS
```

The candidate must pass all applicable faithfulness strata.

If a required stratum has insufficient valid data to compute the frozen
comparison, mark it `INDETERMINATE`. An `INDETERMINATE` required guardrail is
not a pass. Do not treat missing faithfulness as zero.

## 22. Failure and Missingness Definitions

Always report failures on the full expected manifest.

Keep at least these categories separate:

### Retrieval and Structural

- `SHORT_CANDIDATE_LIST`;
- diversification unable to produce five valid unique passages;
- invalid or non-finite selection result;
- retrieval or candidate-artifact failure.

### Generation

- `ERROR`;
- `PARSE_FAILURE`;
- `TRUNCATED`;
- `REFUSAL`, reported separately as a semantic model outcome.

### Evaluation

- decomposer failure;
- verifier failure;
- overlength or evaluator-capacity failure;
- metric or scorer failure.

Do not collapse refusal into infrastructure failure. Do not hide any failure
through complete-case analysis.

## 23. Structural Retrieval Failure Guardrail

The exact allowable structural-failure rule remains:

```text
PENDING_SUPERVISOR_ADJUDICATION
```

The current proposed rule is:

> No additional candidate-attributable structural failures relative to the
> same-retriever baseline on the full expected `SELECTION` manifest.

Do not activate that proposal until it is formally adjudicated. Once
adjudicated, apply the same rule to every clustering candidate. Do not choose
the rule after observing candidate failures.

## 24. Downstream Technical Failure Guardrail

Define a downstream technical-failure indicator prospectively from failures
that prevent required correctness or faithfulness measurement because of:

- generation `ERROR`;
- `PARSE_FAILURE`;
- `TRUNCATED`;
- decomposer failure;
- verifier or evaluator failure.

`REFUSAL` remains separately reported as semantic behavior and is not silently
relabeled technical failure.

For candidate versus baseline, compute on the full expected Stage-B manifest:

```text
failure_difference =
candidate technical failure indicator
-
baseline technical failure indicator
```

at the question level.

Use the same paired 10,000-resample bootstrap.

The candidate passes the technical-failure guardrail only if:

```text
upper endpoint of the 95% CI
<
DELTA_DOWNSTREAM_TECH_FAILURE
```

where the margin remains pending supervisor adjudication. Equality does not
pass.

Also report every component failure category separately.

## 25. Stage-B Survival Rule

A Stage-A survivor remains globally eligible only if:

- every required correctness stratum passes;
- every required faithfulness stratum passes;
- the downstream technical-failure guardrail passes;
- no required guardrail is `INDETERMINATE`.

If no candidate in a family survives, stop.

Do not weaken margins. Do not average harms away. Do not choose based on ACC.
Do not choose based on raw answer quality.

## 26. Global K Rank Aggregation

Rank aggregation occurs only after Stage-A and Stage-B eligibility has been
fully determined.

Apply this procedure separately for:

- KMeans;
- Agglomerative.

For each clustering family:

1. Retain only candidates that passed:

   - Stage A;
   - Stage B;
   - every required non-inferiority and failure guardrail.

2. Call these candidates the **eligible survivors**.
3. Compute each eligible survivor's lead-metric mean using the already-frozen
   original family-common complete-case set from Section 10.
4. Within each of the eight retrieval strata, rank only the eligible
   survivors:

   - higher lead metric equals better;
   - rank 1 is best;
   - exactly equal metric means receive average tied ranks.

5. Compute each eligible survivor's unweighted arithmetic mean rank across the
   eight strata.
6. Apply the existing practical-equivalence and final tie and winner rules.

The baseline is not itself ranked as a `k` candidate.

For each eligible survivor:

```text
mean_rank =
unweighted arithmetic mean of its eight stratum ranks
```

Each of the eight strata has equal weight.

Do not weight by:

- dataset size;
- number of paired observations;
- metric variance;
- raw metric scale.

Select the eligible survivor with the lowest `mean_rank`, subject to the
practical-equivalence and tie rule below.

Always report the underlying raw metric values and candidate-baseline effects
alongside ranks.

A candidate that failed Stage A or Stage B:

- remains preserved and reported as valid `SELECTION` evidence;
- retains its raw metrics, effects, failures, and rejection reason;
- does not participate in final global rank aggregation;
- cannot alter the ranks of eligible survivors.

The family-common complete-case set remains constructed from the original
registered family exactly as frozen in Section 10 and is not recomputed after
candidates are rejected. Candidate eligibility affects which candidates are
ranked; candidate eligibility does not affect which questions define their
ranking metric means. This preserves missingness comparability while
preventing rejected candidates from influencing the final winner.

## 27. Practical-Equivalence Architecture

The practical-equivalence margin:

```text
EPSILON_PRACTICAL_EQUIVALENCE
```

remains `PENDING_SUPERVISOR_ADJUDICATION`.

Once frozen, use equivalence logic at `alpha = 0.05`, implemented as a paired
bootstrap 90% percentile confidence interval for:

```text
candidate1 - candidate2
```

Use the same question-level paired procedure and fixed family-common
complete-case set.

Two eligible `k` candidates are declared practically equivalent for global
selection only if, in all eight lead-metric strata, the entire paired 90%
confidence interval lies strictly inside:

```text
(-EPSILON, +EPSILON)
```

Do not call non-significance equivalence. Do not infer equivalence from
overlapping 95% confidence intervals. Do not activate practical-equivalence
logic until `EPSILON` is formally frozen.

## 28. Final Tie and Winner Rule

For eligible survivors only:

1. Identify the eligible survivor with the lowest `mean_rank`.
2. If another eligible survivor is practically equivalent to the best-ranked
   eligible survivor under Section 27 across all eight strata, place it in the
   eligible-survivor practical-equivalence set.
3. If the practical-equivalence set contains more than one candidate, choose
   the smallest `k` in that set.
4. If practical equivalence is not established but eligible survivors have
   exactly equal `mean_rank`, break the exact mean-rank tie among those
   eligible survivors in this order:

   1. better worst-stratum rank;
   2. greater number of the eight strata with candidate-baseline raw mean
      difference at least zero;
   3. lower aggregate candidate-attributable failure rate over the relevant
      full `SELECTION` manifests;
   4. smaller `k`.

Do not invent another tie-break after outcomes are observed.

Smaller `k` is a deterministic conservative tie-break only. Do not claim
smaller `k` is inherently scientifically superior.

## 29. What Happens to Rejected K Values

Rejected or non-selected `k` values remain valid `SELECTION` evidence.

Preserve and report:

- Stage-A metrics;
- gate results;
- Stage-B results, if generated;
- reason for rejection;
- paired `N` values;
- failures.

Do not delete them. Do not rerun them with altered rules. They do not appear
as canonical protected-final `k` configurations.

## 30. P-Values

P-values are secondary.

They are not used for:

- Stage-A admission;
- Stage-B admission;
- global `k` selection;
- practical equivalence.

Primary decision evidence is:

- raw paired effects;
- confidence intervals;
- predeclared margins.

If a later registered confirmatory contrast requests a p-value, use a
question-level paired sign-flip or permutation test of the paired difference.

For a difference vector `d`, implement null symmetry by independently
multiplying paired differences by `+1` or `-1`.

Use:

- exact enumeration when the complete sign-pattern space contains at most
  10,000 patterns;
- otherwise, 10,000 Monte Carlo sign-flip permutations.

For Monte Carlo reproducibility, derive the per-contrast random-number seed
from:

```text
SHA256(
  "sprint3_perm_v1|" + contrast_id
)
```

using a deterministic documented integer conversion.

Use two-sided p-values unless the later contrast registry explicitly defines
a one-sided confirmatory hypothesis prospectively.

No p-value may override a failed non-inferiority guardrail.

## 31. Multiplicity

Confidence intervals and effect sizes remain the primary reporting language.

When p-values are reported, apply Holm adjustment within each predeclared
logical contrast family.

Do not create one giant correction family spanning unrelated:

- datasets;
- metrics;
- research questions.

Do not choose multiplicity families after seeing significance.

The exact final family memberships are frozen later in
`docs/sprint3/CONFIRMATORY_CONTRAST_REGISTRY.md`.

Selection gates themselves do not depend on Holm-adjusted p-values.

## 32. Retriever and LLM Heterogeneity

Primary reporting remains stratified by:

```text
dataset x retriever
```

for retrieval outcomes, and:

```text
dataset x retriever x LLM
```

for generated-answer outcomes.

Also report:

- effect direction and sign consistency;
- forest-style raw paired effects and confidence intervals;
- equal-weight macro summaries where useful.

Do not treat three LLM outputs as three independent retrieval interventions.
Do not average raw HotpotQA `Recall@5` and ASQA `SRecall@5` into one score.

Mixed-effects modeling is not required for clustering-k selection. Any later
repeated-measures association analysis is secondary and must remain clearly
separated from the winner rule.

## 33. MMR Role

MMR protected and canonical levels are already frozen. They are treatments,
not a selection sweep.

Do not choose the best lambda from `SELECTION`.

Canonical primary comparisons later include baseline versus each frozen MMR
level according to the contrast registry.

MMR `lambda=1` remains a `DEVELOPMENT` equivalence or defect gate, not a final
treatment.

Any MMR trend analysis is secondary and may not change the registered
treatment set.

## 34. DPP Role

`dpp_map` is the canonical deterministic DPP condition.

Exact stochastic fixed-k DPP seeds 1, 2, and 3 are `DEVELOPMENT` sensitivity
only.

Do not select the best DPP seed. Do not use stochastic DPP selection outcomes
to alter `dpp_map`.

## 35. Confirmatory Versus Exploratory

Clustering-k selection is bounded and prospective.

After winners are frozen:

- protected-final conditions are fixed;
- confirmatory contrasts must come from the forthcoming registered contrast
  file;
- unregistered analyses are exploratory.

Do not relabel an exploratory finding as confirmatory because it is
interesting.

## 36. No Selection Result May Change the Protocol

After `SELECTION` opens, do not change based on observed outcomes:

- candidate `k` set;
- Stage-A lead metric;
- margins;
- failure rules;
- Stage-B metrics;
- LLM set;
- bootstrap method;
- resample count;
- seed;
- confidence-interval type;
- common-complete-case rule;
- aggregation rule;
- practical-equivalence rule;
- tie-break;
- multiplicity policy.

A necessary scientifically meaningful change requires:

- stop;
- prospective amendment;
- explicit `SELECTION` exposure accounting;
- no pretense that prior exposure did not occur.

## 37. Provenance

Every selection-statistics artifact must bind or reference:

- protocol version and hash;
- dataset;
- evidence role;
- selection manifest and hash;
- corpus manifest and hash;
- retriever;
- family;
- candidate `k`;
- baseline identity;
- candidate-artifact hashes;
- metric version;
- expected `N`;
- measured `N`;
- pairwise complete `N`;
- family-common complete `N`;
- raw paired differences and hash;
- candidate and reference means;
- paired mean difference;
- bootstrap algorithm;
- resample count;
- seed;
- confidence-interval level and type;
- confidence-interval bounds;
- active margin version;
- Stage-A pass or fail;
- Stage-B generation identity, where applicable;
- LLM;
- correctness version;
- faithfulness version;
- technical-failure counts;
- Stage-B pass or fail;
- eight stratum ranks;
- mean rank;
- practical-equivalence result, if active;
- final tie-break fields;
- winner identity;
- Git commit;
- run-registry identity.

## 38. Relation to Other Protocols

This protocol must be read with:

- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`;
- `docs/sprint3/CANDIDATE_POOL_TOPK_PROTOCOL.md`;
- `docs/sprint3/DIVERSIFICATION_CONFIGURATION_PROTOCOL.md`;
- `docs/sprint3/ASQA_INTERNAL_PARTITION_PROTOCOL.md`;
- `docs/sprint3/ASQA_RETRIEVAL_METRIC_PROTOCOL.md`;
- `docs/sprint3/HOTPOTQA_FINAL_EVALUATION_PROTOCOL.md`;
- `docs/sprint3/GENERATION_PROTOCOL.md`;
- `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`;
- `docs/sprint3/FAITHFULNESS_PROTOCOL.md`;
- `docs/sprint3/ACC_PROTOCOL.md`.

This protocol governs:

- clustering-k selection;
- paired statistics;
- Stage-A and Stage-B gates;
- global rank aggregation;
- missingness and failure handling for selection.

It does not govern:

- protected experiment-condition enumeration;
- final confirmatory contrast IDs and families in full detail;
- corpus-resource fallback admissibility.

Those receive separate authority.

## 39. Remaining External Blocker

All nonnumeric selection mechanics are frozen by this document.

Formal selection remains blocked until the professor or supervisor-dependent
margin and rule table in Section 14 is resolved prospectively in tracked
repository authority.

No `SELECTION` outcome may be opened to choose those values.

## 40. Frozen Before Selection

This protocol is created before HotpotQA or ASQA `SELECTION` outcomes are
inspected under the canonical Sprint-3 experiment.

No `SELECTION` or protected-final outcome was used to choose:

- the two-stage design;
- Stage-B survivor policy;
- all-three-LLM rule;
- complete-case policy;
- bootstrap;
- rank aggregation;
- practical-equivalence architecture;
- failure handling;
- tie-break hierarchy.
