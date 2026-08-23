# NLI Verifier Calibration Protocol Amendment 01

## 1. Authority and Scope

This is a prospective, pre-observation amendment to:

- `docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md`.

It must also be read with:

- `docs/sprint3/ACC_PROTOCOL.md`;
- `docs/sprint3/FAITHFULNESS_PROTOCOL.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_02.md`;
- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`.

Amendment 01 prospectively supersedes only ambiguous wording in
`docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md` concerning:

- exact calibration-call sampling;
- dataset and LLM allocation;
- method and retriever representation;
- class-quota contingency ordering;
- annotation-guide low-kappa handling;
- construction of the 30 faithfulness pair cases;
- construction of the 30 ACC window cases;
- treatment of aggregated ACC `INCONSISTENT` in window validation.

The base protocol remains authoritative for every rule not explicitly
superseded here. Preserve without modification:

- the verifier family;
- four separately fitted thresholds;
- evidence roles;
- the initial 150 fitting plus 60 held-out design per task;
- precision target `= 0.90`;
- fitting predicted-positive `N >= 20`;
- held-out predicted-positive `N >= 15`;
- fitting class minima of at least 30 `ENTAILMENT` and at least 30
  `CONTRADICTION` units;
- held-out class minima of at least 15 `ENTAILMENT` and at least 15
  `CONTRADICTION` units;
- exact observed-probability threshold search;
- the held-out pass rule;
- Wilson reporting;
- exactly one permitted 210-fitting plus fresh-60-held-out expansion;
- faithfulness pair validation `N=30`;
- ACC window validation `N=30`;
- targeted integration pass of at least 27/30;
- the repeatability gate;
- failure semantics;
- prohibition on `SELECTION` or protected-final calibration.

## 2. Canonical Calibration Call Registry

Before human annotation or verifier scoring, construct a complete eligible
`DEVELOPMENT`-only call registry separately for:

- ACC; and
- faithfulness.

Each call must have a stable canonical call ID derived from the scientific
identity of the call. At minimum, bind:

- task;
- dataset;
- source sample ID;
- evidence role;
- LLM logical slot;
- retriever, where applicable;
- retrieval or diversification treatment, where applicable;
- ACC comparison direction, where applicable;
- claim identity and hash;
- premise type;
- exact premise identity and hash;
- generation identity;
- decomposer identity;
- exposure purpose.

Do not include:

- verifier probabilities;
- verifier predictions;
- correctness;
- retrieval effectiveness;
- faithfulness result;
- ACC result;
- `SELECTION` result;
- protected-final result.

Freeze and hash the eligible call registry before annotation and verifier
scoring.

For every calibration-call identity, create this exact ordered string-value
array:

```text
[
  "nli_call_v1",
  task,
  dataset,
  source_sample_id,
  evidence_role,
  llm_logical_slot,
  retriever,
  treatment,
  acc_direction,
  claim_sha256,
  premise_type,
  premise_sha256,
  generation_id,
  decomposer_id,
  exposure_purpose
]
```

Apply these rules:

- every element is a UTF-8 string;
- missing or not-applicable values are exactly the empty string `""`;
- text identifiers use their already-frozen canonical repository spelling;
- SHA fields are lowercase hexadecimal;
- no field may be omitted.

Serialize this array exactly as compact JSON using:

- UTF-8;
- `ensure_ascii=false`;
- `separators=(",", ":")`;
- no indentation;
- no trailing newline.

Define:

```text
canonical_call_id =
SHA256(serialized_call_identity_UTF8).hexdigest()
```

Then create this exact ordered priority-material array:

```text
[
  "sprint3_nli_calibration_amend01_v1",
  "20260823",
  task,
  canonical_call_id
]
```

Serialize it with the same compact-JSON rule and define:

```text
selection_priority =
SHA256(serialized_priority_material_UTF8).hexdigest()
```

Sort ascending lexicographically by the 64-character lowercase hexadecimal
`selection_priority`.

If two registry records somehow have the same `selection_priority`:

1. sort by `canonical_call_id` ascending;
2. if `canonical_call_id` is also identical, treat them as duplicate
   scientific calls and retain exactly one canonical registry record;
3. preserve duplicate-detection evidence in provenance.

The seed material is therefore operationally active, and sample identity
cannot depend on an implementation-time serialization choice. This exact
serialization does not change the scientific fields bound by the registry.

The same canonical call-registry identity and `selection_priority` mechanism
also provides deterministic identity and ordering for targeted-validation
candidates, even though those candidates are excluded from threshold fitting.
Sections 12 and 14 freeze their task-specific premise identities. No second
random or implementation-specific targeted-validation ordering is permitted.

## 3. ACC Eligible Call Scope

Align NLI calibration with the frozen canonical ACC construct.

Canonical ACC calibration calls may come only from actual ACC verifier calls
that can arise from:

```text
relevance-only WITH_CONTEXT
versus
diversified WITH_CONTEXT
```

with both frozen comparison directions.

Do not introduce `WITHOUT_CONTEXT` ACC calibration calls merely because the
older NLI protocol wording said:

> `WITH_CONTEXT` versus `WITHOUT_CONTEXT` where operationally applicable.

For canonical ACC, that phrase is now resolved as:

```text
NOT OPERATIONALLY APPLICABLE
```

because `docs/sprint3/ACC_PROTOCOL.md` defines canonical ACC only for
relevance-only `WITH_CONTEXT` versus diversified `WITH_CONTEXT`.

This amendment does not broaden or narrow ACC itself. It only aligns
calibration with the already-frozen ACC call distribution.

ACC diversified-family strata are, in frozen order:

1. `MMR`;
2. `KMEANS`;
3. `AGGLOMERATIVE`;
4. `DPP`.

ACC direction strata are, in frozen order:

1. `BASELINE_CLAIM_TO_DIVERSIFIED_ANSWER`;
2. `DIVERSIFIED_CLAIM_TO_BASELINE_ANSWER`.

## 4. Faithfulness Eligible Call Scope

Threshold fitting and held-out faithfulness calibration use only canonical
single-passage verifier calls from `WITH_CONTEXT` answers.

Do not use passage-pair calls in threshold fitting or in the ordinary 60-case
held-out threshold validation. Pair behavior is validated separately by the
frozen 30-case pair-support gate.

Faithfulness structural method-family strata are, in frozen order:

1. `RELEVANCE_BASELINE`;
2. `MMR`;
3. `KMEANS`;
4. `AGGLOMERATIVE`;
5. `DPP`.

Retriever order is frozen as:

1. `BM25`;
2. `DPR`;
3. `CONTRIEVER`;
4. `COLBERTV2`.

## 5. Exact Dataset by LLM Initial Allocation

The wording “approximately” and the permission for a “minor deterministic
adjustment” do not apply to the initial fitting and held-out manifests.

Per task, freeze the 150 fitting units as:

| Dataset | `llama-3.3-70b` | `gemma4-26b` | `ministral-3-14b` | Total |
|---|---:|---:|---:|---:|
| PubMedQA | 17 | 17 | 16 | 50 |
| HotpotQA | 16 | 17 | 17 | 50 |
| ASQA | 17 | 16 | 17 | 50 |
| **Total** | **50** | **50** | **50** | **150** |

Per task, freeze the initial 60 held-out units as:

| Dataset | `llama-3.3-70b` | `gemma4-26b` | `ministral-3-14b` | Total |
|---|---:|---:|---:|---:|
| PubMedQA | 7 | 7 | 6 | 20 |
| HotpotQA | 6 | 7 | 7 | 20 |
| ASQA | 7 | 6 | 7 | 20 |
| **Total** | **20** | **20** | **20** | **60** |

No ordinary reallocation is permitted.

If any required dataset-by-LLM cell lacks enough valid eligible calls to
materialize the required manifest, stop with:

```text
NLI_CALIBRATION_INSUFFICIENT_ELIGIBLE_CALLS
```

Do not silently move quota to another dataset or LLM. A later redesign
requires a prospective amendment.

## 6. Exact Fit and Held-Out Disjointness

Within each task-by-dataset-by-LLM cell, Section 7's deterministic balancing
algorithm runs continuously to produce one complete ordered eligible-call
sequence. It is not restarted separately for `FIT` and `HELDOUT`.

Construct the sequence as follows:

1. initialize every stratum selected-count to zero;
2. repeatedly apply the Section 7 minimum-count, frozen-stratum-order, and
   lowest-`selection_priority` rule;
3. continue until every eligible call in that dataset-by-LLM cell has been
   ordered.

Then operate on that one balanced sequence:

1. assign the first required `FIT`-quota entries to the fitting manifest;
2. assign the next required `HELDOUT`-quota entries to the initial held-out
   manifest;
3. place every remaining entry in the deterministic reserve in that exact
   order.

Fitting and held-out units are therefore selected simultaneously and are
exactly disjoint.

For a fresh-60 expansion, continue from the next unused reserve entries. Skip
only entries that have prospectively become ineligible, including
`GUIDE_CHECK_ONLY` cases. Do not reset balancing counts when moving from
`FIT` to `HELDOUT` or reserve.

This exact continuous sequence is frozen and hashable before annotation and
before verifier scoring.

Freeze and hash both manifests before annotation and before verifier scoring.
Do not choose held-out units after observing fitting labels or scores.

## 7. Deterministic Method and Retriever Balancing

This section replaces “as evenly as practical” and “span broadly” with one
deterministic balancing algorithm.

Within every task-by-dataset-by-LLM cell:

- for ACC, balance over the tuple
  `(diversification_family, ACC_direction)` using the frozen family and
  direction orders in Section 3;
- for faithfulness, balance over `(retriever, method_family)` using the frozen
  retriever and method-family orders in Section 4.

Apply this selection algorithm continuously:

1. Among strata that still contain unused eligible calls, find the minimum
   number already selected from that stratum within the current
   dataset-by-LLM allocation.
2. Choose a stratum with that minimum count.
3. Break stratum ties using the frozen order above.
4. Within the chosen stratum, select its lowest remaining
   `selection_priority` eligible call.
5. Repeat until every eligible call in the dataset-by-LLM cell has been
   ordered.

If a stratum has no eligible calls, skip only that unavailable stratum.

Do not compensate using performance, labels, verifier scores, correctness, or
retrieval metrics. This algorithm defines exactly what “balanced as
available” means. The stratum selected-counts are initialized once for the
cell and are never reset at the `FIT`, `HELDOUT`, or reserve boundaries.

## 8. Ordered Annotation, Agreement, and Class-Minimum Procedure

Apply the following order separately for ACC and faithfulness. No verifier
scores are opened during these steps.

### Step A — Initial manifests

Materialize and freeze:

```text
150 FIT
+ 60 HELDOUT
```

according to the already-frozen allocation and sampling rules.

### Step B — First independent annotation

Both annotators independently label the complete original:

```text
150 FIT + 60 HELDOUT
```

set. Do not adjudicate yet for threshold-fitting purposes. Compute the frozen
agreement diagnostics on these independent labels.

### Step C — Kappa gate first

If:

```text
kappa >= 0.70
```

then:

- no guide clarification occurs;
- adjudicate the original 210 labels;
- use those adjudicated labels as the operative labels for the class-minimum
  check in Step D.

If:

```text
kappa < 0.70
```

then:

- activate the one guide-clarification cycle;
- freeze the revised guide;
- conduct the 12-unit `DEVELOPMENT` guide-check pilot defined in Section 11;
- independently rerate from scratch the complete original 150-plus-60 set;
- recompute kappa on the rerated independent labels.

If the rerated kappa remains below 0.70, stop with:

```text
NLI_CALIBRATION_BLOCKED_LOW_AGREEMENT
```

If the rerated kappa is at least 0.70, adjudicate the rerated 150-plus-60
labels. Only those rerated and adjudicated labels become operative for the
class-minimum check.

Do not trigger class-minimum expansion from first-pass labels when first-pass
kappa was below 0.70.

### Step D — Class-minimum check

Only after an accepted annotation pass exists—either original labels with
first-pass kappa at least 0.70 or revised-guide rerated labels with rerated
kappa at least 0.70—check:

```text
FIT:
  ENTAILMENT    >= 30
  CONTRADICTION >= 30

HELDOUT:
  ENTAILMENT    >= 15
  CONTRADICTION >= 15
```

If all minima pass, continue with `150 FIT + 60 HELDOUT`.

If any minimum fails, consume the protocol's one data-expansion allowance:

```text
original accepted 150 FIT
+ original accepted 60 HELDOUT
= 210 FIT

+ one fresh disjoint 60 HELDOUT
```

The fresh 60:

- uses exactly the dataset-by-LLM allocation from Section 5;
- is selected from the next unused reserve calls under the same deterministic
  ordering and balancing rules;
- is frozen and hashed before annotation and verifier scoring.

### Step E — Annotate the class-representation fresh 60

This step governs only a pre-verifier-score data expansion triggered by the
class-representation check in Step D. The post-verifier-score
threshold-validation expansion follows Section 9 Case B instead.

Both annotators independently label the fresh 60 using the same currently
frozen annotation guide:

- the original guide if no clarification occurred; or
- the revised frozen guide if the low-kappa clarification occurred.

Compute kappa on the fresh-60 independent labels.

If fresh-60 kappa is at least 0.70:

- adjudicate the fresh 60;
- proceed to the expanded class-minimum check below.

If fresh-60 kappa is below 0.70 and the guide-clarification allowance has not
previously been used:

- activate that same one guide-clarification allowance;
- revise, freeze, version, and hash the guide once;
- run the 12-unit guide-check pilot if it has not already been run;
- independently rerate from scratch the complete active 270-unit set:

  ```text
  210 FIT + fresh 60 HELDOUT
  ```

- compute both:
  1. overall three-way Cohen's kappa on the complete rerated 270-unit set;
  2. three-way Cohen's kappa on the rerated fresh-60 `HELDOUT` subset alone.

If fresh-60 kappa is below 0.70 and the one guide clarification was already
consumed earlier, stop immediately with:

```text
NLI_CALIBRATION_BLOCKED_LOW_AGREEMENT
```

Do not use a combined 270-unit kappa to hide sub-threshold agreement in the
newly collected fresh held-out block.

After revised-guide rerating, require both:

```text
complete_270_kappa >= 0.70
AND
fresh_60_kappa >= 0.70
```

Only if both pass, adjudicate the complete rerated 270-unit set and proceed to
the expanded class-minimum check.

If either fails, stop with:

```text
NLI_CALIBRATION_BLOCKED_LOW_AGREEMENT
```

There is no second guide clarification, no second guide-check pilot, and no
additional data expansion. Preserve raw agreement statistics for the complete
270 and the fresh 60 separately.

After either acceptable agreement path—fresh-60 kappa at least 0.70 without
rerating, or both rerated agreement gates passing—check the expanded class
minima:

```text
210 FIT:
  ENTAILMENT    >= 30
  CONTRADICTION >= 30

fresh 60 HELDOUT:
  ENTAILMENT    >= 15
  CONTRADICTION >= 15
```

Do not increase these class minima merely because fitting became `N=210`.

If the expanded class minima fail, stop the affected task with:

```text
NLI_CALIBRATION_INSUFFICIENT_CLASS_REPRESENTATION
```

No second data expansion is permitted.

The one data-expansion allowance and one guide-clarification allowance are
different bounded mechanisms:

- maximum one `150+60 -> 210+fresh60` data expansion;
- maximum one annotation-guide clarification and rerating cycle.

Do not create additional calibration data because of low kappa, apart from
the fixed 12 guide-check units.

## 9. Single Expansion Budget Is Shared

The same one-expansion allowance is used for either:

- class-representation failure before verifier scoring; or
- first held-out threshold-validation failure after fitting.

It is not one expansion for each trigger.

This is the data-expansion budget described in Section 8. It is distinct from
the one annotation-guide clarification allowance. A low-kappa guide
clarification does not itself consume the data-expansion budget.

### Case A

If expansion was already consumed because of class-minimum failure and the
fresh held-out threshold validation later fails, calibration is terminally
failed.

### Case B — Threshold-validation-triggered data expansion

Case B occurs only after original held-out verifier validation has failed. At
that point verifier performance has already been observed, so no
annotation-guide change is permitted.

The prerequisites are:

- the original `150 FIT + 60 HELDOUT` already passed the
  annotation-agreement and class-minimum gates;
- thresholds were fitted on the original 150;
- the frozen threshold failed its first held-out validation;
- the one data-expansion allowance has not previously been consumed.

Then:

1. Combine the already accepted and adjudicated original 150 `FIT` plus the
   original failed 60 `HELDOUT` into the 210-unit expanded `FIT` set.
2. Deterministically materialize exactly one fresh, disjoint 60-unit
   `HELDOUT` set from the next unused reserve calls using the same frozen
   allocations and continuous balanced ordering.
3. Freeze and hash the fresh-60 manifest before any human inspection of those
   fresh cases.
4. Use the currently frozen annotation guide exactly as it existed before
   verifier-score observation.
5. Have both annotators independently label all fresh 60 cases.
6. Compute three-way Cohen's kappa on the fresh 60 and require:

   ```text
   kappa >= 0.70
   ```

   If fresh-60 kappa is below 0.70, stop with:

   ```text
   NLI_CALIBRATION_BLOCKED_LOW_AGREEMENT
   ```

   Do not revise the guide, run another guide-check pilot, rerate earlier
   labels, or collect another held-out set. Verifier-performance evidence has
   already been observed.
7. If fresh-60 kappa is at least 0.70, adjudicate the fresh 60.
8. Require these fresh-held-out class minima:

   ```text
   ENTAILMENT    >= 15
   CONTRADICTION >= 15
   ```

   If either minimum fails, stop with:

   ```text
   NLI_CALIBRATION_INSUFFICIENT_CLASS_REPRESENTATION
   ```

   No second data expansion is permitted.
9. Fit the affected threshold again on the 210-unit expanded `FIT` set using
   the frozen base-protocol procedure.
10. Apply it exactly once to the fresh 60 `HELDOUT`.

If the expanded threshold validation fails, stop with:

```text
NLI_CALIBRATION_FAILED_AFTER_SINGLE_EXPANSION
```

Do not change the threshold rule, precision target, count gates, guide,
verifier, or sampling.

### Case C

After the one expansion has been used, no additional expansion is permitted
for any reason.

The terminal status for a failed affected component is:

```text
NLI_CALIBRATION_FAILED_AFTER_SINGLE_EXPANSION
```

Do not weaken thresholds, the precision target, count gates, or replace the
verifier.

## 10. Low-Kappa Guide Clarification: Exactly One

Preserve:

```text
kappa < 0.70
```

as the mandatory annotation-guide review trigger.

Compute the already-required overall three-way Cohen's kappa separately for:

- ACC calibration annotation;
- faithfulness calibration annotation.

Class-specific agreement remains reported diagnostic evidence and does not
create a new independent hard threshold.

The exact kappa-before-adjudication sequence, including the fresh-60 path, is
governed by Section 8. Across that entire sequence, allow at most one
guide-clarification cycle for the affected task.

Annotation-guide clarification is permitted only before any verifier score or
probability from the active threshold-calibration path has been inspected.
Once threshold verifier performance has been observed, the guide is frozen
for the remainder of that task's calibration attempt. This does not prohibit
the prospectively defined data expansion after held-out threshold failure; it
only prohibits changing the human measurement instrument after verifier
performance is seen.

The clarification may only:

- clarify wording;
- add neutral examples;
- clarify boundaries between `ENTAILMENT`, `NEUTRAL`, and `CONTRADICTION`;
- resolve ambiguity already present in the existing categories.

It may not:

- change scientific labels;
- change evidence construction;
- change thresholds;
- change sample membership;
- introduce task-specific score rules;
- inspect verifier scores;
- use `SELECTION` or protected evidence.

Freeze, version, and hash the revised guide.

## 11. Guide-Check Pilot and Complete Rerating

If the one guide clarification activates, materialize exactly:

```text
12 additional DEVELOPMENT guide-check units
```

for the affected task.

These 12 units are:

- training and guide-check evidence only;
- excluded from threshold fitting;
- excluded from held-out validation;
- excluded from pair and window integration gates.

Allocate them as follows:

| Dataset | `llama-3.3-70b` | `gemma4-26b` | `ministral-3-14b` | Total |
|---|---:|---:|---:|---:|
| PubMedQA | 2 | 1 | 1 | 4 |
| HotpotQA | 1 | 2 | 1 | 4 |
| ASQA | 1 | 1 | 2 | 4 |
| **Total** | **4** | **4** | **4** | **12** |

Select the 12 units from the next unused deterministic reserve calls.

Any 12-unit guide-check case, once selected, is permanently marked
`GUIDE_CHECK_ONLY` and is removed from eligibility for `FIT`, `HELDOUT`,
fresh-60 expansion, faithfulness-pair validation, and ACC-window validation.
Later reserve selection deterministically skips those consumed guide-check
calls. A triggered guide clarification may therefore change which later
reserve calls become the fresh 60, but only through this prospectively frozen
agreement-governance path and never through verifier performance.

Both annotators independently label the 12 pilot calls using the revised
guide.

After the guide-check pilot, both annotators independently rerate from scratch
every call in the complete active calibration set for that task, either:

```text
150 FIT + 60 HELDOUT
```

or, if the expansion path already activated:

```text
210 FIT + fresh 60 HELDOUT
```

Preserve original ratings as:

```text
INITIAL_NLI_AGREEMENT_TRIGGER_LABELS
```

Preserve revised-guide ratings as:

```text
FINAL_NLI_CALIBRATION_RAW_LABELS
```

Do not overwrite or mix first-pass and rerated labels.

For an active original `150 FIT + 60 HELDOUT` set, recompute overall three-way
Cohen's kappa and apply Section 8 Step C. For an active class-triggered
`210 FIT + fresh 60 HELDOUT` set, compute both the complete-270 and rerated
fresh-60 kappas and apply both gates in Section 8 Step E. Section 9 Case B
retains its separate post-verifier-score rule.

No second guide clarification, second pilot, threshold weakening, or
annotation-category redesign is permitted without a new prospective
methodology amendment.

## 12. Faithfulness Pair-Support Validation Manifest

The 30 pair-support cases are additional `DEVELOPMENT` cases. They must use
claims that are not present in the active faithfulness threshold-fitting or
held-out manifests.

Before human inspection, freeze a complete deterministic candidate registry
of valid canonical:

```text
(claim, passage_i, passage_j)
```

units.

Represent each valid canonical pair candidate as a faithfulness Section 2
registry record with:

```text
premise_type = "FAITHFULNESS_PAIR"
premise_sha256 =
SHA-256 of the exact frozen two-passage serialized premise
```

Bind every other applicable Section 2 scientific identity field. Its
candidate priority is exactly the Section 2 `selection_priority`.

Order this pre-human registry by that priority and freeze the order before
human labeling. No second or random pair ordering is permitted, and
implementation-specific filesystem or source order must not affect sampling.

Use only:

- exact canonical retrieved passage bodies;
- the frozen pair serialization;
- the frozen pair ordering;
- `DEVELOPMENT` evidence;
- claims from the frozen decomposer winner.

Do not use verifier scores or predictions to construct the candidate
registry.

Two annotators independently label:

1. the combined pair as `SUPPORT` or `NO_SUPPORT`;
2. `passage_i` alone as `SUPPORT` or `NO_SUPPORT`;
3. `passage_j` alone as `SUPPORT` or `NO_SUPPORT`.

After independent human labeling and adjudication, assign every candidate to
exactly one of:

```text
PAIR_REQUIRED_SUPPORT:
  pair = SUPPORT
  passage_i = NO_SUPPORT
  passage_j = NO_SUPPORT

ONE_PASSAGE_SUFFICIENT_SUPPORT:
  pair = SUPPORT
  AND at least one individual passage = SUPPORT

NO_SUPPORT:
  pair = NO_SUPPORT
```

Freeze category order as:

1. `PAIR_REQUIRED_SUPPORT`;
2. `ONE_PASSAGE_SUFFICIENT_SUPPORT`;
3. `NO_SUPPORT`.

The final 30-case manifest must contain exactly:

| Category | Count |
|---|---:|
| `PAIR_REQUIRED_SUPPORT` | 10 |
| `ONE_PASSAGE_SUFFICIENT_SUPPORT` | 5 |
| `NO_SUPPORT` | 15 |
| **Total** | **30** |

For each category independently:

1. filter the pre-frozen candidate registry to adjudicated candidates in that
   category;
2. retain their original frozen Section 2 `selection_priority` values;
3. select the required category quota using deterministic balancing over
   `(dataset, LLM logical slot)`;
4. use this frozen dataset order:
   1. `PUBMEDQA`;
   2. `HOTPOTQA`;
   3. `ASQA`;
5. use this frozen LLM order:
   1. `llama-3.3-70b`;
   2. `gemma4-26b`;
   3. `ministral-3-14b`;
6. repeatedly:
   1. among dataset-by-LLM cells containing unused candidates, find the
      minimum count already selected from that cell for the current category;
   2. choose the first tied cell under the frozen dataset order and then the
      frozen LLM order;
   3. within that cell, choose the unused candidate with the lowest original
      Section 2 `selection_priority`;
   4. repeat until the category quota is filled.

Do not first take a global prefix and then rebalance it. Do not replace
selected cases after verifier scores are observed.

This is human-label-based targeted validation, not verifier-score-based
selection. It remains prospective because:

- the candidate universe and ordering are frozen before human labels;
- verifier scores are not inspected;
- category definitions and quotas are frozen before annotation.

Do not inspect pair verifier scores until the final 30-case manifest is
frozen.

If any category cannot reach its exact quota, stop with:

```text
FAITHFULNESS_PAIR_VALIDATION_INSUFFICIENT_CASES
```

Do not weaken the composition post hoc.

## 13. Pair-Support Human and Model Comparison

For each final pair-support case, the human binary gold is:

```text
SUPPORT
or
NO_SUPPORT
```

The automated binary decision is:

```text
SUPPORT
iff
p_entailment(pair) >= frozen faithfulness entailment threshold
```

Otherwise, the automated decision is `NO_SUPPORT`.

The contradiction threshold does not define pair support.

Any technical verifier failure on a final pair case:

- is preserved as `FAILED`;
- is counted as an incorrect integration decision for the 27/30 gate;
- is not replaced after verifier scoring.

The pass rule remains:

```text
at least 27/30 correct
```

Do not fit a pair-specific threshold.

## 14. ACC Window-Validation Candidate Registry

The 30 ACC window cases must validate actual windowing.

An eligible case must:

- be `DEVELOPMENT` evidence;
- contain a valid decomposed ACC claim;
- have a complete opposing answer that exceeds the frozen verifier capacity;
- therefore invoke the canonical ACC sentence-preserving window procedure.

Do not fill the 30-case set with whole answers that fit without windowing.

Claims used in the final window-validation manifest must not appear in the
active ACC threshold-fitting or held-out manifests.

Represent each valid window-required candidate as an ACC Section 2 registry
record with:

```text
premise_type = "ACC_COMPLETE_OPPOSING_ANSWER_WINDOW_REQUIRED"
premise_sha256 =
SHA-256 of the exact complete opposing-answer surface before windowing
```

Bind every other applicable Section 2 scientific identity field. Its
candidate priority is exactly the Section 2 `selection_priority`. The windows
themselves do not define sampling priority; the complete-answer scientific
call does.

Freeze the complete eligible window-required candidate registry and its
Section 2 ordering before human annotation and verifier scoring. Do not allow
implementation-specific filesystem or source order to affect targeted-case
sampling.

Because operational windowing may occur primarily in long-form datasets, do
not impose an artificial exact 10/10/10 dataset quota.

Use this frozen dataset order:

1. `PUBMEDQA`;
2. `HOTPOTQA`;
3. `ASQA`.

Use this frozen LLM order:

1. `llama-3.3-70b`;
2. `gemma4-26b`;
3. `ministral-3-14b`.

Select exactly 30 from the pre-frozen eligible registry by repeatedly:

1. among dataset-by-LLM cells containing unused eligible candidates, find the
   minimum count already selected from that cell;
2. choose the first tied cell under the frozen dataset order and then the
   frozen LLM order;
3. within that cell, choose the unused candidate with the lowest Section 2
   `selection_priority`;
4. repeat until 30 cases are selected.

Skip unavailable cells.

If fewer than 30 valid `DEVELOPMENT` window-requiring bundles exist, stop with:

```text
ACC_WINDOW_VALIDATION_INSUFFICIENT_CASES
```

Do not substitute non-windowed examples.

## 15. ACC Window Human Label Contract

For each final window bundle, both annotators independently label:

```text
atomic claim
versus
COMPLETE OPPOSING ANSWER
```

using exactly:

- `ENTAILMENT`;
- `NEUTRAL`;
- `CONTRADICTION`.

They do not label individual windows as the reference outcome for the
integration gate.

Preserve raw labels and adjudicate disagreements before verifier comparison.

The verifier then:

1. applies the frozen window construction;
2. scores every window;
3. applies the frozen ACC entailment threshold;
4. applies the frozen ACC contradiction threshold;
5. aggregates using the frozen maximum-entailment and maximum-contradiction
   semantics.

Possible automated aggregate states are:

- `ENTAILMENT`;
- `NEUTRAL`;
- `CONTRADICTION`;
- `INCONSISTENT`;
- `FAILED`.

## 16. Window Inconsistent Rule

For the 30-case window integration gate:

- automated `ENTAILMENT` matches only human `ENTAILMENT`;
- automated `NEUTRAL` matches only human `NEUTRAL`;
- automated `CONTRADICTION` matches only human `CONTRADICTION`.

Automated `INCONSISTENT` does not count as agreement with any of the three
human labels. It counts as one integration mismatch.

Automated `FAILED` also counts as one integration mismatch.

Do not replace `INCONSISTENT` or `FAILED` cases after scoring.

The pass rule remains:

```text
at least 27/30 exact matches
```

Also report `INCONSISTENT` and `FAILED` counts separately.

This rule validates whether window aggregation reproduces the complete-answer
three-way judgment. It does not redefine canonical ACC's downstream
scientific `INCONSISTENT` semantics.

## 17. Targeted-Set Disjointness

Freeze:

- faithfulness pair-support claims are disjoint from active faithfulness
  threshold-fitting and held-out claims;
- ACC window-validation claims are disjoint from active ACC threshold-fitting
  and held-out claims;
- pair-support and ACC-window sets remain different task artifacts;
- every 12-unit low-kappa guide-pilot case is permanently
  `GUIDE_CHECK_ONLY`, is skipped by later reserve selection, and is excluded
  from every fitting, held-out, pair, and window manifest and score.

No targeted validation case may migrate into threshold fitting after its
integration result is observed.

## 18. No Post-Hoc Replacement

Once any final manifest is frozen, do not replace a case because of:

- verifier failure;
- unexpected label;
- disagreement;
- low score;
- surprising prediction;
- poor pair or window integration behavior.

Pre-verifier construction failures discovered before manifest freeze may be
handled only through the frozen deterministic eligibility and reserve rule.
After freeze, failure remains failure.

## 19. Provenance

In addition to the base protocol, preserve:

- Amendment 01 commit and hash;
- eligible call-registry hashes;
- exact call-identity and priority-serialization versions;
- exact SHA namespace and seed material;
- duplicate-registry-record detection evidence;
- `FIT` manifest hash;
- initial `HELDOUT` manifest hash;
- whether the single expansion activated;
- expansion trigger;
- fresh-60 manifest hash, if used;
- deterministic reserve ordering;
- method and retriever stratum counts;
- dataset-by-LLM counts;
- initial class counts;
- final class counts;
- guide version and hash;
- whether low-kappa clarification activated;
- 12-unit guide-pilot manifest and hash, if activated;
- `INITIAL_NLI_AGREEMENT_TRIGGER_LABELS` hash;
- `FINAL_NLI_CALIBRATION_RAW_LABELS` hash;
- `rerated_complete_270_kappa`, where applicable;
- `rerated_fresh_60_kappa`, where applicable;
- pair candidate-registry hash;
- final pair manifest and hash and category counts;
- window candidate-registry hash;
- final window manifest and hash;
- `INCONSISTENT` count;
- technical failure counts;
- terminal status, where relevant.

## 20. Pre-Observation Status

This amendment is frozen before:

- NLI verifier scores or probabilities for the calibration manifests are
  inspected;
- fitted thresholds are known;
- pair or window verifier integration results are known;
- `SELECTION` outcomes are opened;
- protected-final outcomes are opened.

Annotation-guide clarification is permitted only before any verifier score or
probability from the active threshold-calibration path has been inspected.
Once threshold verifier performance has been observed, the guide is frozen
for the remainder of that task's calibration attempt. The already-declared
data expansion after held-out threshold failure remains permitted, but it
cannot change the human measurement instrument.

No observed verifier performance was used to choose:

- the exact allocation;
- deterministic ordering;
- one-expansion sharing;
- guide-rerating policy;
- pair-support composition;
- the window-`INCONSISTENT` rule.

## 21. What This Amendment Does Not Change

This amendment does not change:

- verifier family;
- verifier label mapping;
- ACC metric;
- faithfulness metric;
- decomposer;
- precision target;
- threshold search;
- predicted-positive minima;
- held-out precision gate;
- pair-threshold policy;
- window thresholds;
- the 27/30 integration gates;
- evidence roles;
- failure-as-NA semantics in final metrics;
- Stage-3 `SELECTION` rules;
- protected-final methodology.
