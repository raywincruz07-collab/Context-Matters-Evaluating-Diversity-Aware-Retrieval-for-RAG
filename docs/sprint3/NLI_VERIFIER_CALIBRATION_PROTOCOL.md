# NLI Verifier Calibration and Provenance Protocol

## 1. Purpose

This protocol freezes the prospective calibration, human-validation,
physical-provenance, and admission protocol for the already-selected NLI
verifier used by:

1. Bidirectional Atomic-Content Coverage (`ACC_bi`); and
2. context faithfulness.

It does not reopen:

- the verifier family;
- the ACC construct;
- the faithfulness construct;
- evidence staging;
- ACC whole-answer and window semantics;
- faithfulness single-passage and passage-pair semantics.

## 2. Verifier Family

The canonical verifier family is:

```text
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

The scientific label semantics are:

```text
0 = ENTAILMENT
1 = NEUTRAL
2 = CONTRADICTION
```

One shared verifier model is used. Four thresholds are calibrated separately:

- ACC entailment threshold;
- ACC contradiction threshold;
- faithfulness entailment threshold;
- faithfulness contradiction threshold.

There are no model-specific, dataset-specific, retriever-specific, or
LLM-specific thresholds.

## 3. Physical Snapshot Policy

The upstream head observed during an audit must not automatically become the
frozen snapshot.

At acquisition, before calibration:

1. resolve the model repository once to a full immutable commit SHA;
2. record the repository ID, resolved SHA, and resolution timestamp;
3. acquire only that immutable snapshot;
4. explicitly select `model.safetensors`;
5. record the used-file inventory, file sizes, and SHA-256 hashes;
6. record the tokenizer files and configuration;
7. verify the architecture and `id2label`/`label2id` mappings;
8. pin a compatible evaluator environment;
9. use `local_files_only=True` after acquisition; and
10. prohibit any online or floating-`main` fallback.

The physical snapshot identity must remain unchanged between calibration and
canonical execution.

## 4. Inference Identity

The following behavior and provenance must be frozen and recorded:

- premise first;
- hypothesis second;
- evidence or opposing answer as premise;
- atomic claim as hypothesis;
- no silent tokenizer truncation;
- project-controlled capacity and window handling;
- special-token accounting;
- padding strategy and side;
- batch size;
- dtype and device;
- `eval()`;
- `torch.inference_mode()`;
- `trust_remote_code=False`;
- Transformers, tokenizers, safetensors, PyTorch, CUDA, and driver versions;
- deterministic-runtime settings;
- persistence of full-precision logits and probabilities for calibration.

The exact physical values are Stage-1/Stage-2 implementation closures. Closing
them does not reopen the verifier family or scientific constructs.

## 5. Evidence Roles

Calibration uses only `DEVELOPMENT` or evaluator-exposed evidence.

Allowed evidence is:

- PubMedQA historical/development control evidence;
- HotpotQA BEIR-train `DEVELOPMENT`;
- ASQA internal `DEVELOPMENT`.

Forbidden evidence is:

- HotpotQA `SELECTION`;
- ASQA `SELECTION`;
- the HotpotQA protected-final 500;
- ASQA dev948 `PROJECT_PROTECTED_FINAL`.

ASQA calibration IDs must receive evaluator-development exposure bookkeeping.
Calibration sampling and manifests must be frozen before verifier scores are
inspected.

## 6. Human Calibration Units

### ACC call-level unit

The ACC unit is:

```text
(
  atomic claim as hypothesis,
  exact whole opposing answer OR exact operational ACC window as premise,
  human three-way label
)
```

### Faithfulness single-passage unit

The faithfulness unit is:

```text
(
  atomic claim as hypothesis,
  exact canonical passage body as premise,
  human three-way label
)
```

The human labels are:

- `ENTAILMENT`;
- `NEUTRAL`;
- `CONTRADICTION`.

Malformed claims and construction failures are `FAILED`, not neutral examples.
Only the frozen decomposer winner may supply claims.

## 7. Sample Sizes

Sample sizes are frozen per task, not per threshold.

### ACC

- 150 call-level threshold-fitting units;
- 60 disjoint held-out call-level units.

### Faithfulness

- 150 single-passage threshold-fitting units;
- 60 disjoint held-out single-passage units.

The same ACC fitting set calibrates the separate ACC entailment and
contradiction thresholds. The same faithfulness fitting set calibrates the
separate faithfulness entailment and contradiction thresholds.

This must not be interpreted as 150 examples for each of four thresholds.

## 8. Dataset Representation

Each 150-unit fitting set contains approximately:

- 50 PubMedQA units;
- 50 HotpotQA units;
- 50 ASQA units.

Each 60-unit held-out set contains approximately:

- 20 PubMedQA units;
- 20 HotpotQA units;
- 20 ASQA units.

A minor deterministic adjustment is permitted only where valid operational
units are unavailable. Any adjustment must be documented before verifier
scoring.

## 9. LLM Representation

Within every dataset allocation, represent the following generator LLMs as
evenly as integer counts permit:

- `llama-3.3-70b`;
- `gemma4-26b`;
- `ministral-3-14b`.

No LLM-specific threshold is permitted.

## 10. Method Representation

ACC should span:

- `WITH_CONTEXT` versus `WITHOUT_CONTEXT` where operationally applicable;
- baseline `WITH_CONTEXT` versus diversified `WITH_CONTEXT`;
- both comparison directions.

Faithfulness should span:

- baseline `WITH_CONTEXT`;
- diversified `WITH_CONTEXT`;
- all four retrievers as evenly as practical.

Diversified examples should span broadly:

- MMR;
- clustering;
- DPP.

Calibration need not represent every diversification hyperparameter. No
method-performance result may influence calibration sampling.

## 11. Class and Challenge Enrichment

Natural contradiction frequency may be too small for threshold fitting.
Controlled enrichment must therefore use natural DEVELOPMENT-derived verifier
calls.

Do not synthetically alter claim or evidence text merely to manufacture easy
contradictions. Do not use verifier probabilities or scores to decide which
cases enter calibration. Use human labels and prospectively defined
deterministic candidate sampling.

Minimum adjudicated class representation is:

### Fitting, per 150-unit task set

- at least 30 `ENTAILMENT` units;
- at least 30 `CONTRADICTION` units.

### Held-out, per 60-unit task set

- at least 15 `ENTAILMENT` units;
- at least 15 `CONTRADICTION` units.

The remainder may be `NEUTRAL` or naturally distributed valid calls.

If a quota is not reached, expand the DEVELOPMENT annotation candidate pool
using predeclared deterministic blocks before verifier scores are inspected.

Because these sets are challenge/class enriched, measured precision is
precision on the frozen calibration distribution. It must not be described as
natural-prevalence or real-world positive predictive value.

No 35% enrichment rule is frozen.

## 12. Human Annotation

Two annotators independently label every fitting and held-out validation unit.

Annotators are blind to:

- verifier scores;
- threshold;
- LLM identity;
- retriever;
- diversification method;
- candidate or configuration identity.

Annotators see only the exact claim and exact premise needed for the NLI
decision. Use a written task-specific annotation guide. Conduct a small
DEVELOPMENT-only training/pilot batch that is excluded from threshold fitting.

Preserve independent labels and adjudicate disagreements before threshold
fitting. Record ambiguity and adjudication rationale. Do not use annotator
confidence as a threshold-fitting weight.

## 13. Agreement Policy

Report:

- raw three-way agreement;
- Cohen's kappa;
- class-specific agreement for entailment;
- class-specific agreement for contradiction;
- disagreement counts and categories.

Cohen's kappa is a diagnostic, not the sole hard admission gate, because it is
affected by category prevalence and marginal distributions.

Use:

```text
kappa < 0.70
```

as a mandatory annotation-guide and disagreement review trigger, not automatic
calibration failure.

If triggered:

1. inspect disagreements without verifier scores;
2. clarify the guide using `DEVELOPMENT` only;
3. use a new pilot/training batch if the guide changes;
4. preserve the original labels and evidence.

Persistent conceptual disagreement after guide revision requires explicit
methodological or supervisor adjudication before calibration proceeds.

No unsupported universal hard gate of raw agreement greater than or equal to
0.85 is frozen.

## 14. Precision Operating Target

Freeze:

```text
precision target = 0.90
```

for all four binary target-class threshold calibrations.

This is a **project-defined conservative operating target**. It is not:

- a universal literature standard;
- a claim that population precision is at least 0.90 with 95% confidence;
- a Wilson lower-confidence-bound requirement.

Use the same target for entailment and contradiction for parsimony while
fitting their probability thresholds separately.

## 15. Uncertainty Reporting

For every fitting and held-out precision estimate, report:

- predicted-positive `N`;
- true positives (`TP`);
- false positives (`FP`);
- precision;
- recall;
- standard 95% Wilson confidence interval.

Do not use Wald intervals. The Wilson interval communicates finite-sample
uncertainty and is not a separate hard threshold-admission condition.

No Wilson lower-bound-greater-than-or-equal-to-0.75 gate is frozen.

## 16. Threshold Search

For each of the following:

- ACC entailment;
- ACC contradiction;
- faithfulness entailment;
- faithfulness contradiction;

apply this procedure:

1. use only the 150-unit fitting set;
2. define a human target positive as the adjudicated target class;
3. define candidate thresholds as the sorted unique full-precision observed
   target-class probabilities plus `0.0` and `1.0` boundaries;
4. classify the target as positive when `p_target >= threshold`;
5. compute `TP`, `FP`, `FN`, `TN`, precision, recall, and Wilson interval;
6. exclude thresholds with fewer than 20 predicted positives;
7. among the remaining thresholds satisfying point-estimate precision greater
   than or equal to 0.90, choose the **lowest** threshold;
8. persist the complete threshold-search table;
9. freeze the selected threshold before held-out validation is opened.

Because positive sets are nested by threshold, the lowest eligible threshold
is the recall-favoring choice under the precision constraint. Do not use a
coarse 0.01 grid.

## 17. Held-Out Validation

Apply each frozen fitted threshold exactly once to its 60-unit held-out set.

A threshold passes if and only if:

- predicted-positive `N >= 15`;
- point-estimate precision `>= 0.90`.

Always report the standard 95% Wilson confidence interval. The interval is
uncertainty evidence, not a hidden additional pass criterion.

Do not tighten or relax the threshold after viewing the held-out set.

## 18. One Predeclared Expansion

If a required threshold fails its first held-out validation:

1. do not inspect `SELECTION`;
2. do not adjust the threshold against the failed held-out set;
3. preserve all artifacts;
4. allow the original 150 fitting units plus 60 held-out units to become a
   210-unit `DEVELOPMENT` fitting set;
5. materialize exactly one fresh, disjoint 60-unit `DEVELOPMENT` held-out set
   under the same frozen sampling and class-representation rules;
6. refit using the 210 fitting units;
7. freeze the new threshold;
8. evaluate exactly once on the fresh 60-unit held-out set.

No further iterative expansion is allowed under this protocol.

If the second held-out validation fails:

- the affected verifier component fails calibration;
- do not tune on `SELECTION`;
- do not silently lower the target;
- do not silently replace the verifier;
- any redesign requires a prospective protocol amendment;
- if not amended, report the affected canonical evaluator component as
  unavailable or as having failed validation.

## 19. Faithfulness Pair-Support Validation

Use 30 additional targeted `DEVELOPMENT` pair-support cases. Keep these cases
distinct from ACC window validation. Use the same frozen faithfulness
entailment threshold used for single passages.

Include:

- genuine support requiring two passages;
- one-passage-sufficient cases;
- related-but-unsupported pairs;
- negation;
- numeric, date, or entity mismatch;
- hedging and modality cases.

The project integration gate is:

```text
at least 27/30 correct binary support/no-support decisions
```

Also report:

- confusion counts;
- precision and recall where defined;
- standard Wilson confidence interval;
- failure count.

The 27/30 rule is a project-defined targeted integration criterion, not proof
of 90% population reliability. Do not create a separate pair threshold after
observing these cases. Failure requires prospective amendment and instrument
review, not pair-specific post-hoc tuning.

## 20. Windowed ACC Validation

Use 30 additional `DEVELOPMENT` long-answer bundles.

For each bundle:

1. humans label the claim against the complete opposing answer;
2. apply the frozen deterministic ACC window construction;
3. use the same ACC entailment and contradiction thresholds on every window;
4. aggregate with the frozen maximum-entailment and maximum-contradiction
   semantics;
5. compare the aggregated decision with the complete-answer human label.

The project integration gate is:

```text
at least 27/30 agreement
```

Also report confusion and error categories and standard Wilson uncertainty.
The 27/30 rule is a project integration criterion, not proof of 90% population
reliability.

Do not combine these 30 cases with the 30 faithfulness pair-support cases into
one 60-case score because they validate different scientific mechanisms. Do
not create window-specific thresholds.

## 21. Contradiction Semantics

Contradiction is predicted only from:

```text
p_contradiction >= frozen contradiction threshold
```

Do not define contradiction as merely not entailed. Neutral and contradiction
remain distinct. For faithfulness, contradiction evaluation remains
single-passage only, as already frozen.

## 22. Failure Semantics

Any of the following remains `FAILED`:

- tokenizer-capacity construction failure;
- malformed operational verifier call;
- model-load failure;
- inference exception;
- non-finite output;
- provenance mismatch.

Never map technical failure to:

- `NEUTRAL`;
- `UNSUPPORTED`;
- zero;
- non-entailment.

## 23. Repeatability Gate

The engineering and provenance validation uses:

- 12 frozen `DEVELOPMENT` or synthetic inputs;
- short, near-capacity, pair, and windowed calls;
- 3 fresh model-load executions;
- identical hardware and software inference identity.

Require:

- identical argmax class;
- identical final threshold decisions;
- finite logits and probabilities;
- maximum absolute probability difference `<= 1e-5`.

This is an engineering reproducibility convention, not a statistical
population claim. Do not claim bit-identical cross-hardware inference.

## 24. Calibration Artifact

Preserve:

- protocol and schema version;
- fitting, held-out, pair, and window manifest hashes;
- source sample IDs;
- dataset and evidence roles;
- evaluator exposure labels;
- exact claim and premise hashes;
- independent human labels;
- adjudicated labels;
- agreement statistics;
- ambiguity and adjudication notes;
- immutable model and tokenizer provenance;
- inference environment;
- raw logits and full-precision probabilities;
- complete threshold-search tables;
- selected thresholds;
- fitting and held-out metrics and Wilson intervals;
- pair and window validation outputs;
- failures;
- Git commit;
- calibration artifact hash.

Do not expose personal annotator identifiers publicly.

## 25. Evaluator Bundle

One hash-addressed evaluator bundle must bind:

- frozen decomposer winner and version;
- verifier immutable physical snapshot;
- tokenizer;
- inference configuration and environment;
- ACC entailment threshold;
- ACC contradiction threshold;
- faithfulness entailment threshold;
- faithfulness contradiction threshold;
- ACC windowing and aggregation version;
- faithfulness single/pair evidence-construction version;
- metric and state-classification versions;
- calibration artifact hash.

Result rows reference the bundle hash rather than duplicating all contents.

## 26. Stage-Gate Alignment

The required order is:

1. freeze the decomposer winner;
2. acquire and pin the immutable verifier snapshot;
3. freeze operational inference, window, and pair serialization;
4. freeze `DEVELOPMENT` calibration manifests and exposure accounting;
5. annotate and adjudicate;
6. fit thresholds;
7. lock thresholds;
8. open held-out `DEVELOPMENT` validation;
9. perform pair and window targeted validation;
10. pass repeatability and implementation gates;
11. freeze the evaluator bundle;
12. only then open Stage 3 `SELECTION`.

No calibration threshold or construction change is permitted after
`SELECTION` is opened. Protected-final data never calibrates the evaluator.

## 27. Epistemic Status of Numbers

### Literature and statistical support

- Wilson/score intervals are preferable to naive Wald uncertainty for small
  binomial samples.
- Small predicted-positive denominators produce wide uncertainty.
- Cohen's kappa is affected by prevalence and marginal distributions.

### Project conventions

- point precision target 0.90;
- 150 fitting and 60 held-out units per task;
- minimum 30 entailment and 30 contradiction fitting units;
- minimum 15 entailment and 15 contradiction held-out units;
- predicted-positive minimum 20 for fitting and 15 for held-out validation;
- equal dataset and LLM balancing;
- 30 pair-support cases;
- 30 windowed-ACC cases;
- 27/30 targeted integration criterion;
- 12 inputs by 3 loads for the repeatability gate;
- `1e-5` repeatability tolerance;
- one allowed fresh-validation expansion.

Project conventions must not be presented as literature standards.

## 28. Frozen Before Selection

This protocol is frozen before calibration outcomes, `SELECTION` outcomes, or
protected-final outcomes are observed.

Actual threshold values are empirical outputs of the frozen DEVELOPMENT-only
calibration procedure. They are not methodology decisions made after results.

No threshold may be altered because `SELECTION` or protected-final results look
poor.
