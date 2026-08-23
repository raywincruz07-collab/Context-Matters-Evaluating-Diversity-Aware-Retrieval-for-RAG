# Decomposer Selection Protocol Amendment 02

## 1. Authority and Scope

This is a prospective, pre-observation amendment to:

- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_01.md`.

It must also be read with:

- `docs/sprint3/ACC_PROTOCOL.md`;
- `docs/sprint3/FAITHFULNESS_PROTOCOL.md`;
- `docs/sprint3/ASQA_INTERNAL_PARTITION_PROTOCOL.md`.

Amendment 01 remains authoritative for candidate identities, model snapshots,
interfaces, decoding, parsing, and candidate-specific failure handling unless
Amendment 02 explicitly states otherwise.

The original protocol remains authoritative for sampling, human annotation,
hard gates, the winner hierarchy, and every rule not superseded here.

Amendment 02 supersedes only:

1. the previously vague low-agreement expansion clause; and
2. conflicting downstream decomposer-input wording in the ACC and
   faithfulness protocols.

It does not redesign any other decomposer methodology.

## 2. Original Bake-Off Sample Remains

Preserve the original canonical bake-off design:

```text
54 answers
=
3 datasets
x 3 primary LLM logical slots
x 2 context modes
x 3 answers per cell
```

The datasets are:

- PubMedQA;
- HotpotQA;
- ASQA.

The context modes are:

- `WITH_CONTEXT`;
- `WITHOUT_CONTEXT`.

Use the original frozen:

```text
seed = 20260822
```

and the existing per-cell SHA-256 deterministic ordering.

Preserve all original `DEVELOPMENT` and historical evidence-role restrictions.
For ASQA, preserve internal `DEVELOPMENT`-only use plus
`EVALUATOR_DEVELOPMENT_EXPOSED` bookkeeping.

Do not use `SELECTION` or protected-final evidence.

## 3. Initial Agreement Gate

Preserve the already-frozen agreement threshold:

```text
Cohen's kappa >= 0.70
```

for every critical annotation category already governed by the original
protocol. Do not create new critical categories.

After the two blinded annotators independently rate the original 54-answer
bake-off, apply the following rule.

### Every required critical-category kappa is at least 0.70

- no contingency expansion occurs;
- follow the original frozen adjudication procedure;
- the original 54-answer bake-off remains the final candidate-selection
  evidence;
- winner selection proceeds under the original hard gates and lexicographic
  hierarchy.

### Any required critical-category kappa is below 0.70

Activate exactly the one-time contingency defined below.

## 4. One-Time Contingency Expansion

Freeze:

```text
additional answers = exactly 18
```

Allocation is exactly one additional answer for each of the 18 original
cells:

```text
3 datasets
x 3 LLM logical slots
x 2 context modes
x 1 additional answer
= 18 additional answers
```

Therefore, after contingency activation:

```text
54 original answers
+ 18 additional answers
= 72 total answers
```

This is a project-scale methodological convention, not a power claim or a
universal threshold.

There may be no second expansion.

## 5. Deterministic Selection of the 18 Added Answers

For each original `dataset x LLM x context-mode` cell, use the same frozen
eligible-answer population and the same deterministic per-cell SHA-256
ordering from the original protocol.

The original sample consumed the first three eligible canonical `OK` answers
in that ordering. The contingency answer is the next unused eligible canonical
`OK` answer in that same frozen ordering.

Operationally, this is the fourth eligible canonical `OK` answer after
applying only the already-governed non-OK skip rules. If the fourth ordered
item is governed non-OK or ineligible according to the original protocol,
continue deterministically to the next eligible item using the same
pre-existing ordering.

Do not choose contingency examples based on:

- candidate decomposition content;
- annotator disagreement;
- candidate quality;
- hard-gate outcomes;
- model identity;
- retrieval performance;
- correctness;
- faithfulness;
- ACC;
- any `SELECTION` or protected outcome.

Freeze and hash the 18-answer contingency manifest before human inspection of
the added examples.

## 6. Guide-Clarification Rule

If the initial kappa gate fails, allow exactly one annotation-guide
clarification step.

The clarification may:

- clarify wording;
- add neutral examples;
- resolve interpretation ambiguity;
- explain boundaries of the existing categories.

It may not:

- add or remove scientific categories;
- alter hard-gate thresholds;
- alter winner thresholds;
- alter candidate identities;
- alter candidate interfaces;
- create candidate-specific rules;
- relax a gate because a candidate performed badly;
- change evidence roles.

The clarification must be candidate-blind in wording.

Freeze, version, and hash the revised guide before final rerating begins.
Preserve the original guide and original raw annotation records permanently.

## 7. Final Rerating After Contingency Activation

If the contingency activates, both annotators independently rerate the
complete:

```text
72-answer set
```

under the one revised frozen guide.

The complete 72 consists of:

- all original 54 answers;
- all 18 deterministic additional answers.

For every answer, both decomposer candidate outputs are evaluated. The same
two candidates therefore remain compared on the same answer set.

Use:

- blinded candidate identity;
- blinded LLM identity;
- blinded context-mode and retriever metadata as already governed;
- a new deterministic blinded presentation ordering.

The rerating is from scratch under the revised guide. Do not overwrite the
first-pass raw ratings.

Store the first-pass ratings separately as:

```text
INITIAL_AGREEMENT_TRIGGER_LABELS
```

The rerated labels become:

```text
FINAL_BAKEOFF_RAW_LABELS
```

for the contingency path.

## 8. Adjudication and Final Bake-Off Evidence

After final independent rerating of the 72-answer contingency set:

- calculate agreement again;
- preserve both independent raw label sets;
- perform the same frozen adjudication procedure;
- use adjudicated labels for candidate hard gates and winner statistics, as
  already specified by the original protocol.

If the contingency activated, the final candidate-selection evidence is the
complete 72-answer rerated and adjudicated set.

Thus the added 18 answers may affect the eventual candidate winner. This is
prospective and is not optional after contingency activation.

Do not mix original first-pass labels with final rerated labels for winner
statistics.

## 9. Terminal Agreement Rule

After the complete 72-answer final rerating, apply the following terminal rule.

### Every required critical-category kappa is at least 0.70

Proceed to the already-frozen hard gates and winner hierarchy.

### Any required critical-category kappa remains below 0.70

Stop with status:

```text
DECOMPOSER_SELECTION_BLOCKED_LOW_AGREEMENT
```

Do not:

- collect another sample;
- expand to 90, 108, or another size;
- refine the guide a second time;
- weaken the kappa threshold;
- change candidate models;
- retrain a candidate;
- change winner criteria;
- select the least-bad candidate.

Further action requires explicit methodology or supervisor review and a new
prospective amendment with exposure accounting.

## 10. No Candidate Adaptation

Under both the normal and contingency paths, do not:

- retrain Candidate A;
- retrain Candidate B;
- fine-tune either model;
- change prompts after output observation;
- replace a candidate;
- change model revision;
- change parsing rules;
- change decoding;
- change hard-gate thresholds;
- change lexicographic winner rules.

The contingency concerns reliability of the human measurement instrument, not
optimization of decomposer candidates.

## 11. Resolve the Downstream Interface Conflict

Freeze canonical downstream decomposer execution as candidate-conditional.

The selected winner must continue to use the exact candidate-specific
interface on which it was prospectively registered and evaluated in the
bake-off.

### If Candidate A / FENICE Wins

Candidate A is:

```text
Babelscape/t5-base-summarization-claim-extractor
```

It must receive only the exact frozen dataset-specific answer-content surface.
It must not receive the question.

This preserves Amendment 01's native answer-only contract. Do not construct a
new question-and-answer wrapper for FENICE after it wins.

### If Candidate B / Phi Wins

Candidate B is:

```text
microsoft/Phi-4-mini-instruct
```

It must receive:

1. the exact canonical question; and
2. the exact frozen dataset-specific answer-content surface;

using the exact Amendment-01 prompt and interface.

### Common Rule

The candidate's selection-time input contract and downstream canonical
execution input contract must be identical.

Do not adapt the winning decomposer interface after winner observation.

## 12. Supersession of ACC and Faithfulness Input Wording

This amendment prospectively supersedes only the conflicting
decomposer-input-interface wording in:

- `docs/sprint3/ACC_PROTOCOL.md`;
- `docs/sprint3/FAITHFULNESS_PROTOCOL.md`;

where those documents state or imply that every selected decomposer always
receives the question plus answer surface.

The canonical interpretation after Amendment 02 is:

> The selected decomposer receives the frozen dataset-specific answer-content
> surface plus only the additional input allowed by that selected candidate's
> prospectively registered interface.

Therefore:

```text
FENICE winner: answer surface only
Phi winner: question + answer surface
```

All other ACC and faithfulness semantics remain unchanged, including:

- dataset-specific answer surfaces;
- question identity and provenance;
- claim handling;
- zero-claim rules;
- NLI verification;
- thresholds;
- windows and pairs;
- failures;
- metrics;
- evidence roles.

Do not treat this amendment as permission to change any other ACC or
faithfulness rule.

## 13. Question Provenance Remains Stored

Even if FENICE wins and therefore does not receive the question as model
input:

- preserve the exact canonical question and its hash in evaluator provenance;
- preserve dataset and sample identity;
- preserve the answer-surface hash;
- record:

```text
question_to_decomposer = false
```

If Phi wins, record:

```text
question_to_decomposer = true
```

This makes the selected decomposer interface auditable.

## 14. Winner Rule Remains Unchanged

Preserve the existing hard gates exactly. Preserve the existing lexicographic
winner hierarchy exactly.

Do not add an interface-preference criterion. Do not favor FENICE because it
is answer-only. Do not favor Phi because it can consume the question.

Selection remains determined solely by the already-frozen gates and winner
rule.

## 15. Provenance

For the decomposer-selection artifact, record at minimum:

- original protocol commit and hash;
- Amendment 01 commit and hash;
- Amendment 02 commit and hash;
- original 54-answer manifest and hash;
- whether contingency activated;
- initial per-category kappa values;
- initial raw-label artifact hashes;
- revised-guide hash if contingency activated;
- added 18-answer manifest and hash if activated;
- deterministic reserve-selection rule and version;
- final 72-answer presentation-order hash if activated;
- final independent raw-label hashes;
- final per-category kappa values;
- adjudicated-label hash;
- candidate output hashes;
- hard-gate results;
- winner decision;
- selected candidate;
- selected candidate's canonical downstream interface;
- `question_to_decomposer` boolean;
- Git commit;
- run-registry identity.

## 16. No Output Observation

This amendment is frozen before decomposer candidate outputs are inspected.

No decomposer output, human annotation result, agreement statistic,
candidate-gate result, `SELECTION` result, or protected-final result was used
to choose:

- the 18-answer expansion size;
- one-per-cell allocation;
- next-unused deterministic reserve rule;
- one-expansion cap;
- full-72 rerating policy;
- terminal stop rule;
- candidate-conditional downstream interface.

## 17. After Creation

After creating
`docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_02.md`:

1. show the complete new-file diff;
2. run `git status --short`;
3. confirm no other file changed;
4. do not commit;
5. stop.
