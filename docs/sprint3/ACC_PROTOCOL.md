# Sprint 3 Bidirectional Atomic-Content Coverage Protocol

## 1. Purpose

The canonical Sprint-3 output-change construct is **Bidirectional
Atomic-Content Coverage**, with scalar `ACC_bi`.

The recommended implementation and registry label is:

```text
answer_content_change_bidir_v1
```

The scientific construct is retrieval-induced change in **substantive asserted
content**.

ACC is not:

- lexical diversity;
- stylistic diversity;
- correctness;
- faithfulness;
- usefulness;
- ASQA aspect coverage;
- a generic semantic-distance score.

Higher ACC means more substantive asserted-content change. Higher ACC does not
mean better. Correctness, coverage, and faithfulness separately determine
whether the content change was useful or harmful.

## 2. Primary ACC Comparison Scope

Canonical ACC compares only:

```text
A_0 = relevance-only WITH_CONTEXT answer

versus

A_d = diversified WITH_CONTEXT answer
```

for the exact same:

- dataset;
- `sample_id`;
- base retriever;
- generator LLM;
- prompt version;
- decoding version;
- generation replica.

The diversification treatment `d` may be:

- MMR;
- KMeans;
- Agglomerative;
- deterministic DPP;

according to the applicable frozen experiment matrix.

Do not define canonical ACC across:

- different LLMs;
- different retrievers;
- different questions;
- `WITH_CONTEXT` versus `WITHOUT_CONTEXT`.

`WITHOUT_CONTEXT` is not part of the canonical ACC construct.

The NLI calibration sample may contain varied `DEVELOPMENT` answer types,
including context-mode variation where useful for instrument coverage, but
that does not broaden the reported ACC scientific comparison. This explicitly
resolves the previous repository ambiguity.

## 3. Controlled-Pair Principle

For every ACC pair, hold fixed:

- question text;
- base retriever;
- generator LLM;
- system prompt;
- user template;
- decoding;
- maximum output tokens;
- replica;
- all non-retrieval generation configuration.

Only the retrieved selected-context treatment may differ. ACC therefore
measures propagation from a retrieval-context intervention into expressed
answer content.

## 4. Content Representation

Apply the same frozen deterministic atomic-claim decomposer to both answer
surfaces.

For every eligible answer, the decomposer input is exactly:

1. the exact canonical dataset question text; and
2. the exact dataset-specific ACC answer-content surface defined in Section
   20.

Nothing else is supplied.

The question is non-evidential context used only to support self-contained
reference resolution under the frozen decomposer protocol.

The decomposer must never receive:

- retrieved passages;
- context text;
- document IDs;
- retriever scores;
- gold answers;
- qrels;
- ASQA aliases or aspects;
- correctness results;
- faithfulness results;
- the opposing answer;
- baseline or diversified method identity.

The answer-content surface passed to the decomposer is:

### PubMedQA

- the parsed Explanation only;
- do not pass the categorical Decision as decomposable ACC content.

### HotpotQA

- the canonical generated Answer and Explanation content surface.

### ASQA

- the canonical long-form Answer content.

This does not change the dataset-specific ACC rules in Section 20. It makes
the decomposer input contract explicit.

Question text itself must never be extracted as a factual claim unless that
fact is actually asserted by the answer-content surface. Reference resolution
may use the question only according to the frozen
`docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md` and its amendment.

Let:

```text
P_0 = {p_01, ..., p_0m}
P_d = {p_d1, ..., p_dn}
```

Do not:

- force equal claim counts;
- align claims before decomposition;

The decomposer is a measurement instrument. Its winner and provenance are
governed by `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md` and its
amendments, including
`docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_01.md`.

The eventual physical winner is an execution output and is not reopened here.

## 5. No One-to-One Claim Matching

This rule is load-bearing.

Do not:

- perform proposition-to-proposition one-to-one matching;
- perform bipartite matching;
- compute a shared matched-pair count `M`;
- perform group or subset matching;
- perform combinatorial many-to-many matching.

The retired one-to-one Dice-style ACC formulation is not canonical Sprint-3
methodology. The canonical method is claim-to-complete-opposing-answer
verification.

## 6. Forward Direction

For each baseline claim:

```text
p_0i in P_0
```

ask whether the complete diversified answer content `A_d` entails,
contradicts, both entails and contradicts, or neither entails nor contradicts
`p_0i`.

The NLI orientation is:

```text
premise    = complete opposing diversified answer content
hypothesis = baseline atomic claim
```

Do not replace the opposing answer with its decomposed claim list. Do not
search for one matching diversified claim.

## 7. Reverse Direction

For each diversified claim:

```text
p_dj in P_d
```

ask whether the complete relevance-only answer content `A_0` entails,
contradicts, both entails and contradicts, or neither entails nor contradicts
`p_dj`.

The NLI orientation is:

```text
premise    = complete opposing baseline answer content
hypothesis = diversified atomic claim
```

This reverse direction is mandatory. It distinguishes:

```text
baseline    = A
diversified = A + B
```

because preservation of baseline content alone is not enough to establish
overall content identity.

## 8. Verifier

Use the single shared verifier family already frozen in
`docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md`.

The logical family is:

```text
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

ACC uses its own calibrated:

- ACC entailment threshold;
- ACC contradiction threshold.

Faithfulness uses separate thresholds.

Do not create:

- model-specific ACC thresholds;
- dataset-specific ACC thresholds;
- retriever-specific thresholds;
- diversification-specific thresholds;
- window-specific thresholds.

The exact immutable physical snapshot and numeric thresholds are Stage-1/2
execution and calibration outputs governed by the NLI verifier calibration
protocol.

## 9. Claim Relation Classification

For each claim-to-opposing-answer check, after whole-answer or window
aggregation, let:

```text
p_entail
p_contradiction
```

be the operational probabilities. Use the frozen ACC thresholds:

```text
tau_acc_ent
tau_acc_con
```

Classify:

```text
ENTAILMENT:
p_entail >= tau_acc_ent
AND
p_contradiction < tau_acc_con

CONTRADICTION:
p_contradiction >= tau_acc_con
AND
p_entail < tau_acc_ent

INCONSISTENT:
p_entail >= tau_acc_ent
AND
p_contradiction >= tau_acc_con

NEUTRAL:
neither threshold is met
```

Do not infer contradiction from failure to entail. Do not force
`INCONSISTENT` into entailment, neutral, or contradiction.

For the ACC coverage scalar, only pure `ENTAILMENT` counts as covered.
`INCONSISTENT` does not count as covered and must be reported separately. This
prospectively closes the previously implicit conflict-to-scalar edge rule.

## 10. Directional Components

For `m > 0`, with baseline claims checked against `A_d`:

```text
Retain_0_to_d  = # ENTAILMENT    / m
Drop_0_to_d    = # NEUTRAL       / m
Reverse_0_to_d = # CONTRADICTION / m
Conflict_0_to_d = # INCONSISTENT / m
```

Therefore:

```text
Retain + Drop + Reverse + Conflict = 1
```

for successfully measured non-empty baseline claim sets.

For `n > 0`, with diversified claims checked against `A_0`:

```text
Inherited_d_from_0  = # ENTAILMENT    / n
Add_d_from_0        = # NEUTRAL       / n
Contradict_d_from_0 = # CONTRADICTION / n
Conflict_d_from_0   = # INCONSISTENT  / n
```

Therefore:

```text
Inherited + Add + Contradict + Conflict = 1
```

for successfully measured non-empty diversified claim sets.

Preserve both rates and raw claim counts. The required scientific
interpretation fields therefore include at least:

- Retained;
- Added;
- Dropped;
- Contradicted;
- explicit conflict or inconsistency counts.

Do not combine these into another bespoke usefulness score.

## 11. Symmetric Content Similarity

For `m > 0` and `n > 0`:

```text
R_retain    = Retain_0_to_d
R_inherited = Inherited_d_from_0
```

Define:

```text
S_content =
2 * R_retain * R_inherited
/
(R_retain + R_inherited)
```

when:

```text
R_retain + R_inherited > 0
```

If both are zero:

```text
S_content = 0
```

Then:

```text
ACC_bi = 1 - S_content
```

with:

```text
0 <= ACC_bi <= 1
```

Interpretation:

- `ACC_bi` near 0: most substantive asserted content survives in both
  directions;
- `ACC_bi` near 1: little substantive asserted content is mutually preserved.

## 12. Zero-Claim Edge Cases

A valid zero-claim result is categorically different from decomposer failure.

If:

```text
m = 0
and
n = 0
```

then freeze:

```text
S_content = 1
ACC_bi = 0
```

Reason: neither answer asserts measurable substantive factual content, so
there is no measured substantive-content change.

Emit case code:

```text
BOTH_EMPTY
```

Do not present this as scientifically equivalent to a rich answer pair with
complete bidirectional coverage.

If exactly one side contains zero claims:

```text
m = 0, n > 0
```

or:

```text
m > 0, n = 0
```

freeze:

```text
S_content = 0
ACC_bi = 1
```

Emit case code:

```text
ONE_SIDE_EMPTY
```

Do not run meaningless NLI calls against an empty premise merely to derive
this edge result. A decomposer exception or failure is not a valid zero-claim
answer.

## 13. ACC Case Code

Every ACC pair must carry one of:

```text
NONEMPTY_MEASURED
BOTH_EMPTY
ONE_SIDE_EMPTY
NA_FAILED
```

Additionally preserve all detailed underlying failure and status fields. This
prevents `ACC_bi = 0` from `BOTH_EMPTY` from being mistaken for `ACC_bi = 0`
from complete content preservation.

## 14. Generation-Status Policy

Eligible semantic ACC inputs are:

```text
OK
REFUSAL
```

`REFUSAL` is a semantic model outcome, not infrastructure missingness.

For `REFUSAL`:

- preserve and report the refusal status;
- treat the refusal as asserting zero factual claims for ACC;
- do not run the decomposer merely to manufacture claims from refusal wording;
- apply the zero-claim edge rules.

Thus, refusal versus a substantive answer normally gives `ONE_SIDE_EMPTY` and
`ACC_bi = 1`. Refusal versus refusal gives `BOTH_EMPTY` and `ACC_bi = 0`, while
both refusal flags remain visible and are never hidden by the scalar.

This policy is specific to ACC. Faithfulness handling of `REFUSAL` is governed
separately.

The following are not eligible complete answers for canonical ACC:

```text
PARSE_FAILURE
TRUNCATED
ERROR
```

If either pair member has one of those statuses:

```text
ACC_bi = NA
case_code = NA_FAILED
```

Preserve and report the original generation status. Do not compute ACC from
truncated partial output. Do not convert a generation failure into zero
claims.

## 15. Decomposer Failure Policy

For an eligible `OK` answer, run the frozen decomposer.

If decomposition succeeds and legitimately returns zero claims, that is a
valid zero-claim result.

If decomposition:

- throws an exception;
- violates its required schema;
- produces an instrument-level malformed output;
- fails provenance or load validation;

then record:

```text
DECOMPOSER_FAILED
```

and the ACC pair is:

```text
ACC_bi = NA
case_code = NA_FAILED
```

Do not substitute zero claims. Do not repair using another decomposer. Do not
silently retry with another scientific instrument.

## 16. Verifier Failure Policy

Any required claim verification that ends in:

- model inference exception;
- non-finite probabilities;
- tokenizer construction failure;
- provenance mismatch;
- malformed verifier call;
- `FAILED_OVERLENGTH`;
- another evaluator-capacity failure;

is `FAILED`.

For canonical ACC, if any required directional claim check fails, the entire
ACC pair is:

```text
ACC_bi = NA
case_code = NA_FAILED
```

Do not silently shrink the directional denominator. Do not map failed checks
to:

- `NEUTRAL`;
- `CONTRADICTION`;
- `ENTAILMENT`;
- zero.

This keeps denominators equal to the actual claim-set sizes whenever ACC is
reported.

## 17. Long-Premise Rule

By default, use the complete opposing answer content as the verifier premise.

Do not window merely because the answer is ASQA. Use windowing only when the
exact:

```text
premise + claim + verifier special tokens
```

exceeds verifier input capacity.

Token counting uses the verifier's own pinned tokenizer. The canonical maximum
verifier sequence budget is 512 tokens, including:

- premise;
- hypothesis or claim;
- required special tokens.

## 18. Sentence-Preserving ACC Windows

If the complete opposing answer cannot fit:

1. Split the premise into sentences using the frozen deterministic
   sentence-splitting implementation.
2. Preserve original sentence order.
3. Greedily accumulate complete consecutive sentences while the verifier
   premise, hypothesis, and special-token budget remains at most 512.
4. Begin the next window with exactly one sentence of overlap from the
   previous window.
5. Never use lexical similarity to choose windows.
6. Never use an LLM to choose windows.
7. Never use retrieval scores to choose windows.
8. Never reorder sentences.
9. Never truncate the middle of an answer to head or tail fragments.

The exact physical sentence splitter and version become evaluator-bundle
provenance before Stage 2 closes.

If one individual sentence cannot fit together with the claim and special
tokens:

```text
status = FAILED_OVERLENGTH
```

Do not truncate that sentence and pretend it was completely evaluated.

## 19. Window Aggregation

For a claim evaluated over windows `w_1 ... w_k`:

```text
p_entail        = maximum entailment probability over all valid windows
p_contradiction = maximum contradiction probability over all valid windows
```

Aggregate the two probabilities independently. Then apply the same
`tau_acc_ent` and `tau_acc_con` used for whole-answer calls.

Do not create window-specific thresholds. If maximum entailment and maximum
contradiction both pass:

```text
relation = INCONSISTENT
```

The windowing mechanism must pass the separate frozen 30-case `DEVELOPMENT`
integration validation in
`docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md` before canonical evaluator
use.

## 20. Dataset-Specific Content Surface

### PubMedQA

Primary `ACC_bi` is applied to the parsed Explanation or rationale content.

The categorical:

```text
Decision: yes|no|maybe
```

is not folded into the `ACC_bi` proposition scalar.

Instead, report the separate dataset-native descriptive diagnostic:

```text
verdict_flip =
1 if baseline Decision != diversified Decision
0 otherwise
```

Cross-tabulate verdict flips with canonical correctness so they may be
described as:

- corrective;
- harmful;
- correctness-neutral.

Do not interpret a flip as beneficial merely because content changed.

### HotpotQA

Apply `ACC_bi` to the canonical generated substantive answer text containing
the short Answer and Explanation produced under
`docs/sprint3/GENERATION_PROTOCOL.md`.

Official HotpotQA answer F1 and EM remain external correctness metrics. Do not
incorporate correctness into ACC.

### ASQA

Apply `ACC_bi` to the complete canonical long-form Answer content.

ASQA aspect coverage remains a separate quality and coverage construct. Do not
place S-recall, alpha-nDCG, answer-side alias coverage, or gold aspect
information inside ACC.

## 21. Opposing-Answer Premise

The verifier premise must preserve the complete canonical opposing answer
content surface described above whenever it fits.

Do not replace the premise with:

- extracted claims;
- a summary;
- embeddings;
- a gold or reference answer;
- retrieved passages;
- supporting facts;
- ASQA aliases;
- task labels.

The premise is the model output being compared.

## 22. Claim Text

The verifier hypothesis is the original natural-language atomic claim emitted
by the frozen decomposer.

Do not apply:

- ASQA alias normalization;
- stemming;
- lemmatization;
- semantic canonicalization;
- fuzzy rewrite;
- LLM rewrite.

Do not alter polarity, quantities, dates, entities, modality, causal
direction, or epistemic hedging.

## 23. Exact-Duplicate Policy

After a successful decomposition, exact duplicate claim strings within the
same answer are deterministically deduplicated before `m` or `n` claim counts
and before NLI verification.

Exact means byte or string equality of the emitted `claim_text` under the
decomposer's parsed canonical output representation.

Preserve the first occurrence and discard later exact duplicates.

Record:

- raw claim count before exact deduplication;
- deduplicated claim count;
- number of exact duplicates removed.

Do not:

- casefold claims for deduplication;
- normalize punctuation for deduplication;
- normalize whitespace beyond whatever is already inherent in the parsed
  decomposer output;
- perform semantic deduplication;
- use embeddings;
- use the NLI verifier to merge claims;
- use an LLM to merge claims.

After successful decomposition, do not apply an additional post-hoc trivial
claim filter. The frozen decomposer itself is responsible for omitting purely
stylistic, conversational, or non-substantive text according to its prompt and
contract.

If the decomposer emits a valid-schema but awkward or unusual factual claim,
retain it literally rather than creating a subjective runtime filtering step.

This closes duplicate and trivial-claim handling prospectively and prevents
denominator changes after results are observed.

## 24. What ACC Must Not Use

Do not use as canonical ACC:

- BLEU;
- ROUGE;
- Self-BLEU;
- Distinct-n;
- BERTScore;
- whole-answer embedding distance;
- retrieval-embedding distance;
- entity-count diversity;
- semantic entropy;
- generic LLM-as-judge diversity;
- claim-to-claim matching;
- ASQA aspect coverage.

These may not be added after outcomes are observed without a prospective
exploratory amendment and registry entry.

## 25. Interpretation

Keep the scientific chain:

```text
retrieval/context diversity
  ->
substantive answer-content change (ACC_bi)
  ->
quality consequences measured separately by:
  correctness
  coverage
  faithfulness
```

Examples:

- high ACC with improved quality: potentially useful content change;
- high ACC with worse quality: potentially harmful content change;
- low ACC with high or stable quality: answer robust to retrieval-context
  variation.

Do not call ACC a causal mediator. Any relationship among diversity, ACC, and
quality is described according to the later frozen statistical protocol.

## 26. Role in Configuration Selection

ACC is not itself a quality objective. Do not select a configuration merely
for maximizing ACC. A method changing answers more is not automatically
superior.

ACC is a mechanistic and descriptive endpoint explaining whether context
diversification propagated into output content. The final statistical and
contrast registry will govern its exact inferential analysis.

## 27. Paired-Missingness Policy

ACC analyses use only successfully measured answer pairs according to the
later frozen common paired-analysis and statistical protocol.

Never convert missing pairs to `ACC_bi = 0`.

Report:

- expected pair count;
- measured pair count;
- generation-failure count;
- decomposer-failure count;
- verifier or window-failure count;
- refusal count;
- `BOTH_EMPTY` count;
- `ONE_SIDE_EMPTY` count.

Differential missingness and failure rates remain visible by experimental
cell.

## 28. Provenance

Every canonical ACC artifact must bind or reference:

- dataset;
- evidence role;
- `sample_id`;
- retriever;
- baseline-condition identity;
- diversified-condition identity;
- LLM;
- generation protocol and hash;
- baseline generation-artifact hash;
- diversified generation-artifact hash;
- generation statuses;
- exact ACC content-surface hashes;
- decomposer logical ID;
- decomposer physical revision or snapshot manifest once selected;
- decomposer protocol and version;
- baseline claim list and hash;
- diversified claim list and hash;
- verifier logical ID;
- verifier physical snapshot once frozen;
- tokenizer identity;
- ACC entailment threshold;
- ACC contradiction threshold;
- whole-answer or window mode per claim;
- windowing implementation and version;
- every per-claim probability;
- every per-claim relation;
- directional counts and rates;
- conflict counts and rates;
- `ACC_bi`;
- ACC case code;
- all failure reasons;
- evaluator-bundle ID;
- Git commit;
- run-registry identity.

## 29. Relation to Other Protocols

This protocol must be read with:

- `docs/sprint3/GENERATION_PROTOCOL.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_01.md` and any later
  prospective decomposer amendment documents;
- `docs/sprint3/NLI_VERIFIER_CALIBRATION_PROTOCOL.md`;
- `docs/sprint3/ANSWER_CORRECTNESS_PROTOCOL.md`;
- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`.

Generation defines canonical output surfaces and statuses. The decomposer
protocol defines claim-extraction instrument selection. The NLI protocol
defines the model family, calibration, and integration gates. Correctness
remains scientifically separate. The forthcoming
`docs/sprint3/FAITHFULNESS_PROTOCOL.md` defines grounding separately.

## 30. Frozen Before Observation

This document records and consolidates the Sprint-3 ACC methodology before
Stage-3 `SELECTION` or protected-final outcomes are observed.

No `SELECTION` or protected-final outcome was used to choose:

- the ACC construct;
- the claim-to-opposing-answer architecture;
- the scalar formula;
- the window policy;
- zero-claim handling;
- failure semantics;
- dataset-specific content surfaces.

## 31. Implementation and Execution Items Not Solved Here

The following are downstream execution and provenance work rather than open
ACC methodology:

- decomposer bake-off execution and winner identity;
- immutable decomposer snapshot acquisition;
- immutable verifier snapshot acquisition;
- numeric threshold calibration;
- calibration human annotation;
- windowed-ACC 30-case validation;
- exact sentence-splitter physical pin;
- evaluator implementation;
- artifact-schema implementation;
- tests;
- metric-registry synchronization;
- evaluator-bundle materialization.

These items do not reopen the ACC construct unless a frozen validation gate
formally fails.
