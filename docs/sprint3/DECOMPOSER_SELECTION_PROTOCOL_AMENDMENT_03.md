# Decomposer Selection Protocol Amendment 03

## 1. Authority and Scope

This is a prospective, pre-observation amendment to:

- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_01.md`;
- `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL_AMENDMENT_02.md`.

The original protocol and Amendments 01 and 02 remain authoritative for every
candidate, admission gate, interface, decoding rule, parser, sample, annotation
rule, agreement procedure, metric, hard gate, and winner criterion not
explicitly superseded here.

This amendment supersedes only wording equivalent to:

```text
select the smaller/faster/simpler instrument
```

when every earlier scientific winner criterion remains tied. It closes only
the terminal decomposer winner tie-break ambiguity.

## 2. Existing Winner Hierarchy Remains Primary

Apply the existing decomposer hard gates and winner hierarchy exactly as
already frozen.

Do not apply this amendment's terminal rule unless Candidate A and Candidate B
remain tied after every existing scientific winner criterion has been
exhausted.

In particular, this amendment does not insert parameter count before, replace,
or alter any existing comparison of:

- aggregate completeness;
- atomicity rate;
- decontextualization correctness when defined for both candidates;
- instrument failure rate;
- any existing hard safety gate.

## 3. Terminal Deterministic Tie-Break

Only if Candidate A and Candidate B remain tied after every existing
scientific winner criterion has been exhausted, apply this terminal rule:

1. Prefer the candidate with the lower fixed model parameter count.

2. Parameter-count identities are frozen as the published or configured model
   identities already selected before the bake-off.

3. Do not measure runtime, GPU memory, latency, throughput, or hardware cost
   from this bake-off to decide the winner.

4. Do not use observed downstream ACC, faithfulness, correctness, retrieval,
   SELECTION, or protected-final performance.

5. If parameter count is also exactly tied or cannot be authoritatively
   resolved from the frozen model identity, select:

   ```text
   Candidate A — FENICE
   Babelscape/t5-base-summarization-claim-extractor
   ```

   as the final deterministic lexical fallback.

6. The lexical fallback is not evidence that Candidate A is scientifically
   superior. It exists only to guarantee that the selection procedure always
   terminates without post-hoc discretion.

Parameter-count comparison must use the fixed candidate identities and their
authoritative published or configured architecture information. Do not derive
the comparison from bake-off execution measurements.

## 4. Rationale

Substantive scientific criteria remain primary. Terminal ties are expected to
be rare.

Parameter count is a pre-existing property of the prospectively frozen
instrument identity rather than an observed project-performance result. A
fixed lexical fallback removes residual discretion if parameter count cannot
resolve the terminal tie.

This amendment does not claim that smaller models are generally better atomic-
claim decomposers.

## 5. Preserved Methodology

This amendment does not change:

- Candidate A identity;
- Candidate B identity;
- the FENICE answer-only interface;
- the Phi question-plus-answer interface;
- the 54-case initial bake-off;
- the 18-case low-kappa expansion;
- the 72-case rerating procedure;
- human annotation rules;
- the Cohen's-kappa threshold or agreement gates;
- hard safety gates;
- the existing winner hierarchy before the terminal tie;
- candidate-specific decoding, parsing, failure, or capacity rules;
- ACC;
- faithfulness;
- NLI calibration;
- downstream evaluation.

No interface preference enters the terminal rule. Candidate A is not favored
because it is answer-only, and Candidate B is not favored because it receives
the question.

## 6. Provenance

The decomposer-selection decision artifact must record:

- this amendment's version, commit, and hash;
- whether the terminal tie-break was reached;
- the authoritative parameter-count source and identity for each candidate if
  parameter count was used;
- whether parameter count resolved the tie;
- whether the deterministic Candidate-A lexical fallback was invoked;
- the final selected candidate and reason code.

Do not record bake-off runtime or resource measurements as winner evidence.

## 7. Frozen Before Observation

This amendment is prospective and frozen before decomposer candidate results
are inspected.

No decomposer output, annotation result, agreement statistic, gate result,
downstream ACC, faithfulness, correctness, retrieval result, SELECTION result,
or protected-final result was used to choose:

- fixed parameter count as the terminal comparison;
- the parameter-count resolution rule;
- Candidate A as the final deterministic lexical fallback.
