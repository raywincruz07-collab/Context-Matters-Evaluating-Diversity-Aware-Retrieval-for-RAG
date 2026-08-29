# Experiment Stage-Gate Protocol Amendment 02 — Prospective Reserve-LLM Substitution

## 1. Authority and Scope

This prospective amendment invokes the reserve-LLM procedure already frozen in
`EXPERIMENT_STAGE_GATE_PROTOCOL.md`, `GENERATION_PROTOCOL.md`, and
`FINAL_EXPERIMENT_MATRIX_PROTOCOL.md`. It is adopted while `SELECTION` and
`PROJECT_PROTECTED_FINAL` outcomes remain unopened.

It changes only the affected canonical generator binding. It does not change
datasets, prompts, temperature, maximum tokens, context conditions, retry
rules, evaluation, statistics, or the three-generator design. Historical
Sprint-1/Sprint-2 evidence is unaffected.

## 2. Development Evidence

The initial frozen repeatability gate (`repeatability_gate_v1.json`) recorded:

- `llama-3.3-70b`: 19/20, **PASS**;
- `gemma4-26b`: 7/20, **FAIL**;
- `ministral-3-14b`: operability failure because the provider rejected
  `chat_template_kwargs` for the Mistral tokenizer.

Provider diagnosis on `DEVELOPMENT` established that `ministral-3-14b`
succeeds when the unsupported chat-template control is removed. A fixed
provider seed of `20260823` was then introduced prospectively as a separate
runtime field. Llama and Gemma retained
`chat_template_kwargs.enable_thinking=false`; Ministral used no direct-mode
control because the provider does not support one.

The seeded frozen gate (`repeatability_gate_v2.json`) recorded:

- `llama-3.3-70b`: 20/20, **PASS**;
- `gemma4-26b`: 20/20, **PASS**;
- `ministral-3-14b`: 7/20, **FAIL**;
- overall gate: **FAIL**.

An offline diagnostic found that all 13 failed Ministral triples had repetition
1 differ while repetitions 2 and 3 were identical; no failure was caused by
truncation. Canonical PubMedQA classification across the 60 Ministral calls was
`OK=5` and `PARSE_FAILURE=55`. This parse behavior is secondary supporting
evidence only. It is not the substitution trigger and is not used as an
answer-quality judgment.

## 3. Prospective Substitution Decision

For canonical Stage-3 generation, replace `ministral-3-14b` with the already-
frozen reserve `qwen3.5-122b`.

The precise reason is unresolved repeatability failure under the frozen gate
after legitimate deterministic-control investigation. The substitution is
**not based on answer quality**, correctness, preference, or comparative model
performance.

The canonical generator roster remains exactly three models:

1. `llama-3.3-70b`;
2. `gemma4-26b`;
3. `qwen3.5-122b`.

For frozen contrast-ID and matrix-slot stability, `qwen3.5-122b` occupies the
former `ministral-3-14b` logical design slot. Every artifact must separately
record that Qwen is the physical model serving that slot. Qwen is not a fourth
generator.

## 4. Admission Requirements for Qwen

`qwen3.5-122b` must not enter canonical generation until all of the following
are complete:

1. its exact Maki runtime and model identity are provenance-recorded;
2. provider-supported direct/non-thinking behavior is determined on
   `DEVELOPMENT`, without guessing or sending unsupported controls;
3. provider support for `seed=20260823` is checked and recorded;
4. every affected repeatability/model-dependent development gate is rerun under
   the final runtime binding; and
5. the complete three-model repeatability gate passes under the final bindings.

Only a passing final three-model gate may authorize a canonical generation
block. The frozen gate size and pass criterion remain unchanged.

## 5. Evidence Preservation and Exposure Boundary

`repeatability_gate_v1.json` and `repeatability_gate_v2.json` remain immutable
development evidence. They must not be overwritten, relabeled, or deleted.
Any Qwen validation and replacement gate must use new, provenance-bound
artifacts.

`SELECTION` and `PROJECT_PROTECTED_FINAL` outcomes must remain unopened during
this amendment, Qwen operability work, and repeatability validation. No
selection or protected-final outcome may influence the binding, controls,
seed, gate interpretation, or substitution decision.

## 6. Prospective Status

This amendment records the reserve substitution before Qwen enters canonical
generation. It does not itself authorize or execute model calls. The base
protocol and all other amendments remain authoritative outside this narrowly
defined generator substitution.
