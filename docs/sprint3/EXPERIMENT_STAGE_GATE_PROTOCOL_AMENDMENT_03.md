# Experiment Stage-Gate Protocol Amendment 03 — Post-Reserve Generator Fallback Order

## 1. Authority and Scope

This prospective amendment governs generator fallback after the already-frozen
reserve `qwen3.5-122b` failed its repeatability gate. It extends only the
generator-substitution procedure in
`EXPERIMENT_STAGE_GATE_PROTOCOL_AMENDMENT_02.md`.

It is adopted before testing any post-reserve candidate and while `SELECTION`
and `PROJECT_PROTECTED_FINAL` outcomes remain unopened. It does not change the
frozen prompt, decoding, retry, dataset, retrieval, evaluation, or statistical
methodology. The experiment remains a three-generator design.

## 2. Frozen Evidence Entering This Amendment

The original canonical generator set was `llama-3.3-70b`, `gemma4-26b`, and
`ministral-3-14b`; the original designated reserve was `qwen3.5-122b`.

The seeded gate v2 recorded:

- `llama-3.3-70b`: 20/20, **PASS**;
- `gemma4-26b`: 20/20, **PASS**;
- `ministral-3-14b`: 7/20, **FAIL**.

Amendment 02 therefore invoked `qwen3.5-122b` prospectively. Its final candidate
binding used physical model ID `qwen3.5-122b`, provider seed `20260823`, and
`chat_template_kwargs.enable_thinking=false`.

The immutable `repeatability_gate_v3.json` recorded:

- `llama-3.3-70b`: 20/20, **PASS**;
- `gemma4-26b`: 20/20, **PASS**;
- `qwen3.5-122b`: 8/20, **FAIL**;
- overall gate: **FAIL**.

An offline diagnostic classified 58/60 Qwen responses as `OK` and 2/60 as
`TRUNCATED`. Failed repeatability prompts generally preserved the parsed
decision while explanation wording varied. These are descriptive diagnostics
only. The formal reason Qwen is not admitted is repeatability failure, not
answer quality.

## 3. Frozen Post-Reserve Testing Order

Test the following untested Maki general-generator candidates strictly in this
order:

1. `minimax-m2.7`;
2. `qwen3.8-27b`;
3. `qwen3.6-36b`.

Stop immediately at the first candidate that satisfies every frozen admission
requirement in Section 4. Do not test the next candidate unless the preceding
candidate fails operability or repeatability under those requirements.

This serial order preserves meaningful generator heterogeneity while avoiding
post-hoc model shopping. MiniMax is tested first because it provides an
independent model family relative to the retained Llama and Gemma generators.
No quantitative architecture or parameter claim is made. `qwen3.8-27b` and
`qwen3.6-36b` are fallback candidates only, not additional canonical
generators.

## 4. Candidate Admission Requirements

For each candidate reached in the frozen order, require prospectively:

1. exact Maki physical model and runtime identity recorded;
2. model revision recorded if the provider exposes one, otherwise explicit
   `NOT_PROVIDED_BY_PROVIDER` provenance;
3. provider-supported direct/non-thinking configuration determined only on
   `DEVELOPMENT`, without guessing unsupported controls;
4. support for `seed=20260823` tested and recorded;
5. the exact same frozen prompt, temperature, token budget, retry behavior, and
   repeatability manifest;
6. a new immutable model-binding artifact;
7. a new immutable repeatability-gate artifact; and
8. a complete three-generator gate in which every model satisfies
   `>=19/20` before canonical generation is authorized.

A candidate that fails operability or repeatability is rejected for that reason
only. Do not compare candidate answer quality or choose using correctness,
faithfulness, retrieval results, generated-answer preference, downstream
metrics, or any other favorable output.

## 5. Canonical Roster and Stop Rule

The retained generators are `llama-3.3-70b` and `gemma4-26b`. The first fallback
candidate admitted under Section 4 occupies the third generator slot formerly
held by Ministral and provisionally assigned to Qwen. Frozen matrix/contrast
slot identity and separate physical-model provenance remain governed by the
existing protocols.

Testing stops after the first admission. Later candidates remain untested
fallbacks and do not enter the canonical roster. Until one replacement
candidate passes the complete three-model gate, canonical generation remains
blocked.

## 6. Evidence Preservation and Exposure Boundary

The following remain immutable evidence:

- `repeatability_gate_v1.json`;
- `repeatability_gate_v2.json`;
- `repeatability_gate_v3.json`;
- every existing model-binding version.

Do not overwrite, relabel, delete, or reuse their paths for post-reserve tests.
Every reached candidate receives new provenance-bound binding and gate
artifacts.

`SELECTION` and `PROJECT_PROTECTED_FINAL` outcomes must remain unopened during
candidate configuration, testing, and admission. No such outcome may influence
the order, controls, rejection, or stop decision.

## 7. Prospective Status

This amendment freezes the post-reserve order and rules before candidate
testing. It does not itself authorize or execute model calls. Amendments 01 and
02 and the base protocols remain authoritative outside this narrow fallback
procedure.
