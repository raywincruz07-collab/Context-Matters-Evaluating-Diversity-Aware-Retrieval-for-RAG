# Decomposer Selection Protocol Amendment 01

## Amendment purpose and scope

This is a prospective, pre-observation amendment to the atomic-claim
decomposer selection protocol frozen at commit
`e033ba0f938aab21f565b15608185a243c1c522b`. The original
`docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md` remains unchanged. This
amendment records the admission failure of the original Candidate A, replaces
that candidate prospectively, and freezes the candidate-specific interfaces,
decoding, parsing, and failure handling stated below.

## Original Candidate A admission failure

The original Candidate A was:

```text
SYX/mistral_based_claim_extractor
```

Its audited provenance was:

```text
adapter revision: efa457363ba649914c782e56e784058fc7393301
base model: unsloth/mistral-7b-instruct-v0.2-bnb-4bit
base revision: 9cf6dddd08b225917c73e52aea042bdb40114905
```

The original Candidate A failed its preregistered admission gate before any
decomposer outputs were inspected. The adapter-weight-bearing repository has
no established usable license, and its model card reports that more
information is required for the license. The Apache-2.0 license of the base
model and the Apache-2.0 license of the VeriScore software do not automatically
establish a license for the separately released adapter weights.

This exclusion is an admission and licensing decision. It was not based on
decomposition quality or performance:

- no original Candidate A outputs were inspected;
- no Candidate B outputs were inspected;
- no project-answer decompositions were produced.

The prospective replacement recorded below therefore does not constitute
post-hoc performance selection.

## Replacement Candidate A

Candidate A is prospectively replaced by:

```text
Babelscape/t5-base-summarization-claim-extractor
```

Its frozen repository revision is:

```text
94775fb1c8dc2c3ef1bfec413f9f961e6ba5a1c8
```

The official publication is *FENICE: Factuality Evaluation of Summarization
based on Natural Language Inference and Claim Extraction*, Findings of ACL
2024.

The official FENICE code revision used to define the released inference
behavior is:

```text
9741ec41996f1bf75825d7cdf29e931a066ce4f0
```

The replacement instrument is frozen as follows:

- architecture: `T5ForConditionalGeneration`;
- approximately 222.9 million parameters (`222,903,552`);
- a standalone full model, not an adapter;
- model-repository license: CC BY-NC-SA 4.0;
- `trust_remote_code=False`;
- tokenizer: the released T5/SentencePiece tokenizer.

The model card currently declares `google/flan-t5-base` as base metadata. The
paper and saved configuration do not independently establish that training-base
provenance. It must therefore be recorded as declared metadata only, not as an
independently verified training-base identity.

The replacement is admitted because:

- the repository exists;
- its immutable revision is resolved;
- the model weights have an explicit license compatible with the stated local,
  non-commercial university research use;
- it is specifically trained for claim extraction;
- it requires no retrieved evidence;
- reproducible local inference is technically feasible.

Potential weaknesses arising from summarization-domain training, T5-base
capacity, biomedical hedging, QA coreference, and ASQA long-form coverage are
`QUALITY_RISK` considerations to be tested by the preregistered blinded
bake-off. They are not admission blockers.

## Instrument-comparison principle

The bake-off compares complete decomposition instruments rather than forcing
identical model prompts or interfaces. Candidate-specific released interfaces
are allowed when frozen prospectively.

The common invariant is absolute: **retrieved context, retrieved passages,
retrieval scores, retriever identity, diversifier identity, and gold evidence
must never be given to either decomposer.**

- Candidate A (FENICE) receives the **answer only**, because that is its
  released claim-extraction interface.
- Candidate B (Phi) receives the **question and answer** under the project's
  question-aware extraction interface.

This interface asymmetry must be disclosed in the methodology. It must not be
hidden or retroactively corrected after the bake-off. The human
decontextualization/coreference and completeness criteria are intended to
measure any practical effect of this interface difference.

## Candidate A scientific inference protocol

### Input

Candidate A receives the exact `generated_answer` only.

Do not prepend or supply:

- the question;
- a task instruction;
- a system prompt;
- retrieved context;
- metadata.

Use the pinned released model and tokenizer.

### Scientific decoding

Use the released deterministic FENICE generation behavior:

```text
do_sample = False
num_beams = 5
num_return_sequences = 1
min_length = 2
max_length = 250
```

These scientific settings must not be silently replaced with the greedy
settings used for a synthetic load test. All other behavior must come from the
pinned model and configuration plus the pinned compatible Transformers
environment, and must be recorded.

### Parsing

- Decode with special tokens removed.
- Apply the sentence-parsing behavior from the pinned FENICE code revision.
- Use spaCy `en_core_web_sm` version `3.7.0`.
- Treat the resulting sentence units as extracted claim units.
- Remove exact duplicate claim strings while preserving first occurrence.
- Do not perform semantic deduplication.
- Do not use an LLM or any other mechanism to repair or rewrite extracted
  claims.

### Overlength input policy

Provide the complete generated answer. No project-side head truncation, tail
truncation, lexical-window selection, or silent shortening is allowed. Use the
pinned released tokenizer and model behavior.

If the complete answer cannot be processed successfully under the pinned
instrument, record `INSTRUMENT_FAILURE` for that answer. Evaluator capacity
failure must never be converted into an empty claim list.

### Output-length policy

If generation terminates because the frozen maximum output length is reached
without a natural completed output, classify that answer as
`INSTRUMENT_TRUNCATED` and as an instrument failure for candidate-selection
statistics. Do not treat an incomplete decomposition as valid.

## Candidate B

Candidate B remains:

```text
microsoft/Phi-4-mini-instruct
```

Its frozen repository revision and license are:

```text
revision: cfbefacb99257ffa30c83adab238a50856ac3083
license: MIT
```

Use the repository chat template from the pinned snapshot. No retrieved
context may be supplied.

## Candidate B exact extraction prompt

The following meaning and wording are frozen as the project extraction
instruction.

### System message

```text
You are an atomic factual claim extractor. Extract only factual
propositions asserted in the Answer. Return only the requested JSON.
Do not use external knowledge and do not provide reasoning or
explanations.
```

### User message

```text
Question:
{question}

Answer — this is the ONLY source from which claims may be extracted:
{generated_answer}

Extract the substantive factual claims asserted in the Answer.

Rules:
1. Each claim must contain one independently checkable factual
   proposition.
2. Preserve polarity and negation exactly in meaning.
3. Preserve quantities, dates, units and named entities without
   rounding, normalization or substitution.
4. Preserve hedging, uncertainty and modality such as "may", "might",
   "likely", "suggests", "approximately", and "no significant
   difference".
5. Do not infer, add, repair or normalize information that the Answer
   does not assert.
6. The Question may be used only to resolve references necessary to
   understand the Answer. Never extract a factual claim solely from the
   Question.
7. Resolve a reference only when its resolution is supported by
   explicit text in the Question or Answer.
8. If a reference cannot be resolved without guessing, preserve the
   ambiguity rather than inventing a resolution.
9. Split separate factual assertions into separate claims when doing so
   does not alter their meaning.
10. Omit purely stylistic, conversational or discourse text that makes
    no substantive factual assertion.
11. Do not merge duplicate-looking claims semantically. Return what the
    Answer actually asserts.

Return exactly valid JSON in this schema:

{
  "claims": [
    {"claim_text": "..."}
  ]
}

If the Answer contains no substantive factual claim, return:

{"claims": []}

Return no markdown and no text outside the JSON object.
```

## Candidate B scientific decoding

Freeze:

```text
repository chat template: pinned snapshot
do_sample = False
num_beams = 1
num_return_sequences = 1
max_new_tokens = 1024
```

These settings do not imply bit-identical determinism across hardware or
software environments.

Use strict JSON parsing. Do not automatically repair invalid JSON with:

- another LLM;
- regex reconstruction;
- a retry for better content;
- hand editing.

Malformed structured output is an `INSTRUMENT_FAILURE`. If output reaches
`max_new_tokens` without completing valid JSON, classify it as
`INSTRUMENT_TRUNCATED` and as an instrument failure.

Candidate B must receive the complete frozen Question + Answer input. No silent
truncation is permitted. If the complete input cannot be processed under the
pinned snapshot and environment, classify that answer as
`INSTRUMENT_FAILURE`.

## Standardized bake-off representation

For blinded human evaluation, normalize both candidate outputs into this same
minimal internal representation:

```json
[
  {"claim_text": "..."}
]
```

This normalization may:

- parse the already-produced candidate output;
- remove exact duplicate claim strings as allowed by the original protocol.

It must not:

- rewrite claims;
- semantically deduplicate them;
- add missing claims;
- change wording;
- repair content.

`source_sentence_index` is not required for the blinded human selection
comparison because Candidate A does not natively expose reliable source
indices. This prospective amendment supersedes the original protocol's
non-binding preferred-structured-output suggestion only for the bake-off
representation.

## Unchanged original rules

All other rules in `docs/sprint3/DECOMPOSER_SELECTION_PROTOCOL.md` at commit
`e033ba0f938aab21f565b15608185a243c1c522b` remain unchanged, including:

- the 54-answer target;
- deterministic reserve ordering;
- development-only sampling;
- ASQA evaluator-development exposure handling;
- the same answer set for both candidates;
- two blinded annotators;
- hard safety gates;
- metric denominators;
- the winner rule;
- the Cohen's kappa policy;
- no protected-final use;
- no relaxation after observation.

## PRE-OBSERVATION AMENDMENT STATUS

At the time this amendment was written:

- no original Candidate A project decomposition had been produced;
- no replacement Candidate A project decomposition had been produced;
- no Candidate B project decomposition had been produced;
- no comparative decomposer performance had been observed;
- replacement was triggered exclusively by the preregistered licensing
  admission failure.

Therefore, this amendment is prospective and not performance-contingent.
