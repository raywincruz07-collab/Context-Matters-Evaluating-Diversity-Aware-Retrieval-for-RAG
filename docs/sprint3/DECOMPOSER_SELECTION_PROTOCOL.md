# Atomic-Claim Decomposer Selection Protocol

## Purpose

This pre-registered protocol selects the frozen atomic-claim decomposer using
development evidence before protected-final evaluation. It selects only the
decomposer and must not be changed after candidate outputs are inspected.

## Sample

The target sample is 54 canonical `OK` generated answers:

```text
3 datasets
× 3 frozen generator LLMs
× 2 context modes
× 3 answers per cell
= 54 answers
```

The datasets are:

- PubMedQA;
- HotpotQA;
- ASQA.

The frozen generator LLMs are:

- `llama-3.3-70b`;
- `gemma4-26b`;
- `ministral-3-14b`.

The context modes are:

- `WITH_CONTEXT`;
- `WITHOUT_CONTEXT`.

The permitted development sources are:

- PubMedQA: historical PQA-L control/exposed data;
- HotpotQA: BEIR train/development only;
- ASQA: train4353 only.

Any ASQA IDs used in this protocol become
`EVALUATOR_DEVELOPMENT_EXPOSED` and must remain on the development side of any
later internal train partition.

Question selection must be deterministic and fixed before generation, using:

```text
seed = 20260822
```

For every dataset × generator LLM × context-mode cell, the reserve list is
the entire eligible development-ID universe for that dataset, placed in one
deterministic order. Define the ordering exactly as:

```text
priority(sample_id) =
SHA256(
"20260822|{dataset}|{generator_llm}|{context_mode}|{sample_id}"
)
```

All components use their canonical string identifiers. Sort eligible sample
IDs lexicographically by the lowercase hexadecimal SHA256 digest.

Process IDs sequentially in this fixed order and select the first 3 canonical
`OK` generations. A candidate ID may be skipped only because its upstream
generation status is not `OK`; record every skipped ID and its non-`OK`
generation status.

If the eligible development universe is exhausted before obtaining 3 `OK`
generations for a cell, **STOP** and report insufficient valid generation
inputs. Do not extend the universe, change the seed, change the ordering, or
replace IDs for decomposer-related reasons.

The entire deterministic order can be materialized and hash-persisted before
the first generation call, but generation need only proceed sequentially until
3 `OK` outputs are obtained.

`WITH_CONTEXT` examples use relevance-only baseline context only.
Diversification methods are not part of decomposer selection.

Only canonical `OK` generation outputs are decomposed. `REFUSAL`, `TRUNCATED`,
`PARSE_FAILURE`, and `ERROR` are not normal decomposition inputs. Replacement
is governed only by the frozen reserve-list procedure above and must never be
based on candidate performance.

## Candidates

### Candidate A

Candidate A is:

```text
SYX/mistral_based_claim_extractor
```

Its base model is:

```text
unsloth/mistral-7b-instruct-v0.2-bnb-4bit
```

Before the pilot:

- the exact immutable adapter revision must be resolved;
- the exact immutable base-model revision must be resolved;
- license compatibility must be verified;
- both the adapter and base model must load successfully in the pinned
  environment;
- physical snapshot/file provenance for both must be recorded.

If any admission gate fails, **STOP** before candidate outputs are inspected.
Do not silently substitute another claim extractor. Candidate A may use its
official released inference template.

### Candidate B

Candidate B remains `microsoft/Phi-4-mini-instruct`, using the project
atomic-claim extraction prompt. Before the pilot, its exact immutable revision
must be resolved, it must load successfully in the pinned environment, and its
physical snapshot/file provenance must be recorded.

Both candidates process the generated answer without retrieved evidence. The
retrieved context must **never** be passed to either decomposer.

## Decomposer input

The project decomposer receives:

```text
Question:
{question}

Answer — this is the ONLY source from which claims may be extracted:
{generated_answer}
```

The question may be used only to resolve references necessary to understand
assertions made in the answer.

No fact may be extracted from the question. No retrieved passage or retrieval
metadata may be supplied.

Claims must:

- contain one substantive factual proposition;
- preserve polarity and negation;
- preserve quantities, dates, and named entities;
- preserve hedging and modality;
- introduce no information absent from the answer;
- resolve references only when supported by explicit text in the question or
  answer;
- leave genuinely ambiguous references unresolved or flagged rather than
  inventing a resolution.

The preferred structured output records at least:

- `claim_text`;
- `source_sentence_index`.

## Human evaluation

Both candidates process the exact same answer set.

Two human annotators independently evaluate the candidate decompositions.

Annotators see:

- the question;
- the original generated answer;
- the extracted claims.

Annotators must be blinded to:

- decomposer identity;
- candidate A/B identity;
- generator LLM;
- context mode;
- retriever.

Candidate presentation order must be deterministically randomized.

The critical claim-level error categories are:

1. invented information;
2. polarity/negation alteration;
3. quantity/date/named-entity alteration;
4. hedging/modality alteration;
5. incorrect decontextualization/coreference.

The additional quality criteria are:

- atomicity;
- completeness of material factual assertions;
- self-containment/decontextualization quality;
- structured-output validity.

### Selection statistics and denominators

Final candidate-selection statistics use adjudicated human labels after the
independent annotations. Raw annotator labels and agreement statistics must be
preserved separately.

The critical error rates are:

```text
invented_content_rate =
claims containing invented information /
all valid extracted claims

polarity_negation_error_rate =
claims with polarity/negation alteration /
all valid extracted claims

quantity_date_entity_error_rate =
claims with quantity/date/named-entity alteration /
all valid extracted claims

hedging_modality_error_rate =
claims with hedging/modality alteration /
all valid extracted claims
```

Atomicity is:

```text
atomicity_rate =
valid extracted claims judged atomic /
all valid extracted claims
```

Completeness is evaluated at the **answer level** by identifying material
factual assertions in the source answer and measuring the proportion
represented by at least one extracted claim. Record both per-answer
completeness and the aggregate mean.

Decontextualization correctness is:

```text
decontextualization_correctness_rate =
correctly self-contained/resolved claims /
claims for which decontextualization or reference resolution was required
```

If the denominator is zero, the rate is `NA`, not `0` or `1`.

Instrument failure is:

```text
instrument_failure_rate =
decomposer runs failing to return valid structured output /
attempted answer decompositions
```

## Hard safety gates

A candidate is disqualified if any of the following holds:

```text
invented_content_rate > 0.02
polarity_negation_error_rate > 0.02
quantity_date_entity_error_rate > 0.02
hedging_modality_error_rate > 0.05
```

A candidate must also satisfy:

```text
atomicity_rate >= 0.90
```

These gates must not be weakened after candidate outputs are observed.

## Winner rule

Among candidates passing every hard safety gate:

a. Compare aggregate completeness. If the absolute difference is greater than
   or equal to 3 percentage points, select the higher-completeness candidate.

b. Otherwise compare `atomicity_rate`. If the absolute difference is greater
   than or equal to 3 percentage points, select the higher-atomicity candidate.

c. Otherwise compare `decontextualization_correctness_rate` only if it is
   defined for both candidates. If the absolute difference is greater than or
   equal to 3 percentage points, select the higher candidate. If the rate is
   `NA` for either candidate, skip directly to `instrument_failure_rate`.
   Never impute an `NA` decontextualization rate.

d. Otherwise select the candidate with the lower `instrument_failure_rate`.

e. If still tied, select the smaller/faster/simpler instrument.

If neither candidate passes all hard gates, **STOP**. Never relax gates after
observing results.

## Annotator agreement

Report inter-annotator agreement for the critical categories. The target is:

```text
Cohen's kappa >= 0.70
```

If agreement on a critical category is below `0.70`:

- adjudicate and refine the annotation guidance;
- do not alter the frozen selection thresholds;
- collect a small additional development-only annotation sample before
  selecting a winner.

## Scope

This protocol selects only the atomic-claim decomposer.

It does **not** select or tune:

- the NLI verifier;
- ACC thresholds;
- faithfulness thresholds;
- retrieval methods;
- diversification settings;
- final experiment configurations;
- protected-final results.

The primary verifier model remains conceptually selected as
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, with its exact
immutable revision and physical provenance to be frozen later at its
acquisition/load gate.

## FROZEN BEFORE OBSERVATION

The sample construction, candidate instruments, human evaluation criteria,
hard safety gates, winner rule, and annotator-agreement policy in this protocol
are fixed before candidate outputs are inspected. They must not be revised or
weakened in response to candidate performance.
