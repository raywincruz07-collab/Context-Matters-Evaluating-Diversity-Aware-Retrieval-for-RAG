# Sprint 3 Dataset Decisions

## ASQA

Dataset source:
- din0s/asqa

Available Hugging Face splits:
- train: 4,353 questions
- dev: 948 questions

Sprint 3 roles:
- train = development and method-selection data
- dev = protected final-evaluation data

The ASQA dev split must not be used to choose:
- method family,
- hyperparameters,
- top-k,
- candidate-pool size,
- thresholds,
- metric definitions,
- selection rules.

## ASQA Retrieval Corpus

Final ASQA retrieval will target the DPR Wikipedia passage corpus:

- Wikipedia snapshot: 2018-12-20
- 21,015,300 passages
- passages segmented into blocks of up to 100 words
- same shared corpus for BM25, DPR, Contriever, and ColBERTv2

Gold ASQA annotations, supplied contexts, and wikipage information are
reference/evaluation information and must not be used to construct an oracle
retrieval corpus.

## Development Runs

Small corpus subsets may be used only for:
- implementation testing,
- unit/integration tests,
- smoke runs,
- debugging,
- performance estimation.

Results from reduced development corpora must not be reported as final
open-domain ASQA results.

## ASQA Annotation Caveat

The string "No context provided" is a missing-context placeholder and must not
be treated as gold evidence.

Observed annotation audit:
- every question has multiple disambiguated QA pairs,
- only approximately 45% of QA/aspect pairs have supplied wikipage/context
  grounding,
- therefore supplied contexts are not complete document-level relevance
  judgments for all ASQA aspects.

Aspect-level evaluation must therefore be defined separately in the Sprint 3
metrics protocol.

## Status

These decisions establish the dataset/split/corpus foundation.

Detailed sampling, identifiers, corpus versioning, integrity checks, and
provenance requirements will be specified in DATASET_PROTOCOL.md.
