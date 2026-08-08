# AGENTS.md — Context Matters: RAG Sprint 3

## Project purpose

This repository contains research code for the University of Mannheim project
"Context Matters: Evaluating Diversity-Aware Retrieval for RAG."

Sprint 3 must prioritize:
- scientific validity,
- reproducibility,
- traceability,
- minimal and reviewable code changes,
- preservation of verified Sprint 2 artifacts.

## Core rules

1. Never fabricate, alter, or manually improve experimental results.
2. Never modify finalized Sprint 2 raw results.
3. Never change datasets, dataset samples, seeds, retrievers, generator models,
   prompts, metrics, top-k, candidate-pool sizes, or diversification parameters
   unless explicitly requested.
4. Never tune methods using the final evaluation split.
5. Treat missing/uncomputed metrics as missing values, not numeric zero.
6. Preserve experiment provenance and deterministic settings wherever possible.
7. Inspect existing code and tests before editing.
8. Prefer the smallest correct change.
9. Do not refactor unrelated code while implementing a requested change.
10. Add or update tests when behavior changes.
11. Run relevant tests after implementation.
12. Do not delete files or rewrite Git history without explicit approval.
13. Do not commit secrets, API keys, tokens, credentials, model access keys,
    private datasets, or environment-specific credentials.
14. Do not run large or expensive experiments unless explicitly requested.
15. Local WSL development is CPU-focused. GPU-heavy execution will use RunPod.

## Research integrity

Every experiment should be reproducible from recorded configuration.

Where applicable, record:
- run ID,
- Git commit,
- dataset,
- split,
- sample IDs or sampling procedure,
- random seed,
- retriever,
- diversification method,
- method parameters,
- top-k,
- candidate-pool size,
- generator/model,
- generation parameters,
- prompt version,
- evaluation metrics,
- timestamp,
- environment/hardware information.

Raw finalized experiment outputs should be treated as immutable.
Derived summaries, tables, plots, and statistics must be reproducible from raw outputs.

## Sprint 3 organization

Research protocols:
- docs/sprint3/

Experiment configurations:
- configs/sprint3/

Outputs:
- results/sprint3/raw/
- results/sprint3/summary/
- results/sprint3/figures/
- results/sprint3/logs/

## Working procedure for Codex

For non-trivial tasks:

1. Inspect the relevant repository files.
2. Explain the proposed change before implementation when requested.
3. Identify assumptions and potential research-impacting consequences.
4. Make the smallest necessary implementation.
5. Run focused tests.
6. Run the broader test suite when appropriate.
7. Show the resulting diff or summarize exactly what changed.
8. Do not commit unless explicitly instructed.

## Scientific claims

Do not infer scientific conclusions merely because code executes successfully.

Distinguish clearly between:
- measured results,
- statistical evidence,
- implementation behavior,
- hypotheses,
- interpretations.

Do not describe a method as better, optimal, significant, robust, or superior
without evidence supporting that claim.
