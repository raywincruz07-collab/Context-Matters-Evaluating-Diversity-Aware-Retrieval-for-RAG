# PubMedQA Generation Infrastructure Amendment 01

Date: 2026-08-30

## Scope and incident record

This amendment governs one infrastructure-invalid PubMedQA generation block
from the campaign based on Git commit
`fffcd431d634a8406993b6fee22bd7aad34fab70`:

- block: `WITH_CONTEXT / bm25 / llama-3.3-70b`;
- run ID:
  `run-sprint1-pubmedqa-s1-with-context-bm25-none-llama-3-3-70b-ffb0e1228071b46cb867d649`;
- rows 0--339: 340 valid `OK` outputs;
- rows 340--999: 660 `ERROR` outputs, each recording three
  `MakiInfrastructureError` attempts.

Attempt 1 was interrupted by a terminal disconnect. During resume,
`MAKI_API_KEY` contained a carriage return and additional shell text. The
resulting malformed `Authorization` header caused the 660 infrastructure
errors. The run registry nevertheless reached `COMPLETE` because `ERROR` is a
terminal per-row observation status.

The original registry lifecycle and all original row artifacts remain
append-only evidence and must not be deleted, rewritten, relabelled, or
overwritten.

## Evidentiary disposition

The original block is excluded from the canonical usable generation campaign
because of the client-side malformed Authorization header. This is an
infrastructure/configuration defect, not a model-quality result. It provides no
evidence that `llama-3.3-70b` produced 660 answer-quality failures.

Blocks 1--3 are unaffected by this incident and remain governed by their
existing registry and artifact evidence.

## Governed full replacement

One governed replacement is authorized for the affected run. It must:

1. use a new run ID derived under the later amendment Git commit;
2. record the original run ID in `execution.parent_run_id`;
3. preserve exactly the original block's dataset, sample manifest, retrieval
   and selected-context lineage, logical and physical generator identity,
   prompt, decoding, seed, and all other scientific treatment fields;
4. rerun all 1,000 rows from scratch;
5. write only to the deterministic directory
   `results/sprint3/raw/pubmedqa/generation/replacements/<parent_run_id>/<amendment_git_commit>`;
6. leave the original Block 4 directory and registry records unchanged.

Replacement execution is permitted only after the governed registry confirms
that the named parent is a matching terminal PubMedQA `GENERATION` run with
one or more failed/error rows. A registered replacement must be resumed through
the ordinary governed resume mechanism rather than replaced again silently.

## Scientific invariants

This amendment changes credential preflight and failure-recovery provenance
only. It does not change the scientific generation protocol: prompts, decoding,
model bindings, datasets, retrieval inputs, selected contexts, seeds,
repeatability criteria, and treatment definitions remain frozen.
