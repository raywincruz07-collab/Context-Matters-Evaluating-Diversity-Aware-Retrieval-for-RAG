# Experiment Stage-Gate Protocol Amendment 01

## 1. Authority and Scope

This is a prospective execution amendment and clarification to:

- `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL.md`.

It is adopted before the first new governed retrieval execution and before any
new governed retrieval record has been added to the production run registry.

This amendment closes only the previously unresolved numerical infrastructure-
retry ceiling for canonical retrieval runs. The base protocol remains
authoritative for every other stage-gate, retry, resume, failure, evidence, and
provenance rule.

## 2. Retrieval Infrastructure-Attempt Ceiling

For every governed run with:

```text
run_type = RETRIEVAL
```

freeze:

```text
maximum total infrastructure attempts = 3
```

The permitted lifecycle is:

```text
PLANNED
-> RUNNING attempt 1
-> optionally RUNNING attempt 2
-> optionally RUNNING attempt 3
-> COMPLETE or FAILED
```

Attempt 4 is prohibited.

A retrieval run may truthfully become `COMPLETE` or `FAILED` after its actual
attempt 1, 2, or 3. Do not create fake attempts to reach the ceiling. A terminal
snapshot must retain the attempt count of the immediately preceding `RUNNING`
snapshot. `COMPLETE` and `FAILED` remain terminal.

## 3. Infrastructure Recovery Only

Attempts 2 and 3 are permitted only after a technical or infrastructure
failure. The ceiling must not be used to authorize:

- tuning;
- changing retrieval parameters;
- rerunning because results appear poor or unexpected;
- selecting a preferable ranking;
- changing a retriever, model, corpus, or configuration.

Across attempts, preserve:

- the same scientific `run_id`;
- dataset, sample, corpus, retriever, model, and configuration identity;
- candidate pool and final top-k;
- the complete initial execution scope;
- all already-valid candidate artifacts;
- missing-only processing of unresolved IDs;
- attempt lineage;
- the existing environment and runtime compatibility requirements.

Hardware remains execution provenance under the existing registry contract.
`attempt_count` remains outside scientific run identity.

## 4. Scientific Treatment Is Unchanged

This is an engineering and recovery rule. It does not change retrieval scores,
rankings, candidate membership, candidate-pool size, top-k, missingness, or any
other scientific treatment.

The rule is prospective. It does not rewrite historical Sprint-1 or Sprint-2
methodology and does not fabricate retrospective registry records for
historical execution.

## 5. Implementation Requirement

Enforce the ceiling centrally in the governed run-registry lifecycle rather
than in a retriever-specific runner. Preserve the existing three-attempt
generation rule unchanged. Do not apply this numerical ceiling to other run
types unless a separate prospective authority explicitly does so.

## 6. Frozen Before Execution

This amendment is frozen before the first new governed retrieval execution,
before production registry mutation, and before new retrieval outcomes are
observed. No retrieval-quality or downstream result was used to choose this
infrastructure-recovery limit.
