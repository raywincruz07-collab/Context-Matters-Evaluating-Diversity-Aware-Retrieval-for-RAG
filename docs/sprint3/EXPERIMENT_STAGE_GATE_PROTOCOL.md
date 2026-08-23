# Sprint 3 Experiment Stage-Gate and Provenance Protocol

## 1. Purpose

This protocol freezes the evidence-opening, stage-transition, invalidation,
retry, provenance, and protected-final governance rules for Sprint 3.

This document does not claim that all hard dependencies are already satisfied.
Several Stage 2 blockers remain open, including the numerical
non-inferiority, tie, and failure-rate margins pending professor adjudication.

## 2. Stage model

Sprint 3 uses the following stages.

### Stage 0 — Methodology planning / freeze

No selection or protected-final outcomes are opened.

### Stage 1 — Development / instrument validation

Use only `DEVELOPMENT`, `HISTORICAL_OBSERVED`, or synthetic evidence.

### Stage 2 — Pre-selection hard gate

All applicable scientific, evaluator, statistical, data, and provenance
blockers must close before selection outcomes are opened.

### Stage 3 — Selection

Run one bounded HotpotQA/ASQA selection round. Only the global KMeans `k` and
global Agglomerative `k` may be selected.

### Stage 4 — Pre-protected-final hard gate

Lock selection decisions, the final matrix, protected manifests, models,
evaluators, corpora, statistics, and execution provenance.

### Stage 5 — `PROJECT_PROTECTED_FINAL` execution

Execute the frozen matrix without tuning.

### Stage 6 — Final analysis

Run predefined analyses and clearly separate any later exploratory analyses.

## 3. Evidence roles

### PubMedQA

PubMedQA is `HISTORICAL_OBSERVED`: exposed control and replication evidence.
It must never be relabelled as untouched protected-final evidence.

### HotpotQA

- `DEVELOPMENT`: BEIR train.
- `SELECTION`: the frozen non-protected selection role.
- `PROJECT_PROTECTED_FINAL`: the frozen deterministic `N=500` sample from the
  unexposed BEIR-test complement.

### ASQA

- `DEVELOPMENT`: 3,482 official-train examples.
- `SELECTION`: 871 official-train examples.
- `PROJECT_PROTECTED_FINAL`: all 948 official dev examples.

Historically exposed evidence must not be relabelled as protected final.

## 4. Stage 1 development rules

Stage 1 permits:

- implementation and debugging;
- unit, synthetic, and smoke tests;
- resource pilots;
- the decomposer bake-off;
- NLI calibration;
- matcher validation;
- human evaluator validation;
- candidate-pool sensitivity;
- stochastic DPP sensitivity;
- development retrieval and error analysis;
- technical operability and repeatability gates.

Any choice affecting scientific outputs must be frozen before Stage 3.

### Scientific-methodology change

A scientific-methodology change is any change affecting:

- dataset, sample, or corpus membership;
- evidence roles;
- retriever semantics;
- diversification semantics;
- candidate pool or top-k;
- prompt or context rendering;
- generator model, configuration, or decoding;
- retry or content policy;
- evaluator, decomposer, or verifier;
- thresholds;
- metrics;
- exclusion or missingness policy;
- a statistical or selection decision rule.

Such a change requires a prospective protocol or amendment.

### Neutral implementation fix

A neutral implementation fix preserves the frozen scientific inputs and
outputs by contract. Examples include:

- fail-fast validation;
- atomic artifact writing;
- provenance logging;
- exact resume support;
- scientifically equivalent scalability engineering.

If a change can alter scientific outputs, it must not be called neutral
without explicit equivalence evidence.

## 5. Pre-selection hard gate

Selection must not open unless all applicable blockers below are closed.

### Data

- immutable source revisions;
- `DEVELOPMENT` and `SELECTION` manifests;
- exposure records;
- population and disjointness validation;
- canonical corpus manifests.

### Retrieval

- the four retriever revisions and configurations;
- the corpus decision;
- physical index identities;
- `candidate_pool=20`;
- `final_top_k=5`;
- the shared candidate-artifact contract;
- frozen failure and resume behavior.

### Diversification

- the frozen candidate grid;
- tested MMR, clustering, and DPP implementations;
- deterministic failure and tie behavior;
- stochastic fixed-k DPP as a development sensitivity only.

### Generation

- the three canonical model IDs and revisions;
- the exact frozen prompt and version;
- the context renderer;
- decoding settings;
- the `WITHOUT_CONTEXT` identity contract;
- the status and retry contract;
- a canonical generation artifact and provenance schema.

### Evaluation

- correctness metrics;
- the ASQA matcher implementation;
- the decomposer winner;
- decomposer physical provenance;
- the verifier snapshot;
- verifier thresholds;
- the faithfulness implementation;
- the ACC implementation;
- the evaluator-bundle identity and hash.

### Statistics

- the exact selection objective;
- professor-adjudicated non-inferiority margins;
- the tie and practical-equivalence rule;
- the failure-rate rule;
- the bootstrap procedure;
- the missingness policy;
- the multiplicity and comparison-family policy.

### Provenance

- a clean committed scientific state;
- environment locks;
- a run registry;
- output schemas;
- expected matrix and row counts;
- isolated output directories;
- exclusion of secrets.

Unresolved Stage 2 blockers must be marked explicitly rather than treated as
complete.

## 6. Selection execution

During Stage 3:

- use only HotpotQA `SELECTION` and ASQA `SELECTION` evidence;
- use PubMedQA only as descriptive or control evidence;
- evaluate all predeclared clustering candidates on identical manifests;
- select KMeans `k` only from `{2, 3, 5}`;
- select Agglomerative `k` only from `{3, 5}`;
- introduce no new candidates after results are observed;
- introduce no new thresholds, metrics, or exclusion rules;
- perform no MMR, DPP, retriever, LLM, candidate-pool, or top-k selection;
- never treat poor performance as authorization to redesign the method.

Infrastructure-interrupted work may resume only under identical scientific
identity. Content or model outcome failures are outcomes, not reasons for
replacement.

## 7. Selection invalidation / restart

### Infrastructure or transport failure

- retain valid completed artifacts;
- resume or retry only incomplete work;
- preserve the same scientific identity;
- record attempt lineage.

### Deterministic implementation bug

- stop;
- mark affected outputs invalid;
- validate the fix on `DEVELOPMENT`;
- create a new clean commit;
- restart all affected paired comparisons;
- if shared upstream logic is affected, invalidate the entire affected
  selection scope.

### Scientific design defect

- stop;
- record all outcomes already observed;
- obtain a prospective amendment and supervisor adjudication;
- treat the selection evidence as exposed and never later describe it as
  untouched.

### Poor observed performance

- do not invalidate the run;
- apply the frozen rule;
- accept a null or unfavorable result.

Pre-fix and post-fix outputs must never be pooled.

## 8. Selection output lock

After selection, preserve immutably:

- raw selection artifacts;
- a file inventory and SHA-256 hashes;
- run and attempt lineage;
- manifests;
- corpus, candidate, and configuration identities;
- evaluator, environment, and Git identities;
- expected, present, successful, and failed counts;
- paired-intersection membership;
- exact statistical inputs, seeds, and outputs;
- a machine-readable winner and rejection record;
- a human-readable application of the selection rule;
- rejected alternatives;
- limitations and failure accounting;
- the closing timestamp and Git commit.

The selected KMeans and Agglomerative values must be frozen before Stage 4.

## 9. Protected-final manifest policy

Materializing protected membership or identity is not itself equivalent to
observing protected outcomes. Preferred materialization timing is Stage 4,
after selection is locked.

For HotpotQA:

- apply the already-frozen deterministic `N=500` ID-only rule;
- bind question text only after ID selection;
- use no answers, qrels, or results for membership.

For ASQA, official dev948 remains `PROJECT_PROTECTED_FINAL`.

If technical dataset acquisition requires protected-universe ID hashing before
Stage 4, IDs and hashes may be recorded for integrity only. This does not
permit:

- tuning;
- result inspection;
- evaluator development;
- method selection;
- threshold selection.

Record the identity-materialization timestamp separately from the first
protected content or outcome-opening timestamp.

## 10. Pre-protected-final hard gate

Before the first protected retrieval or generation, require:

- locked Stage 3 selection outputs;
- committed selected global clustering `k` values;
- a clean Git scientific commit;
- immutable protected query manifests;
- immutable logical corpus manifests;
- the HotpotQA full-corpus or fallback resource decision;
- physical index fingerprints;
- pool20 and top5;
- the full MMR curve;
- selected clustering configurations;
- the `dpp_map` configuration;
- generator IDs and revisions;
- prompt, context-renderer, and decoding hashes;
- decomposer, verifier, and evaluator identities;
- evaluator thresholds;
- the final statistical and contrast registry;
- environment, runtime, and hardware provenance;
- the exact final run matrix;
- expected row counts;
- retry and failure rules;
- successful development validation gates;
- an isolated protected-final output namespace;
- an atomic, conflict-safe writer;
- an output-inventory mechanism.

A clean Git status alone is insufficient. Ignored or runtime artifacts still
require inventory and provenance where scientifically relevant.

## 11. Protected-final execution

Once Stage 5 begins, prohibit:

- tuning;
- changing prompts;
- changing selected `k`;
- changing evaluator thresholds;
- changing candidate pool or top-k;
- changing metrics;
- dropping poorly performing conditions;
- swapping an LLM based on quality;
- result-dependent retries;
- repairing successful malformed outputs.

Generate `WITHOUT_CONTEXT` once per dataset, sample, and LLM, and reuse it.

Only exact resume from frozen artifacts and documented infrastructure retry
under identical identity are allowed. A valid scientific outcome must not be
rerun merely because it is unexpected or poor.

## 12. Protected-final invalidation

A transient infrastructure interruption permits safe exact resume; completed
valid artifacts remain authoritative.

The following are valid recorded outcomes and do not invalidate the run:

- `REFUSAL`;
- `TRUNCATED`;
- `PARSE_FAILURE`;
- `SHORT_CANDIDATE_LIST` or retrieval failure;
- evaluator `FAILED` or NA.

A scientific-identity mismatch, corrupt upstream artifact, or shared
implementation bug requires:

- stopping execution;
- quarantining affected outputs;
- recording observed exposure;
- preserving the original run and hashes;
- never overwriting or mixing results.

A corrected execution receives:

- a new `run_id`;
- a new commit, configuration, and environment identity;
- an explicit deviation or amendment record.

After protected outcomes are observed, a scientifically changed rerun must not
silently replace the first canonical run.

## 13. Reserve LLM

The primary LLMs are:

- `llama-3.3-70b`;
- `gemma4-26b`;
- `ministral-3-14b`.

The reserve LLM is:

- `qwen3.5-122b`.

Reserve substitution is allowed only before Stage 2 closes and only for a
predeclared operability, repeatability, or unavailability failure. It must
never occur because of observed answer quality.

Substitution requires:

- a prospective amendment;
- immutable model provenance;
- rerunning affected model-dependent development gates.

After selection outcomes open, canonical model substitution is prohibited.

## 14. Failure / retry governance

### Generation

- `REFUSAL`: record the outcome; no content retry.
- `PARSE_FAILURE`: a successful malformed response; no content retry.
- `TRUNCATED`: a length or finish-limit outcome; no content retry.
- `ERROR`: only a transport or infrastructure error may retry.
- Maximum total infrastructure attempts: 3.

### Retrieval

Fewer than 20 unique, valid, finite-score documents is
`SHORT_CANDIDATE_LIST`, a retrieval-production failure. Do not supplement,
duplicate, inject gold, or replace the query.

### Evaluator

A technical evaluator failure is `FAILED` or NA. Never convert it to zero,
neutral, or unsupported.

Every retry must retain identical scientific input and configuration identity
and must record attempt lineage.

## 15. Confirmatory vs exploratory

Before Stage 5, freeze the confirmatory:

- research questions;
- lead metrics;
- baseline contrasts;
- MMR curve;
- selected clustering treatments;
- DPP treatment;
- `WITH_CONTEXT` versus `WITHOUT_CONTEXT` comparisons;
- correctness;
- faithfulness;
- ACC;
- predefined heterogeneity summaries;
- confidence-interval and statistical procedures;
- multiplicity families.

After protected-final execution, new subgroup cuts, correlations, comparison
definitions, or hypotheses are exploratory. Exploratory analyses:

- must be labelled;
- may aid interpretation and future work;
- cannot change selection;
- cannot change thresholds;
- cannot be retroactively promoted to confirmatory evidence.

A new visualization of an already-predefined analysis is permitted as a
derived output.

## 16. Run registry minimum schema

Freeze a minimal run-registry architecture containing:

- `schema_version`;
- `run_id`;
- `stage`;
- `run_type`;
- `evidence_role`;
- `dataset`;
- `split`;
- protocol and configuration bundle hash;
- sample-manifest hash;
- corpus-manifest hash;
- Git commit;
- `worktree_clean`;
- retriever, index, and candidate identity when applicable;
- diversification method and configuration;
- upstream artifact identity;
- `context_mode`;
- generator ID and revision;
- prompt hash;
- decoding hash;
- evaluator-bundle and metric hash;
- environment and runtime hash;
- hardware summary;
- `started_at`;
- `completed_at`;
- status;
- controlled failure reason;
- `attempt_count`;
- `parent_run_id` or `resume_of`;
- expected, completed, successful, and failed row counts;
- output directory;
- output-inventory hash;
- raw-artifact hash.

Do not store secrets, tokens, endpoint credentials, or private environment
values. The implementation may reference frozen bundles rather than duplicate
large metadata in every row.

## 17. Notebook analysis contract

The canonical notebooks are:

1. `01_sprint1_analysis.ipynb`
2. `02_sprint2_analysis.ipynb`
3. `03_sprint3_analysis.ipynb`
4. `04_final_research_analysis.ipynb`

They must:

- consume frozen artifacts only;
- verify input and artifact hashes;
- contain no retrieval logic;
- contain no generation logic;
- contain no decomposer or verifier logic;
- contain no core metric implementation;
- contain no manually typed result values;
- treat raw artifacts as read-only;
- preserve missingness and paired intersections;
- produce reproducible tables, figures, and statistical summaries;
- record the input bundle, Git commit, parameters, and environment.

## 18. Protocol-amendment rule

Any scientific-identity change requires:

1. a prospective amendment before affected outcomes open;
2. the reason, evidence, and authority;
3. alternatives considered;
4. affected stages, runs, manifests, and configurations;
5. an explicit declaration of outcomes already observed;
6. exposure accounting;
7. scientific consequences;
8. a clean amendment commit before new execution.

After selection opens, poor performance cannot justify an amendment.

For a neutral bug, invalidate affected outputs, validate the fix on
`DEVELOPMENT`, commit the fix, and restart the governed scope.

After protected-final outcomes open, a scientifically changed execution is a
separately labelled protocol-deviation or corrected replication and must
preserve the first raw evidence.

## 19. Current open hard dependencies

The following remain unresolved and are not resolved by this document:

- professor-adjudicated numerical non-inferiority margins;
- the tie and practical-equivalence margin;
- the failure-rate margin;
- the complete final statistical protocol;
- materialized `DEVELOPMENT` and `SELECTION` manifests;
- the HotpotQA full-corpus resource outcome;
- the HotpotQA title, body, and `retrieval_content` implementation contract;
- ASQA physical source and corpus acquisition;
- HotpotQA and ASQA candidate producers;
- the diversified-output artifact schema;
- a committed canonical generation runner and provenance bundle;
- the decomposer bake-off and winner;
- NLI physical snapshot, load provenance, and calibration;
- evaluator-bundle implementation;
- the run registry and output inventory;
- final analysis notebooks;
- protected-final manifests and indexes.

Where the repository blueprint still contains stale unresolved notes for
methodology constructs that were subsequently frozen, those notes are
documentation and provenance synchronization issues. They are not permission
to reopen the methodology.

## 20. Frozen before outcome opening

This document freezes the stage architecture and evidence-opening rules
prospectively.

It does not assert that Stage 2 is currently ready to open.

No selection or protected-final outcomes are observed while creating this
document.
