# Sprint 3 Project Blueprint

## Status and authority

This document is the canonical current research blueprint for Sprint 3 of
*Context Matters: Evaluating Diversity-Aware Retrieval for RAG*.

Decision authority, from highest to lowest, is:

1. later explicit supervisor/professor instructions;
2. kickoff hard requirements;
3. frozen blueprint decisions;
4. implementation choices.

The decisions marked **FROZEN** below govern new Sprint 3 development,
selection, evaluation, and final reporting. Earlier Sprint 3 documents remain
useful for their compatible protocol detail and decision history. Where an
earlier document conflicts with this blueprint, that statement is
**STALE/SUPERSEDED**; historical Sprint 1 and Sprint 2 artifacts themselves
remain unchanged.

No protected-final data may be used to select a method, hyperparameter, prompt,
metric definition, threshold, top-k, candidate-pool size, or statistical
procedure. Missing or uncomputed metrics remain missing values, not numeric
zero.

## Project objective and research questions — FROZEN

The objective is to determine **when diversity-aware retrieval helps RAG and
when it hurts by introducing irrelevant evidence, noise, or hallucination**.
The project is not designed to prove that diversification always helps.

The main research question is:

> When does diversity-aware retrieval improve RAG, and when does it introduce
> noise or hallucination?

The analysis must address:

- retrieval with context versus generation without retrieved context;
- diversification versus relevance-only retrieval;
- the relevance/diversity trade-off;
- dependence of effects on dataset and LLM;
- effects on faithfulness and hallucination.

Claims must remain conditional on the evaluated information-need regime. Code
execution alone is not scientific evidence of improvement.

## Three-sprint, three-dataset research story — FROZEN

PubMedQA, HotpotQA, and ASQA together form the three-dataset research story
across all three sprints. Historical artifacts remain unchanged. Any later
canonical replication or completion work must be labeled explicitly as
replication or completion work and kept distinguishable from the historical
artifacts.

| Dataset | Research role | Evidence status |
| --- | --- | --- |
| PubMedQA | Focused/single-document control; historical and replication evidence | `HISTORICAL_OBSERVED` |
| HotpotQA | Multi-hop/complementary-evidence benchmark | Historical 500-query experiment preserved; untouched test questions may support protected-final evaluation |
| ASQA | Ambiguous/multi-aspect diversity benchmark | Train is the selection universe; dev is protected final |

### PubMedQA exposure and use — FROZEN

- Historical Sprint 1 evaluated all 1,000 PQA-L questions.
- PubMedQA therefore has status `HISTORICAL_OBSERVED`.
- It may be reused across Sprint 2, Sprint 3, and the final notebook as
  control/replication evidence.
- It must never be described as an untouched, held-out, or protected final set.
- Existing validated historical artifacts must remain distinguishable from any
  new replication run.

### HotpotQA historical evidence — FROZEN

- Historical Sprint 2 evaluated the fixed seed-42 sample of 500 BEIR-test
  queries against a 25,997-document pooled corpus.
- That experiment and its finalized artifacts must remain unchanged and must be
  reported as historical evidence.
- Historical settings such as top-k 5 and candidate pool 20 describe that
  experiment; they do not freeze the corresponding final Sprint 3 sensitivity
  decisions.

### HotpotQA Sprint 3 corpus and protected-final policy — FROZEN

- BEIR train has role `DEVELOPMENT`.
- BEIR dev, containing 5,447 queries, has role `SELECTION`.
- The historical seed-42 sample of 500 BEIR-test IDs has role
  `HISTORICAL_OBSERVED`.
- The remaining 6,905 BEIR-test IDs are candidate
  `PROJECT_PROTECTED_FINAL` questions, pending explicit verification in an
  immutable manifest.
- The preferred Sprint 3 corpus is the complete BEIR HotpotQA corpus of
  5,233,329 passages.
- Development, method-selection, and final query sets must retrieve from the
  same full corpus.
- The BEIR-test questions not included in the historical seed-42 sample are
  candidates for a protected-final question set.
- The historical 25,997-document pooled corpus must not be used for
  protected-final retrieval.
- If and only if a resource-feasibility pilot shows that full-corpus indexing is
  infeasible, the fallback is one large, frozen, shared BEIR corpus subsample.
  It must not consist of query-specific pools.

### ASQA split assignment — FROZEN

- ASQA train contains 4,353 questions and is the development and
  method-selection universe.
- ASQA dev contains 948 questions and is the protected final split, as directed
  by the supervisor.
- ASQA dev must not influence method family, hyperparameters, prompt,
  generation settings, metric definitions, statistical protocol, top-k,
  candidate-pool size, thresholds, or selection rules.
- Final results must not trigger retrospective tuning.

### ASQA corpus — FROZEN

- The preferred retrieval corpus is the standard DPR Wikipedia snapshot dated
  2018-12-20.
- It contains 21,015,324 non-overlapping passages, segmented into blocks of 100
  words.
- BM25, DPR, Contriever, and ColBERTv2 must receive the same exact passage IDs,
  boundaries, and text. Physical indexes and model-specific representations may
  differ; dataset membership and passage content may not.
- Query-specific ASQA gold, supplied-context, or wikipage pools are rejected for
  the main evidence.
- If and only if full-corpus indexing fails the resource-feasibility pilot, the
  fallback is one large, frozen, shared Wikipedia subsample used by all
  retrievers. It must not be query-specific or constructed by injecting
  protected-query gold evidence.

### ASQA aspect relevance and metrics — FROZEN

For ASQA, `J(d,i)` indicates whether passage `d` covers aspect `i`.

- The frozen methodological rule is deterministic, normalized, token-bounded
  matching of all official `short_answers` aliases for aspect `i` against the
  exact canonical passage body.
- The exact normalization implementation is `NOT YET FROZEN`.
- It must be developed and unit-tested only on ASQA train, then frozen before
  protected ASQA dev is opened.
- A passage may cover multiple aspects.
- S-recall@k and alpha-nDCG@k are required aspect-aware retrieval metrics.
- `c*` is required as a passage-level diversification-ceiling diagnostic.
- Embedding diversity is a manipulation check, not a primary result.
- The exact alpha value or range and the lead ASQA metric remain unresolved and
  are listed under `NOT YET FROZEN`.

The unresolved matcher implementation and metric parameters must be frozen
using only permitted development evidence before protected-final evaluation.

## Retrieval and diversification design — FROZEN

The retrievers are:

- BM25;
- DPR;
- Contriever;
- ColBERTv2.

The diversification conditions are:

- relevance-only baseline;
- MMR;
- K-Means clustering;
- Agglomerative clustering;
- DPP.

For every fixed dataset, query, and retriever, the same first-stage candidate
list must feed the relevance-only baseline, MMR, K-Means, Agglomerative
clustering, and DPP. Comparisons must not be confounded by independently
retrieved candidate lists.

Within a controlled dataset experiment, all four retrievers must index and
search the same underlying corpus/passages. Retriever-specific physical index
formats, tokenization, embeddings, and internal representations are permitted,
but corpus identity is not retriever-specific.

Retrieval is independent of the generator LLM: changing the LLM must not change
the query set, corpus, first-stage candidate list, or diversified retrieval
output.

## LLM evaluation design — FROZEN

The evaluated LLMs are:

- `llama-3.3-70b`;
- `gemma4-26b`;
- `ministral-3-14b`.

The primary LLM mode is direct/non-thinking. Each LLM is evaluated in both:

- `WITH_CONTEXT`: generation using the retrieved context for the applicable
  retriever/diversification condition;
- `WITHOUT_CONTEXT`: generation without retrieved context.

`WITHOUT_CONTEXT` is one condition per dataset/query/LLM. It does not multiply
by retriever or diversification method because no retrieval condition is
present. `WITH_CONTEXT` runs are paired with their corresponding retrieval
conditions.

The final generation prompt and decoding settings are not yet frozen.

## Experimental control structure — FROZEN

For a controlled comparison, preserve:

- the same dataset split or frozen query manifest;
- the same underlying corpus manifest;
- the same query text;
- the same retriever-specific first-stage candidate list across all
  diversification methods;
- the same final top-k definition once frozen;
- the same generator configuration within an LLM comparison;
- stable sample, passage, run, and configuration identifiers;
- explicit missing values and retained row-level errors.

Protected-final data must not be used for method or hyperparameter selection.
Any reduced corpus used for smoke testing, profiling, or debugging must be
labeled as development-only and must not be presented as equivalent to the
preferred full-corpus experiment.

## Decision and evidence standard — FROZEN

Every important scientific or experimental decision must record:

1. evidence considered;
2. explicit selection criteria;
3. result of applying those criteria;
4. reason for the selected option;
5. rejected alternatives and why they were rejected;
6. known limitations and affected claims.

Where applicable, the record must also identify the run, Git commit, dataset
revision, split or sample manifest, corpus manifest, seed, retriever,
diversification configuration, candidate-pool size, top-k, LLM, prompt version,
decoding configuration, metrics, timestamp, and environment. Finalized raw
historical results remain immutable; corrected or extended evidence must be a
new, separately identified run.

## Notebook plan — FROZEN

The analysis notebooks are:

1. `01_sprint1_analysis.ipynb`
2. `02_sprint2_analysis.ipynb`
3. `03_sprint3_analysis.ipynb`
4. `04_final_research_analysis.ipynb`

Notebooks must load frozen result artifacts. They must not contain the core
retrieval, diversification, matching, generation, or metric algorithms, and
must not contain manually typed result numbers. Derived tables, figures, and
statistics must be reproducible from the referenced frozen artifacts.

## Relationship to existing Sprint 3 documents

The following documents remain supporting protocols where compatible with this
blueprint:

- `RESEARCH_QUESTIONS.md` for claim boundaries and the earlier research-question
  framing;
- `DATASET_PROTOCOL.md` for provenance, sampling, integrity, error handling, and
  protected-data controls;
- `DATASET_DECISIONS.md` for the earlier ASQA dataset audit;
- `METHOD_SELECTION_PROTOCOL.md` for evidence-based selection discipline;
- `DECISION_LOG.md` for dated implementation and protocol decisions;
- `DPP_IMPLEMENTATION_AUDIT.md` for the historical DPP implementation audit.

Statements in those documents that conflict with a frozen decision here are
`STALE/SUPERSEDED`. In particular, old corpus counts, claims that the ASQA split
assignment remains open, and any interpretation of historical Sprint 2 top-k or
candidate-pool settings as final Sprint 3 settings are not current policy.

## NOT YET FROZEN

The following items are deliberately unresolved. No decision is implied by
their ordering or by historical settings:

- resource-feasibility outcome for full HotpotQA and ASQA indexing;
- exact ASQA matcher normalization implementation;
- exact ASQA alpha value or range and the lead metric;
- exact faithfulness metric;
- final generation prompt;
- decoding settings;
- exact statistical-analysis protocol;
- final candidate-pool and top-k sensitivity decisions.

These items must be frozen using permitted development evidence and the decision
standard above before protected-final evaluation. They must not be chosen or
revised after inspecting protected-final results.
