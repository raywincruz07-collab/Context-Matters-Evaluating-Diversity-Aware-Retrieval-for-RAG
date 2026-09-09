# Context Matters — Sprint 3 Project Handoff

Date: 2026-09-09

## Working rule

Keep execution minimal.

- One immediate task at a time.
- No unnecessary pilots.
- No redesign of completed experiments.
- Reuse completed artifacts whenever possible.
- Do not rerun valid content outcomes such as TRUNCATED or PARSE_FAILURE.
- Do not cancel queued jobs without a concrete reason.
- Large result/cache artifacts stay outside Git.
- Git stores code, protocols, manifests, notebooks, and documentation.

---

# Research question

Is diversifying retrieved context helpful for LLM reasoning, or does it introduce noise and hallucinations?

---

# Experiment hierarchy

## Sprint 1
Relevance baseline only.

- BM25
- DPR
- Contriever
- ColBERTv2
- WITH_CONTEXT
- WITHOUT_CONTEXT
- 3 LLMs

No MMR, clustering, or DPP in Sprint 1.

## Sprint 2
Diversification:

- MMR
- KMeans
- Agglomerative clustering
- DPP

## Sprint 3
Final controlled matrix and analysis:

- correctness
- faithfulness
- coverage/diversity
- final comparisons

---

# LLMs

Physical maKI models:

1. llama-3.3-70b
2. gemma4-26b
3. qwen3.6-36b

The legacy logical registry name `ministral-3-14b` maps to physical
`qwen3.6-36b`.

Bindings:

`configs/sprint3/maki_model_bindings_v7.json`

---

# PubMedQA Sprint 1

Status: CLOSED.

Completed:

- WITHOUT_CONTEXT
- BM25 context
- DPR context
- Contriever context
- ColBERT context
- all 3 LLMs

One Qwen DPR TRUNCATED outcome is a valid frozen content outcome and
must not be rerun.

---

# HotpotQA canonical Sprint 1

Status: CLOSED.

Dataset:

- BEIR HotpotQA
- full corpus: 5,233,329 documents
- official test queries: 7,405
- canonical retrieval candidate pool: Top-20
- final context passed to maKI: exactly 5 passages

Retrievers:

- BM25
- DPR
- Contriever
- ColBERTv2

Generation completed for:

- WITHOUT_CONTEXT
- all four retrievers
- all three LLMs

Nominal matrix:

111,075 generations

Actual calls:

111,057

Difference:

18 Contriever missing cells from the frozen completed experiment.

Do not reopen the completed canonical HotpotQA Sprint-1 experiment.

---

# HotpotQA Top-100 extension

Purpose:

Top-100 is an additional retrieval-depth / sensitivity artifact.

It does NOT replace the canonical HotpotQA Top-20 experiment.

It does NOT mean 100 passages are sent to maKI.

Generation rule remains:

- retrieve/rank deeper
- maKI receives exactly 5 passages

For Sprint-1 relevance baseline, ranks 1-5 are the five context passages.

For canonical Sprint-2 diversification:

Top-20 -> diversification -> select 5 -> maKI

Andreea guidance allows Top-50 / Top-100 as sensitivity analysis.

## Top-100 jobs on bwUniCluster

BM25:
- job 6840388
- partition cpu

DPR:
- job 6840706
- partition gpu_h100
- 1 H100
- 24 CPUs
- 180 GB RAM
- 8 hour walltime

Contriever:
- job 6840712
- partition gpu_h100
- 1 H100
- 24 CPUs
- 180 GB RAM
- 8 hour walltime

ColBERT:
- job 6840782
- partition gpu_h100_il

At checkpoint time all four jobs are PENDING due to Priority.

Do not cancel or resubmit them simply because they are pending.

Top-100 output files are separate from Top-20:

- `candidates_top100.jsonl`
- `summary_top100.json`

Completed Top-20 artifacts remain protected.

---

# ASQA canonical corpus

Supervisor guidance:

Use the full DPR Wikipedia collection.

Do not subsample.

Canonical collection:

- 21,015,324 passages
- source `psgs_w100.tsv.gz`
- canonical generation passage surface is the exact DPR body
- no title in generation context

Source SHA-256:

`c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d`

ASQA official dev:

- 948 questions
- evidence role: PROJECT_PROTECTED_FINAL

Ordered-ID SHA-256:

`ca694f4e29ffd8b2c3330d51368a26ac9103b767609b4fd4a4ed587bad4fea30`

---

# ASQA DPR

Status: CLOSED.

Source:

ALCE published DPR Top-100.

File:

`data/asqa/alce_published/ALCE-data/asqa_eval_dpr_top100.json`

Rows:

948 x 100

SHA-256:

`221b4a7fc074346096cf6298319feb635256ec12c5cdf10aae528402ee39c252`

---

# ASQA BM25

Status: CLOSED.

Pyserini prebuilt `wikipedia-dpr` Lucene index.

Full collection:

21,015,324 passages.

Output:

`data/asqa/retrieval/bm25_pyserini_wikipedia_dpr_top100_v1.jsonl`

Rows:

948

Hits:

94,800

SHA-256:

`ec71626bfcffe6c921f6696a894cefea2729a17a4a5b6c082a5a7b21399bdbee`

---

# ASQA Contriever

Status: CLOSED.

Uses Meta published Wikipedia embeddings.

Top-100 output:

`data/asqa/retrieval/contriever_meta_published_top100_v1.jsonl`

Rows:

948

Hits:

94,800

SHA-256:

`5a9b90fbd5761d75837a9f0c96c13747d139edf253602d8ad805e1a1f97f4e57`

---

# ASQA generation package

Status: CLOSED.

Directory:

`data/asqa/generation_package_2026-09-09/`

Files:

## WITHOUT_CONTEXT

`without_context_queries.jsonl`

Rows: 948

SHA-256:

`c58819096a3196a94e01a530763f65dca3b668c1a7c36cbeabed912866175513`

## DPR

`dpr_with_context.jsonl`

Rows: 948

SHA-256:

`b09cc3507aa95bb0f2cd889d05c95a940e024cb80d87541979ce7fc7ac7f8927`

## BM25

`bm25_with_context.jsonl`

Rows: 948

SHA-256:

`187e2e5347626441488df81618720b015ac04993d27420f47fab4bec9972d723`

## Contriever

`contriever_with_context.jsonl`

Rows: 948

SHA-256:

`266c32bef94f00535506be4345dbc575ee5b6cec2db5b0c16c667a068d0f70dc`

---

# ASQA Sprint-1 maKI generation

Status for currently available retrievers: COMPLETE.

Output:

`data/asqa/generation_outputs_2026-09-09/sprint1`

Completed:

11,376 / 11,376

Statuses:

- OK: 11,334
- TRUNCATED: 39
- PARSE_FAILURE: 3
- ERROR: 0

These TRUNCATED and PARSE_FAILURE outcomes are valid frozen content
outcomes and must not be rerun.

Completed conditions:

- WITHOUT_CONTEXT x 3 LLMs
- DPR x 3 LLMs
- BM25 x 3 LLMs
- Contriever x 3 LLMs

Each WITH_CONTEXT ASQA generation uses exactly 5 passages.

ASQA max_tokens:

512

---

# ASQA ColBERT

Not yet closed.

Only ColBERT requires a new full 21M-passage index.

Previous full-index job failed because the default NCCL process-group
timeout was 10 minutes during a long clustering all-reduce.

Infrastructure-only fix:

- process-group timeout extended to 6 hours
- scientific configuration unchanged

Commit containing timeout fix:

`671e6a675fa8953c846b276a05967058c49f53cf`

Current retry:

- job 6839827
- keep queued unless there is a concrete reason to change strategy

Supervisor explicitly allows ASQA ColBERT to be dropped if it cannot be
completed in time.

Do not shrink the ASQA corpus instead.

---

# ASQA Sprint-1 notebook plan

Create the ASQA Sprint-1 notebook now using:

- WITHOUT_CONTEXT
- DPR
- BM25
- Contriever
- all 3 LLMs

Do not wait for ColBERT.

When ColBERT finishes:

- materialize Top-5 context
- run ColBERT x 3 LLMs
- append ColBERT to the same ASQA Sprint-1 notebook

Follow the same general reporting style already used for PubMedQA and
HotpotQA.

---

# Frozen ASQA generation protocol

System prompt:

Answer the question accurately using the requested output format. When context is provided, base your answer on that context. If you cannot answer reliably, state that briefly rather than inventing facts. Give only the requested output; do not provide step-by-step reasoning.

WITH_CONTEXT:

Question:
{question}

Context:
{context_block}

Output format:
Answer: <clear long-form answer resolving ambiguity and covering relevant distinct interpretations when applicable>

WITHOUT_CONTEXT:

Question:
{question}

Output format:
Answer: <clear long-form answer resolving ambiguity and covering relevant distinct interpretations when applicable>

Context contains exactly five bodies:

[Document 1]
<body_1>

...

[Document 5]
<body_5>

No titles, scores, IDs, qrels, or retriever identity are exposed to the LLM.

---

# Current Git state

Latest checkpoint commit at time of handoff:

`71b548c5a3d2648683da909e868d97356c3019f3`

Latest Top-100 commits:

- `1325d51947aa20074d398f0a7ee11cd90aa38926`
  HotpotQA Top-100 BM25 support

- `380fe876975fff4407e113a85e19930183df9d1b`
  HotpotQA Top-100 DPR/Contriever support

- `71b548c5a3d2648683da909e868d97356c3019f3`
  HotpotQA Top-100 ColBERT support

Pre-existing untracked directories:

- `artifacts/hotpotqa_generation_materialization/`
- `experiments/`

Do not delete or add them accidentally.

---

# Immediate continuation

1. Leave queued retrieval/index jobs alone.
2. Create ASQA Sprint-1 notebook using the three completed retrievers.
3. Add ColBERT later when available.
4. Do not overcomplicate the project.
