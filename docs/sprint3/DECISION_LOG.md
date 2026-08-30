# Sprint 3 Decision Log

## 2026-08-30 — MiniMax fallback rejection for frozen-protocol operability

- **Decision:** Candidate physical model `minimax-m2.7` occupied the frozen
  logical third-generator slot `ministral-3-14b`. It is rejected from admission
  for **OPERABILITY FAILURE UNDER THE FROZEN GENERATION PROTOCOL.** This is not
  an answer-quality rejection, not a clean repeatability failure, and not a
  claim that MiniMax is generally unusable.
- **Binding and direct-mode evidence:** The binding is
  `configs/sprint3/maki_model_bindings_v5.json`, with seed `20260823`, direct-mode
  status `NOT_SUPPORTED_BY_PROVIDER`, and an empty direct-mode control (`{}`).
  Exact maKI-served checkpoint evidence showed no provider-supported
  `enable_thinking`/non-thinking switch, and the generation template starts
  generation in `<think>`.
- **Immutable gate evidence:**
  `artifacts/generation_repeatability/repeatability_gate_v4.json`, gate ID
  `generation-repeatability-gate:sha256:7aa1f0d51004c1245d5a9999522ab605d489b8decee5110fe6566d275804693f`,
  file SHA256
  `ab462f78f3f2847385516de2ee0bc565388514ce5ad7ce0b724b347a7e1e6db4`.
  The complete gate recorded `llama-3.3-70b` at 20/20 (**PASS**), `gemma4-26b`
  at 20/20 (**PASS**), and `minimax-m2.7` at 0/20, with the overall gate
  **FAIL**. MiniMax's recorded 0/20 must not be interpreted as a clean
  repeatability failure because most calls did not return evaluable final
  content.
- **Offline integrity diagnosis:** The artifact contains 60 MiniMax call
  records: 44 transport-exhausted calls, 44 calls with missing raw content, and
  136 infrastructure-error attempts. All 136 errors were
  `Maki message content is not a string`; only 16 calls returned usable final
  content.
- **Bounded DEVELOPMENT diagnostic:** A bounded raw-response diagnostic
  reproduced the cause: HTTP 200, `finish_reason = length`,
  `completion_tokens = 256`, reasoning content present, and final
  `content = null`.
- **Reason and consequence:** Under the frozen generation protocol, MiniMax can
  consume the frozen 256-token output budget in reasoning and fail to produce
  evaluable final answer content, while no provider-supported non-thinking
  configuration is available. Under Amendment 03, this operability failure
  activates the next frozen fallback candidate, `qwen3.8-27b`. `SELECTION` and
  `PROJECT_PROTECTED_FINAL` outcomes remained unopened.
- **Authority:**
  `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL_AMENDMENT_03.md`.

## 2026-08-29 — Post-reserve generator fallback order

- **Decision:** After `qwen3.5-122b` failed the frozen v3 repeatability gate at
  8/20, test `minimax-m2.7`, then `qwen3.8-27b`, then `qwen3.6-36b`, stopping at
  the first candidate that passes every frozen admission requirement.
- **Reason:** Preserve a three-generator design and meaningful model-family
  heterogeneity while preventing answer-quality comparison or post-hoc model
  shopping. MiniMax is first as an independent family relative to retained
  Llama and Gemma.
- **Conditions:** Each reached candidate requires immutable provenance,
  provider-supported development configuration, seed `20260823` verification,
  a new binding, and a new complete gate with `>=19/20` for all three models.
  Existing gates/bindings remain immutable; canonical generation stays blocked
  until a candidate passes; selection/protected outcomes remain unopened.
- **Authority:**
  `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL_AMENDMENT_03.md`.

## 2026-08-29 — Prospective reserve-LLM substitution

- **Decision:** Replace `ministral-3-14b` with the already-frozen reserve
  `qwen3.5-122b` for canonical Stage-3 generation while preserving exactly
  three generator slots.
- **Reason:** After the provider-unsupported Ministral chat-template control
  was removed and fixed provider seed `20260823` was introduced prospectively,
  the frozen seeded gate still gave Ministral 7/20 and failed overall. This is
  an unresolved repeatability trigger, not an answer-quality decision.
- **Conditions:** Before canonical use, record Qwen's exact runtime/model
  provenance, establish provider-supported direct/non-thinking behavior on
  `DEVELOPMENT`, check seed support, rerun affected development gates, and pass
  the complete final three-model gate. Existing v1/v2 gate artifacts remain
  immutable; `SELECTION` and protected-final outcomes remain unopened.
- **Authority:**
  `docs/sprint3/EXPERIMENT_STAGE_GATE_PROTOCOL_AMENDMENT_02.md`.

## 2026-08-18 — Canonical ColBERTv2 production backend

- **Decision:** Use direct Stanford ColBERT from `colbert-ai==0.2.22` with
  `colbert-ir/colbertv2.0` pinned to immutable revision
  `0855eac81381e0323a846f1ed7d8452d4c648b50`. Production runtime must resolve
  that revision to a local immutable snapshot path before initializing Stanford
  ColBERT.
- **Evidence:** Repository audit found the legacy retriever uses RAGatouille and
  an unpinned checkpoint identifier. The reviewed production design requires
  exact configuration and corpus provenance.
- **Alternative considered:** RAGatouille's `PLAIDModelIndex` and
  `RAGPretrainedModel`.
- **Reason:** For this project's small corpus, RAGatouille's PLAID wrapper can
  silently adapt `nbits`, k-means, and search behavior to collection size. The
  canonical Sprint 3 method instead fixes those scientific settings explicitly.
- **Affected experiments:** Future Sprint 3 canonical ColBERTv2 indexing and
  PubMedQA top-20 candidate production. Historical artifacts and the legacy
  retriever remain unchanged.
- **Git commit:** To be recorded when this decision is reviewed and committed.

### Scientific-field roles

The complete retrieval-method identity contains model/checkpoint, indexing, and
search behavior. The checkpoint-loading requirement is recorded here but is not
yet implemented or runtime-verified in this configuration-only milestone.

**Model / checkpoint side**

- checkpoint: `colbert-ir/colbertv2.0` at immutable revision
  `0855eac81381e0323a846f1ed7d8452d4c648b50`;
- `dim=128`, `query_maxlen=32`, and `doc_maxlen=180`;
- cosine similarity with ColBERT late interaction / MaxSim;
- `mask_punctuation=True` and `attend_to_mask_tokens=False`;
- exact query strings and exact validated `CorpusRecord.retrieval_content`, with
  no manual lowercasing or external normalization and tokenizer-native behavior
  preserved.

The checkpoint identity, 128-dimensional ColBERT representation, cosine/MaxSim
method, punctuation masking, mask-token behavior, and tokenizer-native behavior
come from the pinned checkpoint and direct Stanford ColBERT method. The maximum
query/document lengths are deliberately frozen study controls.

**Index side**

- direct Stanford ColBERT PLAID through `colbert-ai==0.2.22`;
- `nbits=2`, `kmeans_niters=4`, `index_bsize=64`, `nranks=1`, and `seed=12345`.

The PLAID engine is Stanford ColBERT behavior; its exact compression,
clustering, batching, rank, and seed settings are deliberate study controls that
prevent collection-size-dependent wrapper adaptation.

**Search side**

- `candidate_pool_size=20`, `search_ncells=2`,
  `search_centroid_score_threshold=0.45`, and `search_ndocs=1024`;
- native higher-is-better ColBERT ranking;
- no post-filtering, post-reranking, or tie manipulation.

Native ranking is Stanford ColBERT behavior. The pool size and PLAID search
parameters, together with the explicit absence of downstream ranking changes,
are deliberate study controls.

## 2026-08-18 — Isolated canonical ColBERT runtime dependencies

- **Decision:** Run canonical direct Stanford ColBERT in an isolated environment
  with `colbert-ai==0.2.22`, `transformers==4.57.6`, and
  `huggingface_hub==0.36.2`. The ColBERT-only RunPod lock also pins
  `ninja==1.13.0`, required by PyTorch's C++ extension loader.
- **Reason:** The general project environment's `transformers==5.14.1`
  reproducibly failed during real Stanford ColBERT checkpoint initialization
  because `HF_ColBERT` lacked `all_tied_weights_keys`.
- **Evidence:** The same pinned checkpoint initialized under Transformers
  4.40.2 and 4.57.6. Version 4.57.6 was selected because a fresh environment
  mechanically derived from the project lock remained dependency-consistent
  with `sentence-transformers==5.7.0`, passed `pip check`, initialized the real
  checkpoint successfully, and retained physical checkpoint manifest
  `0be9d161036288daa494d3627655691c07812a6bea0cb488e148845ffb4b0287`.
- **Scope:** Runtime dependency compatibility and reproducibility only. No
  scientific ColBERT retrieval parameter changed; no claim of scientific
  superiority is made.
- **Status:** Local CPU environment validated. RunPod CUDA 13.0 / RTX 5090
  environment pending GPU validation.
