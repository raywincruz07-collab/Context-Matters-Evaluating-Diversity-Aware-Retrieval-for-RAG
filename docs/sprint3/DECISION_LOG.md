# Sprint 3 Decision Log

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
