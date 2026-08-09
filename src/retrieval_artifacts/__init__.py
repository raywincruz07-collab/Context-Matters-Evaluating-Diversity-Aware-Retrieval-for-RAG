"""Contracts for immutable Sprint 3 retrieval artifacts."""

from retrieval_artifacts.contracts import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateArtifact,
    CandidateEmbeddingArtifactRef,
    CandidateEntry,
    CorpusProvenance,
    DatasetProvenance,
    RetrieverProvenance,
)
from retrieval_artifacts.producer import (
    CorpusRecord,
    RawCandidateResult,
    build_bm25_retriever_provenance,
    compute_corpus_manifest_sha256,
    compute_document_id_map_sha256,
    compute_index_fingerprint_sha256,
    document_content_sha256,
    produce_bm25_candidate_artifact,
    validate_bm25_index_binding,
)

__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "CandidateArtifact",
    "CandidateEmbeddingArtifactRef",
    "CandidateEntry",
    "CorpusProvenance",
    "DatasetProvenance",
    "RetrieverProvenance",
    "CorpusRecord",
    "RawCandidateResult",
    "build_bm25_retriever_provenance",
    "compute_corpus_manifest_sha256",
    "compute_document_id_map_sha256",
    "compute_index_fingerprint_sha256",
    "document_content_sha256",
    "produce_bm25_candidate_artifact",
    "validate_bm25_index_binding",
]
