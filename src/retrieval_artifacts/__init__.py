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
from retrieval_artifacts.sample_manifest import (
    SAMPLE_MANIFEST_SCHEMA_VERSION,
    SampleManifest,
    SampleManifestEntry,
    dataset_provenance_from_sample_manifest,
    query_text_sha256,
    verify_manifest_sample,
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
    "SAMPLE_MANIFEST_SCHEMA_VERSION",
    "SampleManifest",
    "SampleManifestEntry",
    "dataset_provenance_from_sample_manifest",
    "query_text_sha256",
    "verify_manifest_sample",
]
