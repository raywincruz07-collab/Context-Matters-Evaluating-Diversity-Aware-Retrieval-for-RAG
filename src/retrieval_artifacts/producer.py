"""Pure producers and checksum helpers for frozen candidate artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from retrievers.bm25_config import BM25_CONFIG, BM25Config
from retrieval_artifacts.contracts import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateArtifact,
    CandidateEntry,
    CorpusProvenance,
    DatasetProvenance,
    RetrieverProvenance,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

def _require_stable_id(value: object, name: str) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a string or non-boolean integer")


def _canonical_id(value: str | int) -> str | int:
    return value if isinstance(value, str) else int(value)


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")


def _canonical_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorpusRecord:
    """Authoritative corpus row used to validate raw retrieval results."""

    document_id: str | int
    source_document_id: str | int | None
    text: str
    corpus_position: int

    def __post_init__(self) -> None:
        _require_stable_id(self.document_id, "document_id")
        if self.source_document_id is not None:
            _require_stable_id(self.source_document_id, "source_document_id")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not self.text:
            raise ValueError("text must be non-empty")
        if isinstance(self.corpus_position, (bool, np.bool_)) or not isinstance(
            self.corpus_position, Integral
        ):
            raise TypeError("corpus_position must be a non-boolean integer")
        if self.corpus_position < 0:
            raise ValueError("corpus_position must be >= 0")


@dataclass(frozen=True)
class RawCandidateResult:
    """Minimal ordered output emitted by a retriever."""

    document_id: str | int
    native_score: float

    def __post_init__(self) -> None:
        _require_stable_id(self.document_id, "document_id")
        if isinstance(self.native_score, (bool, np.bool_)) or not isinstance(
            self.native_score, Real
        ):
            raise TypeError("native_score must be a non-boolean real number")
        if not math.isfinite(float(self.native_score)):
            raise ValueError("native_score must be finite")


def document_content_sha256(text: str) -> str:
    """Hash the exact, unnormalized document text as UTF-8 bytes."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validated_corpus_records(
    corpus_records: tuple[CorpusRecord, ...],
) -> tuple[CorpusRecord, ...]:
    if not isinstance(corpus_records, tuple):
        raise TypeError("corpus_records must be an immutable tuple")
    if not corpus_records:
        raise ValueError("corpus_records must not be empty")
    if not all(isinstance(record, CorpusRecord) for record in corpus_records):
        raise TypeError("corpus_records must contain CorpusRecord objects")
    document_ids = [record.document_id for record in corpus_records]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("corpus document_id values must be unique")
    positions = [int(record.corpus_position) for record in corpus_records]
    if len(set(positions)) != len(positions):
        raise ValueError("corpus_position values must be unique")
    expected = list(range(len(corpus_records)))
    if sorted(positions) != expected:
        raise ValueError("corpus positions must exactly cover 0..len(corpus_records)-1")
    if positions != expected:
        raise ValueError("corpus tuple order must match corpus_position")
    return corpus_records


def compute_document_id_map_sha256(
    corpus_records: tuple[CorpusRecord, ...],
) -> str:
    """Hash the ordered position-to-document-ID mapping."""
    records = _validated_corpus_records(corpus_records)
    payload = [
        {
            "corpus_position": int(record.corpus_position),
            "document_id": _canonical_id(record.document_id),
        }
        for record in records
    ]
    return _canonical_hash(payload)


def compute_corpus_manifest_sha256(
    corpus_records: tuple[CorpusRecord, ...],
) -> str:
    """Hash ordered corpus identities and exact text checksums."""
    records = _validated_corpus_records(corpus_records)
    payload = [
        {
            "corpus_position": int(record.corpus_position),
            "document_content_sha256": document_content_sha256(record.text),
            "document_id": _canonical_id(record.document_id),
            "source_document_id": (
                None
                if record.source_document_id is None
                else _canonical_id(record.source_document_id)
            ),
        }
        for record in records
    ]
    return _canonical_hash(payload)


def compute_index_fingerprint_sha256(
    *,
    corpus_manifest_sha256: str,
    document_id_map_sha256: str,
    library_version: str,
    bm25_config: BM25Config = BM25_CONFIG,
) -> str:
    """Fingerprint BM25 configuration and its exact corpus binding."""
    _require_sha256(corpus_manifest_sha256, "corpus_manifest_sha256")
    _require_sha256(document_id_map_sha256, "document_id_map_sha256")
    if not isinstance(bm25_config, BM25Config):
        raise TypeError("bm25_config must be BM25Config")
    if not isinstance(library_version, str) or not library_version.strip():
        raise ValueError("library_version must be a non-empty string")
    return _canonical_hash(
        {
            "bm25_config": bm25_config.scientific_payload(),
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "document_id_map_sha256": document_id_map_sha256,
            "library_name": "rank_bm25",
            "library_version": library_version,
        }
    )


def build_bm25_retriever_provenance(
    *,
    library_version: str,
    index_fingerprint_sha256: str,
    index_artifact_sha256: str | None = None,
    bm25_config: BM25Config = BM25_CONFIG,
) -> RetrieverProvenance:
    """Describe the repository's current BM25Retriever without inspecting runtime."""
    if not isinstance(bm25_config, BM25Config):
        raise TypeError("bm25_config must be BM25Config")
    return RetrieverProvenance(
        retriever_name="bm25",
        implementation="retrievers.bm25_retriever.BM25Retriever",
        library_name="rank_bm25",
        library_version=library_version,
        model_id=None,
        model_revision=None,
        tokenizer_id=None,
        tokenizer_revision=None,
        query_preprocessing=bm25_config.query_preprocessing,
        document_preprocessing=bm25_config.document_preprocessing,
        normalization=bm25_config.normalization,
        score_semantics=bm25_config.score_semantics,
        index_type=bm25_config.index_type,
        index_config=bm25_config.scientific_json(),
        index_fingerprint_sha256=index_fingerprint_sha256,
        index_artifact_sha256=index_artifact_sha256,
    )


def validate_bm25_index_binding(
    *,
    corpus_provenance: CorpusProvenance,
    retriever_provenance: RetrieverProvenance,
    bm25_config: BM25Config = BM25_CONFIG,
) -> None:
    """Validate declared BM25 configuration-to-corpus fingerprint consistency.

    This validates provenance declarations only. Runtime cache binding is enforced
    separately by ``BM25Retriever`` and is not the scientific index fingerprint.
    """
    if not isinstance(corpus_provenance, CorpusProvenance):
        raise TypeError("corpus_provenance must be CorpusProvenance")
    if not isinstance(retriever_provenance, RetrieverProvenance):
        raise TypeError("retriever_provenance must be RetrieverProvenance")
    if not isinstance(bm25_config, BM25Config):
        raise TypeError("bm25_config must be BM25Config")
    if retriever_provenance.retriever_name != "bm25":
        raise ValueError("retriever_provenance must describe the bm25 family")
    if retriever_provenance.library_name != "rank_bm25":
        raise ValueError("BM25 retriever provenance must use rank_bm25")
    if (
        retriever_provenance.model_id is not None
        or retriever_provenance.model_revision is not None
    ):
        raise ValueError("BM25 retriever provenance must not declare a model")
    if (
        retriever_provenance.tokenizer_id is not None
        or retriever_provenance.tokenizer_revision is not None
    ):
        raise ValueError("BM25 retriever provenance must not declare a tokenizer")
    expected_fields = {
        "query_preprocessing": bm25_config.query_preprocessing,
        "document_preprocessing": bm25_config.document_preprocessing,
        "normalization": bm25_config.normalization,
        "score_semantics": bm25_config.score_semantics,
        "index_type": bm25_config.index_type,
        "index_config": bm25_config.scientific_json(),
    }
    if any(
        getattr(retriever_provenance, name) != value
        for name, value in expected_fields.items()
    ):
        raise ValueError("BM25 retriever provenance does not match bm25_config")

    expected = compute_index_fingerprint_sha256(
        corpus_manifest_sha256=corpus_provenance.manifest_sha256,
        document_id_map_sha256=corpus_provenance.document_id_map_sha256,
        library_version=retriever_provenance.library_version,
        bm25_config=bm25_config,
    )
    if retriever_provenance.index_fingerprint_sha256 != expected:
        raise ValueError(
            "BM25 index fingerprint does not match the declared corpus and configuration"
        )


def produce_bm25_candidate_artifact(
    *,
    dataset_provenance: DatasetProvenance,
    corpus_provenance: CorpusProvenance,
    sample_id: str | int,
    query_text: str,
    retriever_provenance: RetrieverProvenance,
    requested_top_n: int,
    raw_results: Sequence[RawCandidateResult],
    corpus_records: tuple[CorpusRecord, ...],
    producing_git_commit: str,
    worktree_clean: bool,
    environment_fingerprint_sha256: str,
    bm25_config: BM25Config = BM25_CONFIG,
) -> CandidateArtifact:
    """Validate and freeze already-returned raw BM25 retrieval results."""
    if not isinstance(dataset_provenance, DatasetProvenance):
        raise TypeError("dataset_provenance must be DatasetProvenance")
    if not isinstance(corpus_provenance, CorpusProvenance):
        raise TypeError("corpus_provenance must be CorpusProvenance")
    if not isinstance(retriever_provenance, RetrieverProvenance):
        raise TypeError("retriever_provenance must be RetrieverProvenance")
    records = _validated_corpus_records(corpus_records)
    if corpus_provenance.document_count != len(records):
        raise ValueError("corpus document_count does not match corpus_records")
    actual_manifest = compute_corpus_manifest_sha256(records)
    if corpus_provenance.manifest_sha256 != actual_manifest:
        raise ValueError("corpus manifest_sha256 does not match corpus_records")
    actual_id_map = compute_document_id_map_sha256(records)
    if corpus_provenance.document_id_map_sha256 != actual_id_map:
        raise ValueError("corpus document_id_map_sha256 does not match corpus_records")
    validate_bm25_index_binding(
        corpus_provenance=corpus_provenance,
        retriever_provenance=retriever_provenance,
        bm25_config=bm25_config,
    )

    if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Sequence):
        raise TypeError("raw_results must be an ordered sequence")
    results = tuple(raw_results)
    if not results:
        raise ValueError("raw_results must contain at least one result")
    if not all(isinstance(result, RawCandidateResult) for result in results):
        raise TypeError("raw_results must contain RawCandidateResult objects")
    # CandidateArtifact validates requested_top_n and the final count as well.
    if len(results) > requested_top_n:
        raise ValueError("raw result count cannot exceed requested_top_n")
    result_ids = [result.document_id for result in results]
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("raw result document_id values must be unique")

    records_by_id = {record.document_id: record for record in records}
    candidates = []
    for rank, result in enumerate(results, start=1):
        try:
            record = records_by_id[result.document_id]
        except KeyError as exc:
            raise ValueError(
                f"raw result document_id is absent from corpus: {result.document_id!r}"
            ) from exc
        candidates.append(
            CandidateEntry(
                rank=rank,
                document_id=record.document_id,
                source_document_id=record.source_document_id,
                corpus_position=record.corpus_position,
                native_score=result.native_score,
                document_content_sha256=document_content_sha256(record.text),
            )
        )

    return CandidateArtifact(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        dataset=dataset_provenance,
        corpus=corpus_provenance,
        sample_id=sample_id,
        query_text=query_text,
        retriever=retriever_provenance,
        requested_top_n=requested_top_n,
        candidates=tuple(candidates),
        producing_git_commit=producing_git_commit,
        worktree_clean=worktree_clean,
        environment_fingerprint_sha256=environment_fingerprint_sha256,
    )
