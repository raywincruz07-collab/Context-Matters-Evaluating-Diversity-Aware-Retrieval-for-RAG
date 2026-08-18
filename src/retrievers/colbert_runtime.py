"""Controlled direct-Stanford-ColBERT configuration and checkpoint adapter."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from retrievers.colbert_checkpoint import ResolvedColBERTCheckpoint
from retrievers.colbert_config import COLBERT_CONFIG, ColBERTConfig


_SCIENTIFIC_INTERACTION = "ColBERT late interaction / MaxSim"
_STANFORD_INTERACTION = "colbert"


@dataclass(frozen=True)
class StanfordColBERTEffectiveConfig:
    """Validated effective Stanford settings retained as runtime provenance."""

    checkpoint: str
    dim: int
    query_maxlen: int
    doc_maxlen: int
    similarity: str
    interaction: str
    mask_punctuation: bool
    attend_to_mask_tokens: bool
    index_bsize: int
    nbits: int
    kmeans_niters: int
    nranks: int
    ncells: int
    centroid_score_threshold: float
    ndocs: int

    def payload(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class InitializedColBERTCheckpoint:
    """Initialized Stanford checkpoint plus validated effective runtime state."""

    checkpoint: Any
    resolved_checkpoint: ResolvedColBERTCheckpoint
    effective_config: StanfordColBERTEffectiveConfig
    colbert_ai_version: str


def _stanford_config_class():
    from colbert.infra import ColBERTConfig as StanfordColBERTConfig

    return StanfordColBERTConfig


def _stanford_checkpoint_class():
    from colbert.modeling.checkpoint import Checkpoint

    return Checkpoint


def _require_colbert_ai_version(config: ColBERTConfig) -> str:
    try:
        installed = importlib_metadata.version("colbert-ai")
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError("required distribution colbert-ai is not installed") from error
    if installed != config.colbert_ai_version:
        raise RuntimeError(
            "colbert-ai version mismatch: "
            f"expected {config.colbert_ai_version}, got {installed}"
        )
    return installed


def _validated_snapshot_path(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    config: ColBERTConfig,
) -> Path:
    if not isinstance(resolved_checkpoint, ResolvedColBERTCheckpoint):
        raise TypeError("resolved_checkpoint must be a ResolvedColBERTCheckpoint")
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")
    if resolved_checkpoint.checkpoint_id != config.checkpoint_id:
        raise ValueError("resolved checkpoint ID does not match ColBERTConfig")
    if resolved_checkpoint.checkpoint_revision != config.checkpoint_revision:
        raise ValueError("resolved checkpoint revision does not match ColBERTConfig")

    snapshot_path = resolved_checkpoint.snapshot_path.resolve()
    if not snapshot_path.is_dir():
        raise NotADirectoryError("resolved ColBERT snapshot must be a directory")
    if snapshot_path.name != config.checkpoint_revision:
        raise ValueError("resolved snapshot directory does not match checkpoint revision")
    return snapshot_path


def _expected_stanford_values(
    snapshot_path: Path,
    config: ColBERTConfig,
) -> dict[str, Any]:
    if config.interaction != _SCIENTIFIC_INTERACTION:
        raise ValueError("unsupported ColBERT scientific interaction")
    return {
        "checkpoint": str(snapshot_path),
        "dim": config.dim,
        "query_maxlen": config.query_maxlen,
        "doc_maxlen": config.doc_maxlen,
        "similarity": config.similarity,
        # Stanford names the frozen ColBERT late-interaction/MaxSim mode "colbert".
        "interaction": _STANFORD_INTERACTION,
        "mask_punctuation": config.mask_punctuation,
        "attend_to_mask_tokens": config.attend_to_mask_tokens,
        "index_bsize": config.index_bsize,
        "nbits": config.nbits,
        "kmeans_niters": config.kmeans_niters,
        "nranks": config.nranks,
        "ncells": config.search_ncells,
        "centroid_score_threshold": float(
            config.search_centroid_score_threshold
        ),
        "ndocs": config.search_ndocs,
    }


def validate_effective_stanford_config(
    stanford_config: Any,
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig = COLBERT_CONFIG,
) -> StanfordColBERTEffectiveConfig:
    """Validate every translated value after Stanford config merging."""
    snapshot_path = _validated_snapshot_path(resolved_checkpoint, config)
    expected = _expected_stanford_values(snapshot_path, config)
    for field, expected_value in expected.items():
        if not hasattr(stanford_config, field):
            raise ValueError(f"Stanford ColBERTConfig is missing {field}")
        actual = getattr(stanford_config, field)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"effective Stanford ColBERTConfig {field} mismatch: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    return StanfordColBERTEffectiveConfig(**expected)


def build_stanford_colbert_config(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig = COLBERT_CONFIG,
):
    """Translate the frozen project method into explicit Stanford settings.

    Stanford performs the learned 768→128 projection, token-vector L2
    normalization, document padding/punctuation masking, and ColBERT MaxSim sum.
    This adapter does not normalize, lowercase, or score text itself.

    Candidate-pool/ranking/post-processing controls are intentionally not
    Stanford constructor parameters. The project seed is likewise deferred to
    the later indexing milestone because Stanford ColBERTConfig has no project
    seed field.
    """
    if not isinstance(config, ColBERTConfig):
        raise TypeError("config must be a ColBERTConfig")
    _require_colbert_ai_version(config)
    snapshot_path = _validated_snapshot_path(resolved_checkpoint, config)
    values = _expected_stanford_values(snapshot_path, config)
    stanford_config = _stanford_config_class()(**values)
    validate_effective_stanford_config(
        stanford_config,
        resolved_checkpoint,
        config=config,
    )
    return stanford_config


def initialize_colbert_checkpoint(
    resolved_checkpoint: ResolvedColBERTCheckpoint,
    *,
    config: ColBERTConfig = COLBERT_CONFIG,
    verbose: int = 0,
) -> InitializedColBERTCheckpoint:
    """Initialize only from the validated local immutable snapshot path."""
    installed_version = _require_colbert_ai_version(config)
    snapshot_path = _validated_snapshot_path(resolved_checkpoint, config)
    stanford_config = build_stanford_colbert_config(
        resolved_checkpoint,
        config=config,
    )
    checkpoint = _stanford_checkpoint_class()(
        str(snapshot_path),
        colbert_config=stanford_config,
        verbose=verbose,
    )
    if not hasattr(checkpoint, "colbert_config"):
        raise ValueError("initialized Checkpoint is missing colbert_config")
    effective_config = validate_effective_stanford_config(
        checkpoint.colbert_config,
        resolved_checkpoint,
        config=config,
    )
    return InitializedColBERTCheckpoint(
        checkpoint=checkpoint,
        resolved_checkpoint=resolved_checkpoint,
        effective_config=effective_config,
        colbert_ai_version=installed_version,
    )
