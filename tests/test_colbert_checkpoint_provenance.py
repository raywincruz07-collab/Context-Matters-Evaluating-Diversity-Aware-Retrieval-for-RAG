from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrievers.colbert_checkpoint import ResolvedColBERTCheckpoint
from retrievers.colbert_checkpoint_provenance import (
    build_colbert_checkpoint_physical_provenance,
    validate_colbert_checkpoint_weight_structure,
)
from retrievers.colbert_config import COLBERT_CONFIG


REVISION = "0855eac81381e0323a846f1ed7d8452d4c648b50"
REQUIRED_FILES = (
    "artifact.metadata",
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def model_metadata(**changes):
    values = {
        "dim": 128,
        "query_maxlen": 32,
        "doc_maxlen": 180,
        "similarity": "cosine",
        "mask_punctuation": True,
        "attend_to_mask_tokens": False,
        "nbits": 1,
        "kmeans_niters": 20,
        "nranks": 4,
    }
    values.update(changes)
    return values


def make_snapshot(tmp_path: Path, *, metadata=None, do_lower_case=True):
    snapshot = tmp_path / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    contents = {
        "artifact.metadata": json.dumps(
            {"config": model_metadata() if metadata is None else metadata}
        ).encode(),
        "config.json": b'{"fixture":true}',
        "pytorch_model.bin": b"synthetic weights",
        "special_tokens_map.json": b"{}",
        "tokenizer.json": b"synthetic tokenizer",
        "tokenizer_config.json": json.dumps(
            {"do_lower_case": do_lower_case}
        ).encode(),
        "vocab.txt": b"token\n",
        ".gitattributes": b"*.bin filter=lfs\n",
        "nested/extra.txt": b"extra",
    }
    for relative_path, content in contents.items():
        path = snapshot / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return snapshot, contents


def resolved(snapshot: Path):
    return ResolvedColBERTCheckpoint(
        checkpoint_id=COLBERT_CONFIG.checkpoint_id,
        checkpoint_revision=REVISION,
        snapshot_path=snapshot,
    )


def test_deterministic_file_order_hashes_manifest_count_and_bytes(tmp_path):
    snapshot, contents = make_snapshot(tmp_path)
    value = build_colbert_checkpoint_physical_provenance(resolved(snapshot))

    expected_paths = sorted(contents)
    assert [file.relative_path for file in value.files] == expected_paths
    expected_lines = []
    for relative_path in expected_paths:
        content = contents[relative_path]
        digest = hashlib.sha256(content).hexdigest()
        file = next(item for item in value.files if item.relative_path == relative_path)
        assert file.sha256 == digest
        assert file.size_bytes == len(content)
        expected_lines.append(f"{digest}  {len(content)}  {relative_path}\n")
    expected_manifest = "".join(expected_lines)
    assert value.snapshot_manifest_sha256 == hashlib.sha256(
        expected_manifest.encode("utf-8")
    ).hexdigest()
    assert value.file_count == len(contents)
    assert value.total_bytes == sum(map(len, contents.values()))
    assert value.payload() == value.payload()


@pytest.mark.parametrize("missing", REQUIRED_FILES)
def test_missing_required_file_fails(tmp_path, missing):
    snapshot, _ = make_snapshot(tmp_path)
    (snapshot / missing).unlink()

    with pytest.raises(ValueError, match="missing required files"):
        build_colbert_checkpoint_physical_provenance(resolved(snapshot))


def test_model_metadata_and_tokenizer_native_lowercasing_are_validated(tmp_path):
    snapshot, _ = make_snapshot(tmp_path)
    value = build_colbert_checkpoint_physical_provenance(resolved(snapshot))

    assert value.metadata.dim == 128
    assert value.metadata.query_maxlen == 32
    assert value.metadata.doc_maxlen == 180
    assert value.metadata.similarity == "cosine"
    assert value.metadata.mask_punctuation is True
    assert value.metadata.attend_to_mask_tokens is False
    assert value.metadata.tokenizer_do_lower_case is True


@pytest.mark.parametrize(
    "field,wrong",
    [
        ("dim", 64),
        ("query_maxlen", 64),
        ("doc_maxlen", 256),
        ("similarity", "dot"),
        ("mask_punctuation", False),
        ("attend_to_mask_tokens", True),
    ],
)
def test_every_model_side_metadata_mismatch_fails(tmp_path, field, wrong):
    snapshot, _ = make_snapshot(tmp_path, metadata=model_metadata(**{field: wrong}))

    with pytest.raises(ValueError, match=field):
        build_colbert_checkpoint_physical_provenance(resolved(snapshot))


def test_historical_index_metadata_is_preserved_not_validated_as_sprint3(tmp_path):
    historical = model_metadata(nbits=1, kmeans_niters=20, nranks=4)
    snapshot, _ = make_snapshot(tmp_path, metadata=historical)
    value = build_colbert_checkpoint_physical_provenance(resolved(snapshot))

    assert value.metadata.historical_nbits == 1
    assert value.metadata.historical_kmeans_niters == 20
    assert value.metadata.historical_nranks == 4
    assert COLBERT_CONFIG.nbits == 2
    assert COLBERT_CONFIG.kmeans_niters == 4
    assert COLBERT_CONFIG.nranks == 1


def test_tokenizer_native_lowercasing_mismatch_fails(tmp_path):
    snapshot, _ = make_snapshot(tmp_path, do_lower_case=False)

    with pytest.raises(ValueError, match="do_lower_case must be true"):
        build_colbert_checkpoint_physical_provenance(resolved(snapshot))


def state_dict(**changes):
    values = {
        "linear.weight": torch.empty((128, 768), dtype=torch.float32, device="meta"),
        "bert.embeddings.word_embeddings.weight": torch.empty(
            (30522, 768), dtype=torch.float32, device="meta"
        ),
        "bert.embeddings.position_embeddings.weight": torch.empty(
            (512, 768), dtype=torch.float32, device="meta"
        ),
    }
    values.update(changes)
    return values


def write_weights(snapshot: Path, values):
    torch.save(values, snapshot / "pytorch_model.bin")


def test_weight_structure_acceptance_uses_safe_cpu_load(tmp_path, monkeypatch):
    snapshot, _ = make_snapshot(tmp_path)
    write_weights(snapshot, state_dict())
    original_load = torch.load
    calls = []

    def load_spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load_spy)
    value = validate_colbert_checkpoint_weight_structure(resolved(snapshot))

    assert calls[0][1] == {"map_location": "cpu", "weights_only": True}
    assert value.projection_name == "linear.weight"
    assert value.projection_shape == (128, 768)
    assert value.word_embeddings_shape == (30522, 768)
    assert value.position_embeddings_shape == (512, 768)
    assert value.dtype == "float32"


def test_wrong_projection_dimension_fails(tmp_path):
    snapshot, _ = make_snapshot(tmp_path)
    values = state_dict()
    values["linear.weight"] = torch.empty((64, 768), device="meta")
    write_weights(snapshot, values)

    with pytest.raises(ValueError, match="exactly one 128x768"):
        validate_colbert_checkpoint_weight_structure(resolved(snapshot))


def test_missing_projection_fails(tmp_path):
    snapshot, _ = make_snapshot(tmp_path)
    values = state_dict()
    values.pop("linear.weight")
    write_weights(snapshot, values)

    with pytest.raises(ValueError, match="exactly one 128x768"):
        validate_colbert_checkpoint_weight_structure(resolved(snapshot))


def test_duplicate_projection_candidate_fails(tmp_path):
    snapshot, _ = make_snapshot(tmp_path)
    values = state_dict(
        **{"duplicate.weight": torch.empty((128, 768), device="meta")}
    )
    write_weights(snapshot, values)

    with pytest.raises(ValueError, match="exactly one 128x768"):
        validate_colbert_checkpoint_weight_structure(resolved(snapshot))


@pytest.mark.parametrize(
    "name,shape",
    [
        ("bert.embeddings.word_embeddings.weight", (10, 768)),
        ("bert.embeddings.position_embeddings.weight", (256, 768)),
    ],
)
def test_wrong_embedding_shapes_fail(tmp_path, name, shape):
    snapshot, _ = make_snapshot(tmp_path)
    values = state_dict()
    values[name] = torch.empty(shape, device="meta")
    write_weights(snapshot, values)

    with pytest.raises(ValueError, match=name):
        validate_colbert_checkpoint_weight_structure(resolved(snapshot))


def test_scientific_config_unchanged_and_snapshot_path_not_scientific(tmp_path):
    snapshot, _ = make_snapshot(tmp_path)
    before = asdict(COLBERT_CONFIG)
    scientific_before = COLBERT_CONFIG.scientific_payload()

    value = build_colbert_checkpoint_physical_provenance(resolved(snapshot))

    assert asdict(COLBERT_CONFIG) == before
    assert COLBERT_CONFIG.scientific_payload() == scientific_before
    assert "snapshot_path" not in scientific_before
    assert value.payload()["snapshot_path"] == str(snapshot.resolve())
