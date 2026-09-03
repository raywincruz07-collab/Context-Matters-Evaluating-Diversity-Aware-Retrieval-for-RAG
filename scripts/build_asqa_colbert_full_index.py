from __future__ import annotations

from pathlib import Path
import hashlib
import json
import torch

from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig


WS = Path(
    "/pfs/work9/workspace/scratch/"
    "ma_rthummar-context_matters_sprint3"
)

CHECKPOINT = WS / "models" / "colbertv2.0"

COLLECTION = (
    WS
    / "data"
    / "asqa"
    / "dpr_wikipedia_2018_12_20"
    / "colbert_collection.tsv"
)

CORPUS_MANIFEST = (
    WS
    / "data"
    / "asqa"
    / "dpr_wikipedia_2018_12_20"
    / "asqa_dpr_corpus_manifest.json"
)

ROOT = WS / "colbert_production"

EXPERIMENT = "asqa_dpr_wikipedia_full"
INDEX_NAME = "colbertv2.nbits2"


EXPECTED_PASSAGES = 21_015_324
EXPECTED_COLLECTION_SIZE = 13_112_886_050

EXPECTED_COLLECTION_SHA256 = (
    "2a0649ee6d4c925bf713b59012d6f4d0"
    "2762bf7a7fcd960fbe475fe1190d9c1d"
)

EXPECTED_CORPUS_SCIENTIFIC_SHA256 = (
    "cb3a98b38afd1cf486f9172ab9fd92fa"
    "7ecaa6391d399f059c0bfe4c0077f4d4"
)

EXPECTED_CHECKPOINT_WEIGHTS_SHA256 = (
    "26e4c2f9f95a3da4442252bb40d99e4f"
    "bfd098e733edac1d785c9937b8a278da"
)

EXPECTED_CHECKPOINT_METADATA_SHA256 = (
    "0ddc5a54234cff6d13bc9411250a5479"
    "d9d96f3ffbace76d8a1884144377e434"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def preflight() -> None:
    print("=== PRODUCTION PREFLIGHT ===")

    if not CHECKPOINT.is_dir():
        raise FileNotFoundError(CHECKPOINT)

    if not COLLECTION.is_file():
        raise FileNotFoundError(COLLECTION)

    if COLLECTION.stat().st_size != EXPECTED_COLLECTION_SIZE:
        raise ValueError(
            "Unexpected ColBERT collection size: "
            f"{COLLECTION.stat().st_size}"
        )

    print("Verifying ColBERT collection SHA256...", flush=True)
    collection_sha256 = file_sha256(COLLECTION)

    if collection_sha256 != EXPECTED_COLLECTION_SHA256:
        raise ValueError(
            "ColBERT collection SHA256 mismatch: "
            f"{collection_sha256}"
        )

    weights = CHECKPOINT / "pytorch_model.bin"
    metadata = CHECKPOINT / "artifact.metadata"

    print("Verifying ColBERTv2 checkpoint SHA256...", flush=True)

    if file_sha256(weights) != EXPECTED_CHECKPOINT_WEIGHTS_SHA256:
        raise ValueError("ColBERTv2 model weights SHA256 mismatch")

    if file_sha256(metadata) != EXPECTED_CHECKPOINT_METADATA_SHA256:
        raise ValueError("ColBERTv2 artifact.metadata SHA256 mismatch")

    with CORPUS_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest["scientific_sha256"] != EXPECTED_CORPUS_SCIENTIFIC_SHA256:
        raise ValueError("Canonical DPR scientific SHA256 mismatch")

    payload = manifest["scientific_payload"]

    if payload["passage_count"] != EXPECTED_PASSAGES:
        raise ValueError("Canonical DPR passage count mismatch")

    if payload["canonical_surface"] != "exact DPR passage body":
        raise ValueError("Canonical corpus is not body-only")

    if payload["title_used_for_canonical_retrieval"] is not False:
        raise ValueError("Canonical corpus unexpectedly uses titles")

    print("collection:", COLLECTION)
    print("collection bytes:", COLLECTION.stat().st_size)
    print("collection SHA256:", collection_sha256)
    print("checkpoint identity: PASS")
    print("passages:", EXPECTED_PASSAGES)
    print("checkpoint:", CHECKPOINT)
    print("output root:", ROOT)
    print("experiment:", EXPERIMENT)
    print("index:", INDEX_NAME)

    print()
    print("CUDA available:", torch.cuda.is_available())
    print("visible GPUs:", torch.cuda.device_count())

    for gpu in range(torch.cuda.device_count()):
        print(f"GPU {gpu}:", torch.cuda.get_device_name(gpu))

    if torch.cuda.device_count() != 4:
        raise RuntimeError(
            f"Expected exactly 4 GPUs; found {torch.cuda.device_count()}"
        )

    print("PRODUCTION PREFLIGHT: PASS")


def main() -> None:
    preflight()

    ROOT.mkdir(parents=True, exist_ok=True)

    print()
    print("=== BUILD FULL COLBERT INDEX ===")

    with Run().context(
        RunConfig(
            nranks=4,
            experiment=EXPERIMENT,
            root=str(ROOT),
        )
    ):
        config = ColBERTConfig(
            root=str(ROOT),
            nbits=2,
            doc_maxlen=180,
            index_bsize=64,
        )

        indexer = Indexer(
            checkpoint=str(CHECKPOINT),
            config=config,
        )

        index_path = indexer.index(
            name=INDEX_NAME,
            collection=str(COLLECTION),
            overwrite="resume",
        )

    index_path = Path(index_path)

    print()
    print("=== INDEX COMPLETE ===")
    print("index_path:", index_path)

    if not index_path.is_dir():
        raise RuntimeError("Index directory missing after indexing")

    if not (index_path / "metadata.json").is_file():
        raise RuntimeError("metadata.json missing after indexing")

    if not (index_path / "ivf.pid.pt").is_file():
        raise RuntimeError("ivf.pid.pt missing after indexing")

    print()
    print("ASQA FULL COLBERT INDEX BUILD: PASS")


if __name__ == "__main__":
    main()
