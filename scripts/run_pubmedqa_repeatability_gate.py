#!/usr/bin/env python3
"""Run the frozen 3-model PubMedQA generation repeatability gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPOSITORY_ROOT / "src"
for value in (REPOSITORY_ROOT, SRC_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from generation.cli_support import load_model_bindings
from generation.maki import CanonicalMakiAdapter
from generation.repeatability import run_repeatability_gate
from generation.runner import utc_now


CONFIRMATION = "I_UNDERSTAND_THIS_MAKES_180_MODEL_REQUESTS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--model-bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-api-calls", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confirm_api_calls != CONFIRMATION:
        raise SystemExit(f"refusing API execution; pass --confirm-api-calls {CONFIRMATION}")
    bindings = load_model_bindings(args.model_bindings)
    adapters = {
        logical_id: CanonicalMakiAdapter(config)
        for logical_id, config in bindings.items()
    }
    gate = run_repeatability_gate(
        prompt_manifest_path=args.prompt_manifest,
        adapters=adapters,
        output_path=args.output,
        created_at=utc_now(),
    )
    print(gate["gate_id"])
    print(f"passed={gate['scientific_payload']['all_primary_models_passed']}")


if __name__ == "__main__":
    main()
