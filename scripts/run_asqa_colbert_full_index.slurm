#!/bin/bash
#SBATCH --job-name=asqa_colbert_full
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00

set -euo pipefail

WS=$(ws_find context_matters_sprint3)
REPO="$WS/context-matters-rag-sprint3"

module load devel/cuda/12.8

export HF_HOME="$WS/.hf-cache"
export TORCH_EXTENSIONS_DIR="$WS/.torch_extensions"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME"
mkdir -p "$TORCH_EXTENSIONS_DIR"
mkdir -p "$WS/logs"

echo "=== SLURM JOB ==="
echo "job id: ${SLURM_JOB_ID:-unknown}"
echo "node: $(hostname)"
echo "start: $(date --iso-8601=seconds)"
echo

nvidia-smi -L

echo
echo "=== ENVIRONMENT ==="
"$WS/envs/colbert/bin/python" - <<'PY'
import sys
import torch
import transformers
import faiss
import importlib.metadata

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("faiss:", faiss.__version__)
print("colbert-ai:", importlib.metadata.version("colbert-ai"))
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY

echo
echo "=== START INDEXING ==="

"$WS/envs/colbert/bin/python" \
    "$REPO/scripts/build_asqa_colbert_full_index.py"

echo
echo "=== JOB FINISHED ==="
echo "end: $(date --iso-8601=seconds)"
