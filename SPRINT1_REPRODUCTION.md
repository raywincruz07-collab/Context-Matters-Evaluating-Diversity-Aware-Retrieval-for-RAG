# Sprint 1 Reproduction Guide

This guide details how to reproduce the Sprint 1 baseline RAG evaluation for the University of Mannheim "Context Matters" team project.

## 1. Install Dependencies

Ensure you have a Python 3.10+ environment, then install the required packages:

```bash
pip install -r requirements.txt
```

If you wish to use ColBERTv2, you must also install `ragatouille`:
```bash
pip install ragatouille
```

## 2. Set API Key

The evaluation uses the Mannheim Maki OpenAI-compatible API for generation. You must set the `MAKI_API_KEY` environment variable before running the evaluation.

**In Colab (Python):**
```python
import os
from getpass import getpass

os.environ["MAKI_API_KEY"] = getpass("Paste Mannheim Maki API key: ")
os.environ["MAKI_HOST"] = "https://maki.uni-mannheim.de/v1"
os.environ["MAKI_MODEL"] = "ministral-3-14b"
os.environ["MAKI_DEFAULT_CTX"] = "7680"
```

## 3. Verify Original DPR

To confirm that the Original DPR implementation (Facebook dual-encoder) is correctly configured, run:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from retrievers.factory import get_retriever; r=get_retriever('dpr'); print(type(r))"
```

**Expected output:**
```
<class 'retrievers.dpr_original_retriever.OriginalDPRRetriever'>
```

> **Note:** The final DPR embeddings are saved to `data/embeddings/dpr_embeddings.npy` and the shape must be exactly `(3358, 768)`. The script will automatically reject cached embeddings with incorrect shapes.

## 4. Run 5-Question Smoke Test

To ensure the pipeline is working end-to-end, run a quick smoke test with 5 questions:

```bash
python src/eval_all_retrievers_safe.py \
  --top_k 5 \
  --retrievers bm25 dpr contriever colbertv2 \
  --with-generation \
  --limit 5 \
  --results_dir "/content/drive/MyDrive/medical_rag_sprint1_results/test_after_source_fix"
```

## 5. Run Full Evaluation

Run the full 1,000-question evaluation across all retrievers:

```bash
python src/eval_all_retrievers_safe.py --with-generation --results_dir results/raw
```

## 6. Resuming After Disconnect (Colab)

The `eval_all_retrievers_safe.py` script automatically saves intermediate results after every question.
If your Colab session disconnects or the script fails, simply re-run the exact same command:

```bash
python src/eval_all_retrievers_safe.py --with-generation --results_dir results/raw
```

The script will read the existing CSVs in `results/raw/`, skip completed questions, and resume from where it left off.
