"""
Original DPR retriever using Facebook's dual-encoder architecture.

Uses two separate encoders as in Karpukhin et al. (2020):
  - facebook/dpr-question_encoder-single-nq-base  (query side)
  - facebook/dpr-ctx_encoder-single-nq-base       (context side)

Similarity: dot product (NOT cosine). Do NOT normalize embeddings.
FAISS: IndexFlatIP (inner product = dot product on raw vectors).
"""

import os
import gc
import numpy as np
import faiss
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple

from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder

from retrievers import BaseRetriever
from config import EMBEDDINGS_DIR, INDEX_DIR, DPR_QUERY_MODEL, DPR_CTX_MODEL


class OriginalDPRRetriever(BaseRetriever):
    def __init__(self):
        super().__init__("dpr")
        self.query_model_name = DPR_QUERY_MODEL   # facebook/dpr-question_encoder-single-nq-base
        self.ctx_model_name   = DPR_CTX_MODEL     # facebook/dpr-ctx_encoder-single-nq-base

        self.q_tokenizer  = None
        self.ctx_tokenizer = None
        self.q_encoder    = None
        self.ctx_encoder  = None

        self.faiss_index   = None
        self.embedding_dim = 768                   # BERT-base hidden size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        if self.q_encoder is not None and self.ctx_encoder is not None:
            return

        print("=" * 70)
        print("Loading ORIGINAL DPR models (facebook dual-encoder)")
        print(f"  Question encoder : {self.query_model_name}")
        print(f"  Context encoder  : {self.ctx_model_name}")
        print(f"  Device           : {self.device}")
        print("=" * 70)

        self.q_tokenizer   = AutoTokenizer.from_pretrained(self.query_model_name)
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(self.ctx_model_name)

        self.q_encoder  = DPRQuestionEncoder.from_pretrained(self.query_model_name).to(self.device)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(self.ctx_model_name).to(self.device)

        self.q_encoder.eval()
        self.ctx_encoder.eval()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_questions(self, questions: List[str], batch_size: int = 16) -> np.ndarray:
        self._load_models()
        all_emb = []

        for start in range(0, len(questions), batch_size):
            batch = questions[start : start + batch_size]
            inputs = self.q_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = self.q_encoder(**inputs).pooler_output   # (B, 768)

            all_emb.append(emb.cpu().numpy().astype("float32"))

        return np.vstack(all_emb)

    def _encode_contexts(self, corpus: List[Dict], batch_size: int = 16) -> np.ndarray:
        self._load_models()
        all_emb = []

        texts  = [str(doc.get("text",  "")) for doc in corpus]
        titles = [""] * len(texts)            # PubMedQA has no separate titles

        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding DPR contexts"):
            batch_titles = titles[start : start + batch_size]
            batch_texts  = texts[start  : start + batch_size]

            # DPR context encoder takes (title, text) as a token-type-separated pair
            inputs = self.ctx_tokenizer(
                batch_titles,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = self.ctx_encoder(**inputs).pooler_output  # (B, 768)

            all_emb.append(emb.cpu().numpy().astype("float32"))

        return np.vstack(all_emb)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, corpus: List[Dict]):
        self.corpus = corpus

        emb_path = os.path.join(EMBEDDINGS_DIR, "dpr_embeddings.npy")
        idx_path  = os.path.join(INDEX_DIR,      "dpr_faiss.index")

        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR,      exist_ok=True)

        embeddings = None

        # Load cached embeddings only if shape matches (guards against stale MiniLM cache)
        if os.path.exists(emb_path):
            loaded = np.load(emb_path)
            if loaded.shape == (len(corpus), self.embedding_dim):
                print(f"Loading cached DPR embeddings: {loaded.shape}")
                embeddings = loaded
            else:
                print(f"Stale cache found (shape {loaded.shape}). Recomputing...")

        if embeddings is None:
            embeddings = self._encode_contexts(corpus, batch_size=16)
            np.save(emb_path, embeddings)
            print(f"Saved DPR embeddings: {emb_path}  shape={embeddings.shape}")

        # Build or load FAISS index
        rebuild = True
        if os.path.exists(idx_path):
            idx = faiss.read_index(idx_path)
            if idx.d == embeddings.shape[1] and idx.ntotal == len(corpus):
                print("Loading cached DPR FAISS index.")
                self.faiss_index = idx
                rebuild = False
            else:
                print(f"Stale FAISS index (d={idx.d}, n={idx.ntotal}). Rebuilding...")

        if rebuild:
            # Inner product = dot product (DPR is NOT cosine; do NOT normalize)
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index.add(embeddings)
            faiss.write_index(self.faiss_index, idx_path)
            print(f"Saved DPR FAISS index: {idx_path}")

        self.is_indexed = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index(corpus) first.")

        query_emb = self._encode_questions([query], batch_size=1)
        scores, indices = self.faiss_index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.corpus[int(idx)], float(score)))

        return results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload_model(self):
        self.q_tokenizer   = None
        self.ctx_tokenizer = None
        self.q_encoder     = None
        self.ctx_encoder   = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
