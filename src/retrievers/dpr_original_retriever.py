"""Original DPR dual-encoder runtime backed by authoritative DPRConfig."""

import gc
import os
from collections.abc import Mapping
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

from config import EMBEDDINGS_DIR, INDEX_DIR
from retrievers import BaseRetriever
from retrievers.dpr_config import DPR_CONFIG, DPRConfig


class OriginalDPRRetriever(BaseRetriever):
    def __init__(self, config: DPRConfig = DPR_CONFIG):
        super().__init__("dpr")
        if not isinstance(config, DPRConfig):
            raise TypeError("config must be a DPRConfig")
        self.config = config
        self._validate_runtime_config()

        self.q_tokenizer = None
        self.ctx_tokenizer = None
        self.q_encoder = None
        self.ctx_encoder = None

        self.faiss_index = None
        self.embedding_dim = config.embedding_dimension
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _validate_runtime_config(self) -> None:
        """Reject scientific modes this frozen DPR runtime does not implement."""
        supported = {
            "query_preprocessing": "exact query string passed to tokenizer",
            "document_preprocessing": (
                "exact passage text passed to tokenizer paired with empty title"
            ),
            "paired_truncation_strategy": (
                "delegated to tokenizer/Transformers default"
            ),
            "padding_semantics": (
                "dynamic padding to longest encoded input in each batch"
            ),
            "context_title_policy": "empty title paired with passage text",
            "representation": "pooler_output",
            "embedding_dtype": "float32",
            "normalization": "none",
            "score_semantics": (
                "raw dot product / inner product of unnormalized embeddings"
            ),
            "ranking_direction": "higher score is better",
            "index_type": "faiss.IndexFlatIP",
            "score_sign_filtering": "none",
        }
        for field, expected in supported.items():
            if getattr(self.config, field) != expected:
                raise ValueError(
                    f"unsupported DPR {field}: {getattr(self.config, field)!r}"
                )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        if self.q_encoder is not None and self.ctx_encoder is not None:
            return

        print("=" * 70)
        print("Loading ORIGINAL DPR models (facebook dual-encoder)")
        print(f"  Question encoder : {self.config.question_model_id}")
        print(f"  Context encoder  : {self.config.context_model_id}")
        print(f"  Device           : {self.device}")
        print("=" * 70)

        self.q_tokenizer = AutoTokenizer.from_pretrained(
            self.config.question_tokenizer_id,
            revision=self.config.question_tokenizer_revision,
        )
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(
            self.config.context_tokenizer_id,
            revision=self.config.context_tokenizer_revision,
        )

        self.q_encoder = DPRQuestionEncoder.from_pretrained(
            self.config.question_model_id,
            revision=self.config.question_model_revision,
        ).to(self.device)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(
            self.config.context_model_id,
            revision=self.config.context_model_revision,
        ).to(self.device)

        self.q_encoder.eval()
        self.ctx_encoder.eval()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_questions(
        self, questions: List[str], batch_size: int | None = None
    ) -> np.ndarray:
        self._load_models()
        all_emb = []
        effective_batch_size = batch_size or self.config.query_batch_size

        for start in range(0, len(questions), effective_batch_size):
            batch = questions[start : start + effective_batch_size]
            inputs = self.q_tokenizer(
                batch,
                padding=True,
                truncation=self.config.truncation_enabled,
                max_length=self.config.query_max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = self._extract_representation(self.q_encoder(**inputs))

            all_emb.append(
                emb.cpu().numpy().astype(self.config.embedding_dtype, copy=False)
            )

        return np.vstack(all_emb)

    def _encode_contexts(
        self, corpus: List[Dict], batch_size: int | None = None
    ) -> np.ndarray:
        texts = []
        for position, document in enumerate(corpus):
            if not isinstance(document, Mapping):
                raise TypeError(f"corpus document {position} must be a mapping")
            if "retrieval_content" not in document:
                raise ValueError(
                    f"corpus document {position} is missing retrieval_content"
                )
            retrieval_content = document["retrieval_content"]
            if not isinstance(retrieval_content, str):
                raise TypeError(
                    f"corpus document {position} retrieval_content must be a string"
                )
            texts.append(retrieval_content)

        self._load_models()
        all_emb = []
        effective_batch_size = batch_size or self.config.context_batch_size
        titles = [""] * len(texts)

        for start in tqdm(
            range(0, len(texts), effective_batch_size),
            desc="Encoding DPR contexts",
        ):
            batch_titles = titles[start : start + effective_batch_size]
            batch_texts = texts[start : start + effective_batch_size]

            # DPR context encoder takes (title, text) as a token-type-separated pair
            inputs = self.ctx_tokenizer(
                batch_titles,
                batch_texts,
                padding=True,
                truncation=self.config.truncation_enabled,
                max_length=self.config.context_max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                emb = self._extract_representation(self.ctx_encoder(**inputs))

            all_emb.append(
                emb.cpu().numpy().astype(self.config.embedding_dtype, copy=False)
            )

        return np.vstack(all_emb)

    def _extract_representation(self, encoder_output):
        if self.config.representation != "pooler_output":
            raise ValueError(
                f"unsupported DPR representation: {self.config.representation!r}"
            )
        return encoder_output.pooler_output

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, corpus: List[Dict]):
        self.corpus = corpus

        emb_path = os.path.join(EMBEDDINGS_DIR, "dpr_embeddings.npy")
        idx_path = os.path.join(INDEX_DIR, "dpr_faiss.index")

        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)

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
            embeddings = self._encode_contexts(corpus)
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
            self.faiss_index = faiss.IndexFlatIP(self.config.embedding_dimension)
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

        query_emb = self._encode_questions([query])
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
        self.q_tokenizer = None
        self.ctx_tokenizer = None
        self.q_encoder = None
        self.ctx_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
