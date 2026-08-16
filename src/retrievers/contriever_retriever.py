"""Canonical in-memory Contriever runtime backed by ContrieverConfig."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from retrievers import BaseRetriever
from retrievers.contriever_config import CONTRIEVER_CONFIG, ContrieverConfig
from retrieval_artifacts.contriever_cache_identity import (
    build_contriever_cache_identity,
)
from retrieval_artifacts.runtime_corpus import (
    contriever_runtime_documents_from_corpus_records,
)


class ContrieverRetriever(BaseRetriever):
    """Pinned, unnormalized Contriever retrieval with an in-memory CPU index."""

    def __init__(self, config: ContrieverConfig = CONTRIEVER_CONFIG):
        super().__init__("contriever")
        if not isinstance(config, ContrieverConfig):
            raise TypeError("config must be a ContrieverConfig")
        self.config = config
        self._validate_runtime_config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.faiss_index = None
        self.embedding_dim = config.embedding_dimension
        self.cache_identity = None

    def _validate_runtime_config(self) -> None:
        supported = {
            "model_loader": "AutoModel",
            "tokenizer_loader": "AutoTokenizer",
            "query_preprocessing": (
                "exact query string passed to tokenizer; no manual lowercasing or "
                "external text normalization; tokenizer-native behavior preserved"
            ),
            "document_preprocessing": (
                "exact validated CorpusRecord.retrieval_content passed to tokenizer; "
                "no manual lowercasing or external text normalization; "
                "tokenizer-native behavior preserved"
            ),
            "pooling": "attention-mask-aware mean pooling of last_hidden_state",
            "truncation_enabled": True,
            "truncation_semantics": (
                "single-sequence truncation delegated to tokenizer/Transformers"
            ),
            "padding_semantics": (
                "dynamic padding to longest encoded input in each batch"
            ),
            "normalization": "none",
            "compute_dtype": "float32",
            "autocast": False,
            "embedding_dtype": "float32",
            "score_semantics": (
                "raw dot product / inner product of unnormalized embeddings"
            ),
            "ranking_direction": "higher score is better",
            "index_type": "faiss.IndexFlatIP",
            "index_device": "cpu",
            "score_sign_filtering": "none",
        }
        for field, expected in supported.items():
            if getattr(self.config, field) != expected:
                raise ValueError(
                    f"unsupported Contriever {field}: "
                    f"{getattr(self.config, field)!r}"
                )

    def _load_model(self) -> None:
        """Lazily load the exact pinned tokenizer and float32 model."""
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_id,
            revision=self.config.tokenizer_revision,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            revision=self.config.model_revision,
        )
        self.model = self.model.to(self.device)
        self.model = self.model.float()
        self.model.eval()

    @staticmethod
    def _mean_pool(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        if not hasattr(model_output, "last_hidden_state"):
            raise ValueError("Contriever model output must contain last_hidden_state")
        last_hidden = model_output.last_hidden_state
        if not isinstance(last_hidden, torch.Tensor):
            raise TypeError("last_hidden_state must be a torch.Tensor")
        if last_hidden.ndim != 3:
            raise ValueError("last_hidden_state must have rank 3")
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor")
        if attention_mask.shape != last_hidden.shape[:2]:
            raise ValueError("attention_mask shape must match last_hidden_state")
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        denominator = mask.sum(dim=1)
        if torch.any(denominator == 0):
            raise ValueError("Contriever attention mask contains a zero-token row")
        return (last_hidden * mask).sum(dim=1) / denominator

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be an ordered sequence of strings")
        if not texts:
            raise ValueError("texts must not be empty")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must contain only strings")

        self._load_model()
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=self.config.truncation_enabled,
                max_length=max_length,
                return_tensors="pt",
            )
            if "attention_mask" not in inputs:
                raise ValueError("Contriever tokenizer output requires attention_mask")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = self.model(**inputs)
                pooled = self._mean_pool(output, inputs["attention_mask"])
            if pooled.dtype != torch.float32:
                raise ValueError("Contriever pooled embeddings must be float32")
            chunks.append(pooled.cpu().numpy())

        embeddings = np.vstack(chunks)
        self._validate_embeddings(embeddings, document_count=len(texts))
        return embeddings

    def _encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode_texts(
            queries,
            batch_size=self.config.query_batch_size,
            max_length=self.config.query_max_length,
        )

    def _encode_documents(self, documents: Sequence[Mapping]) -> np.ndarray:
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be an ordered sequence")
        texts = []
        for position, document in enumerate(documents):
            if not isinstance(document, Mapping):
                raise TypeError(f"corpus document {position} must be a mapping")
            if "retrieval_content" not in document:
                raise ValueError(
                    f"corpus document {position} is missing retrieval_content"
                )
            content = document["retrieval_content"]
            if not isinstance(content, str):
                raise TypeError(
                    f"corpus document {position} retrieval_content must be a string"
                )
            texts.append(content)
        return self._encode_texts(
            texts,
            batch_size=self.config.document_batch_size,
            max_length=self.config.document_max_length,
        )

    def _validate_embeddings(self, embeddings, *, document_count: int) -> None:
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Contriever embeddings must be a numpy.ndarray")
        expected_shape = (document_count, self.config.embedding_dimension)
        if embeddings.ndim != 2 or embeddings.shape != expected_shape:
            raise ValueError(f"Contriever embedding shape must be {expected_shape}")
        if embeddings.dtype != np.dtype(self.config.embedding_dtype):
            raise ValueError(
                f"Contriever embedding dtype must be {self.config.embedding_dtype}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Contriever embeddings must contain only finite values")

    def _validate_faiss_index(self, index, *, document_count: int) -> None:
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Contriever FAISS index must be faiss.IndexFlatIP")
        if index.d != self.config.embedding_dimension:
            raise ValueError("Contriever FAISS dimension does not match config")
        if index.ntotal != document_count:
            raise ValueError("Contriever FAISS ntotal does not match corpus")

    def index(self, corpus: List[Dict]):
        raise ValueError(
            "Contriever indexing requires index_from_corpus_records() with a "
            "validated CorpusManifest and CorpusRecords"
        )

    def index_from_corpus_records(self, *, corpus_manifest, corpus_records) -> None:
        runtime_documents = contriever_runtime_documents_from_corpus_records(
            corpus_manifest=corpus_manifest,
            corpus_records=corpus_records,
        )
        cache_identity = build_contriever_cache_identity(
            corpus_manifest=corpus_manifest,
            contriever_config=self.config,
        )
        embeddings = self._encode_documents(runtime_documents)
        self._validate_embeddings(
            embeddings,
            document_count=corpus_manifest.document_count,
        )
        index = faiss.IndexFlatIP(self.config.embedding_dimension)
        index.add(embeddings)
        self._validate_faiss_index(
            index,
            document_count=corpus_manifest.document_count,
        )

        self.corpus = runtime_documents
        self.cache_identity = cache_identity
        self.faiss_index = index
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index_from_corpus_records() first.")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be a positive non-boolean integer")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self._encode_queries([query])
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if int(index) == -1:
                continue
            results.append((self.corpus[int(index)], float(score)))
        return results

    def unload_model(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
