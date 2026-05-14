"""
ColBERTv2 retriever using RAGatouille library.

ColBERTv2 uses late interaction: it encodes queries and documents into
multi-vector representations and scores via MaxSim (maximum similarity
between token-level embeddings).

Note: ColBERT indexing is heavier than dense retrievers. On 8GB RAM,
use a smaller corpus subset or precomputed index.
"""

import os
from typing import List, Dict, Tuple

from retrievers import BaseRetriever
from config import COLBERT_INDEX_NAME


class ColBERTRetriever(BaseRetriever):
    def __init__(self):
        super().__init__("colbertv2")
        self.rag = None
        self.index_path = os.path.join(
            ".ragatouille",
            "colbert",
            "indexes",
            COLBERT_INDEX_NAME
        )

    def _patch_transformers_for_colbert(self):
        try:
            from transformers.modeling_utils import PreTrainedModel

            def safe_mark_tied_weights_as_initialized(self):
                if not hasattr(self, "all_tied_weights_keys"):
                    self.all_tied_weights_keys = {}
                return None

            PreTrainedModel.mark_tied_weights_as_initialized = safe_mark_tied_weights_as_initialized
            print("Applied Transformers compatibility patch for ColBERTv2.")

        except Exception as e:
            print("Could not patch Transformers for ColBERTv2.")
            print("Patch error:", e)

    def index(self, corpus: List[Dict]):
        """Build ColBERTv2 index using RAGatouille."""
        self.corpus = corpus

        try:
            self._patch_transformers_for_colbert()
            import sys
            if os.name == 'nt' and 'pwd' not in sys.modules:
                import types
                sys.modules['pwd'] = types.ModuleType('pwd')
            from ragatouille import RAGPretrainedModel

            if os.path.exists(self.index_path):
                print("Loading existing ColBERTv2 index...")
                self.rag = RAGPretrainedModel.from_index(self.index_path)
            else:
                print("Building ColBERTv2 index (this will take a while on CPU)...")
                self.rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

                documents = [doc["text"] for doc in corpus]
                doc_ids = [str(doc["doc_id"]) for doc in corpus]

                self.rag.index(
                    collection=documents,
                    document_ids=doc_ids,
                    index_name=COLBERT_INDEX_NAME,
                    max_document_length=256,
                    split_documents=False,
                )
                print("ColBERTv2 index built.")

            self.is_indexed = True

        except ImportError as ie:
            import traceback
            traceback.print_exc()
            print(f"ImportError details: {ie}")
            print("RAGatouille not installed. Install with: pip install ragatouille")
            print("ColBERTv2 retriever will not be available.")
        except Exception as e:
            print(f"Error building ColBERTv2 index: {e}")
            print("ColBERTv2 retriever will not be available.")
            print("This is often due to memory constraints on 8GB RAM.")
            print("Consider using BM25, DPR, or Contriever instead.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve using ColBERTv2 late interaction."""
        if not self.is_indexed or self.rag is None:
            raise RuntimeError("ColBERTv2 index not available. Call index() first or check errors above.")

        results = self.rag.search(query=query, k=top_k)

        output = []
        for result in results:
            doc_id = int(result["document_id"])
            score = float(result["score"])
            # Find the matching corpus document
            doc = next((d for d in self.corpus if d["doc_id"] == doc_id), None)
            if doc:
                output.append((doc, score))

        return output
