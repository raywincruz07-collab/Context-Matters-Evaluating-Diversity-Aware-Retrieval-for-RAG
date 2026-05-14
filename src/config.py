"""
Configuration for Medical RAG Pipeline
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "indices")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# Ensure directories exist
for d in [DATA_DIR, INDEX_DIR, EMBEDDINGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Dataset
DATASET_NAME = "qiaojin/PubMedQA"
DATASET_SUBSET = "pqa_labeled"  # 1,000 expert-labeled examples
CHUNK_SIZE = 256  # tokens per chunk (for longer abstracts)

# Retriever settings
RETRIEVER_CHOICES = ["bm25", "dpr", "contriever", "colbertv2"]
TOP_K = 5  # number of documents to retrieve

# DPR-specific
DPR_QUERY_MODEL = "facebook/dpr-question_encoder-single-nq-base"
DPR_CTX_MODEL = "facebook/dpr-ctx_encoder-single-nq-base"

# Contriever-specific
CONTRIEVER_MODEL = "facebook/contriever"

# ColBERTv2
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLBERT_INDEX_NAME = "medical_rag_colbert"

# Generator: University GPU-backed LiteLLM/OpenAI-compatible endpoint
# Do not hardcode keys. Set MAKI_API_KEY in PowerShell or paste it in the Streamlit sidebar.
MAKI_API_KEY = os.environ.get("MAKI_API_KEY", "")
MAKI_HOST = os.environ.get("MAKI_HOST", "https://maki.uni-mannheim.de/v1")
MAKI_MODEL = os.environ.get("MAKI_MODEL", "ministral-3-14b")
MAKI_DEFAULT_CTX = int(os.environ.get("MAKI_DEFAULT_CTX", "7680"))

# Backward compatibility for old code paths
GROQ_API_KEY = MAKI_API_KEY
GROQ_MODEL = MAKI_MODEL

# Prompt template
PROMPT_TEMPLATE = """You are a helpful medical assistant. Answer the question based ONLY on the provided context. 
If the context doesn't contain enough information, say "I cannot answer this based on the provided context."

Context:
{context}

Question: {question}

Answer:"""

# Evaluation
EVAL_METRICS = ["exact_match", "f1", "rouge_l"]
