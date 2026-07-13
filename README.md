# MedRAG — Diversity-Aware Retrieval-Augmented Generation Pipeline

**Team Project: Context Matters — Evaluating Diversity-Aware Retrieval for RAG**  
University of Mannheim | Chair of Data Science | Supervisor: Andreea Iana

## Overview

A two-sprint RAG evaluation pipeline comparing 4 retrieval strategies (BM25, DPR,
Contriever, ColBERTv2) with and without MMR diversification, evaluated on two
datasets: PubMedQA (Sprint 1) and BEIR HotpotQA (Sprint 2). Generator is fixed
across all conditions: Mannheim Maki API (qwen3.5-122b, temperature=0.0, max_tokens=512).

---

## Sprint 1 — PubMedQA Baseline

**Dataset:** PubMedQA-derived corpus — 3,358 documents, 1,000 questions.  
**Model:** ministral-3-14b via Mannheim Maki API.

| Rank | Retriever | Recall@5 | MRR | F1 | ROUGE-L |
|------|-----------|----------|-----|----|---------|
| 1 | **ColBERTv2** | 0.7506 | 0.9795 | 0.2336 | 0.1543 |
| 2 | **Contriever** | 0.6056 | 0.9527 | 0.2297 | 0.1542 |
| 3 | **BM25** | 0.5992 | 0.9115 | 0.2244 | 0.1512 |
| 4 | **DPR** | 0.2679 | 0.4746 | 0.1689 | 0.1162 |

Results in [`results/sprint1/`](results/sprint1/).

---

## Sprint 2 — BEIR HotpotQA + MMR Diversification

**Dataset:** BEIR HotpotQA, Option B subsampling — 500 test queries (seed=42).
Corpus: 997 unique gold passages + 50 random negatives per query = **25,997 docs total**.
BEIR qrels contain no free-text answers; EM/F1/ROUGE-L not computed — evaluation
uses Recall@5, MRR, retrieval diversity, and faithfulness NLI.

**Experimental grid:** 4 retrievers × 6 MMR conditions (λ ∈ {none, 0.0, 0.25, 0.5, 0.75, 1.0})
= 24 combinations, 12,000 total generations, **0 errors**.

**Model:** qwen3.5-122b via Mannheim Maki API.

### Core Finding

Diversification (decreasing λ) **monotonically reduces** both Recall@5 and
faithfulness NLI across all 4 retrievers. Relative Recall@5 drop from λ=1.0 → λ=0.0:

| Retriever | Recall@5 drop |
|-----------|--------------|
| BM25 | 33.6% |
| DPR | 38.7% |
| Contriever | 33.1% |
| ColBERTv2 | 40.2% |

### Cross-Sprint Comparison

DPR Recall@5 rose from 0.268 (Sprint 1, PubMedQA) to 0.648 (Sprint 2, BEIR) —
consistent with DPR's NQ/Wikipedia training distribution matching HotpotQA better
than biomedical text.

### New Sprint 2 Metrics

- **Retrieval diversity:** mean pairwise cosine distance among top-5 retrieved docs'
  Contriever embeddings.
- **Faithfulness NLI:** entailment probability (facebook/bart-large-mnli) between
  generated answer (hypothesis) and each retrieved context doc (premise); max across 5 docs.

Results in [`results/sprint2/`](results/sprint2/).

---

## Project Structure

```
medical-rag-maki-colab/
├── src/
│   ├── config.py                       # All constants (Sprint 1 + Sprint 2)
│   ├── data_prep.py                    # Sprint 1: PubMedQA corpus builder
│   ├── data_prep_hotpot.py             # Sprint 2: HotpotQA distractor corpus
│   ├── data_prep_hotpot_beir.py        # Sprint 2: BEIR HotpotQA Option B corpus
│   ├── eval_all_retrievers_safe.py     # Resumable eval harness (Sprint 1 + Sprint 2)
│   ├── pipeline.py                     # RAG pipeline with diversification param
│   ├── generator.py                    # Maki API generator
│   ├── app.py                          # Streamlit demo UI
│   ├── pubmed_fetch.py                 # Live PubMed fetcher (for app.py)
│   ├── retrievers/
│   │   ├── __init__.py                 # BaseRetriever interface
│   │   ├── factory.py                  # Retriever factory
│   │   ├── bm25_retriever.py           # BM25 (sparse, rank-bm25)
│   │   ├── dpr_original_retriever.py   # Original Facebook DPR + FAISS
│   │   ├── dense_retriever.py          # Contriever + FAISS
│   │   └── colbert_retriever.py        # ColBERTv2 via RAGatouille (Colab-patched)
│   ├── diversification/
│   │   ├── __init__.py
│   │   └── mmr.py                      # MMR reranking + corpus embedding cache
│   └── evaluation/
│       └── __init__.py                 # Metrics: Recall@K, MRR, EM, F1, ROUGE-L,
│                                       #          retrieval_diversity, faithfulness_nli
├── tests/
│   └── test_mmr.py                     # 11 MMR unit tests (all passing)
├── results/
│   ├── sprint1/
│   │   ├── raw/                        # Sprint 1 per-retriever CSVs (4 retrievers)
│   │   └── final_csv_outputs/          # Sprint 1 formatted summary outputs
│   └── sprint2/
│       ├── raw/                        # 24 Sprint 2 CSVs (4 retrievers × 6 conditions)
│       ├── summary/                    # Master summary CSV
│       └── graphs/                     # 5 analysis graphs (PNG)
├── reports/
│   ├── DEEP_UNDERSTANDING_BRIEF.md     # Sprint 1 contributor/architecture guide
│   └── sprint2/
│       ├── methods_notes.md            # Sprint 2 methodology reference
│       └── SUBMISSION_CHECKLIST.md     # Sprint 2 submission status
├── notebooks/
│   └── Sprint1_Baseline_RAG_Evaluation.ipynb
├── create_colab_zip.py                 # Builds medical-rag-maki-colab-sprint2.zip
├── download_hotpotqa.py                # Downloads HotpotQA distractor dataset
└── requirements.txt
```

---

## Running the Evaluation

### Environment

```bash
export MAKI_API_KEY="your_key_here"          # Linux/Mac/Git Bash
$env:MAKI_API_KEY = "your_key_here"          # Windows PowerShell
```

`MAKI_HOST` and `MAKI_MODEL` default to `https://maki.uni-mannheim.de/v1` and
`qwen3.5-122b` respectively. Override via env vars if needed.

### Sprint 1 — PubMedQA

```bash
cd src
python eval_all_retrievers_safe.py --top_k 5 --with-generation
```

### Sprint 2 — BEIR HotpotQA (Option B)

```bash
cd src
python eval_all_retrievers_safe.py --beir --top_k 5 --with-generation
```

Data is downloaded automatically from HuggingFace on first run
(`BeIR/hotpotqa` corpus/queries/qrels).

### Colab Deployment

```bash
python create_colab_zip.py   # produces medical-rag-maki-colab-sprint2.zip
```

Upload the zip to Colab, extract, and run from `src/`.

---

## References

- Carbonell & Goldstein (1998). The Use of MMR, Diversity-Based Reranking for Reordering Documents. SIGIR 1998.
- Jin et al. (2019). PubMedQA: A Dataset for Biomedical Research QA. EMNLP 2019.
- Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA. EMNLP 2020.
- Izacard & Grave (2022). Unsupervised Dense Information Retrieval with Contrastive Learning. TMLR 2022.
- Khattab & Zaharia (2020). ColBERT: Efficient and Effective Passage Search. SIGIR 2020.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
- Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models. NeurIPS 2021.
