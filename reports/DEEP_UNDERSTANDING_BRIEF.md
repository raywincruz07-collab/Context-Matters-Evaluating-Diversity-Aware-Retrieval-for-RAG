# Deep Understanding Brief: `medical-rag-maki-colab`

This document is an onboarding brief for contributors who need to understand both the engineering and research shape of the project. It is grounded in the current codebase, the checked-in results, the Sprint 1 report, and the project kick-off slides.

## 1. What This Project Is

As implemented today, this repository is a Sprint 1 baseline medical RAG system for question answering on PubMedQA. Its job is not yet to study diversification directly. Its current purpose is to establish a controlled, non-diversified baseline across multiple retrievers so Sprint 2 can compare MMR, clustering, DPP, and related diversification strategies against a stable reference point.

The research question from the project materials is broader:

> Is diversifying retrieved context actually helpful for LLM reasoning in a RAG pipeline, or does it introduce noise and increase hallucination?

The code in this repository answers only the baseline portion of that question. It measures how well four retrievers perform when the generator, prompt, dataset, and top-k setting are fixed.

## 2. Current Scope vs Planned Scope

### Implemented now

- PubMedQA-derived corpus preparation
- Four interchangeable retrievers behind a common interface
- One fixed generator using the Mannheim Maki OpenAI-compatible API
- One reusable RAG pipeline class
- One Streamlit UI for interactive querying and lightweight evaluation
- One batch-safe evaluation harness for full retriever comparison
- Checked-in raw and summarized Sprint 1 results

### Planned for Sprint 2 and beyond

- Diversification algorithms such as MMR, clustering, and DPP
- Top-k and candidate-pool sweeps
- Faithfulness and hallucination metrics
- Retrieval diversity metrics
- Output diversity and answer-coverage metrics

Those future items are stated in the Sprint 1 report and kick-off materials, but they are not implemented in the current code.

## 3. Architecture: What Actually Runs

There are three main execution paths in the repository.

### A. Interactive Streamlit app

File: `src/app.py`

This is the user-facing demo. It lets a user:

- load the PubMedQA-derived dataset
- build an index for one selected retriever
- configure the Maki API generator
- run a query against the static corpus
- run a custom live PubMed query
- run a small batch evaluation through the UI
- inspect corpus examples and gold documents

Important implementation reality:

- Live PubMed retrieval exists only in the Streamlit app for custom user queries. It is not part of the main batch experiment pipeline and is not used in the checked-in baseline results.

### B. Reusable Python pipeline

File: `src/pipeline.py`

`RAGPipeline` is the cleanest code path if someone wants to script a single-retriever experiment without going through Streamlit. It:

- prepares or loads the dataset
- indexes a chosen retriever
- calls the fixed generator
- exposes `query(...)`, `evaluate(...)`, and `save_results(...)`

This class is useful as the conceptual backbone of the repo, even though the large baseline comparison was run through a different script.

### C. Batch-safe evaluation harness

File: `src/eval_all_retrievers_safe.py`

This is the main experiment runner for Sprint 1. It evaluates multiple retrievers sequentially, writes per-question progress to CSV, and resumes incomplete runs by reading previously saved rows.

This script is what most directly explains the checked-in experiment outputs:

- raw run metadata in `results/raw/EXPERIMENT_METADATA.json`
- one CSV per retriever in `results/raw/`
- summary and ranking CSVs in `results/raw/`
- cleaned final outputs in `results/final_csv_outputs/`

## 4. Data Layer and Contracts

File: `src/data_prep.py`

The project uses `qiaojin/PubMedQA`, specifically the `pqa_labeled` subset. Each PubMedQA example contains a question, a long-form answer, a final decision label, and multiple context sections from a PubMed abstract.

The repo converts that into two artifacts.

### Corpus documents

Saved to `data/corpus.json`

Each context section becomes one retrieval document. The document contract is:

- `doc_id`: integer identifier
- `pubid`: PubMed ID
- `section_label`: section name from the abstract context
- `text`: section text
- `source_question_idx`: index of the originating QA example

In the Streamlit live-PubMed path, documents may also include fields like:

- `source`
- `title`

### QA pairs

Saved to `data/qa_pairs.json`

The QA contract is:

- `qa_id`: integer identifier
- `pubid`: PubMed ID
- `question`: PubMedQA question
- `long_answer`: reference answer
- `final_decision`: yes/no/maybe style label
- `gold_doc_ids`: list of document IDs derived from that question's context sections

Important limitation:

- Gold documents for each question all come from sections of the same paper. That makes the baseline convenient to evaluate, but it constrains later claims about cross-source diversity.

## 5. Retriever Layer

Files:

- `src/retrievers/__init__.py`
- `src/retrievers/factory.py`
- `src/retrievers/bm25_retriever.py`
- `src/retrievers/dense_retriever.py`
- `src/retrievers/dpr_original_retriever.py`
- `src/retrievers/colbert_retriever.py`

All retrievers implement the same interface through `BaseRetriever`:

- `index(corpus)`
- `retrieve(query, top_k=5) -> List[Tuple[Dict, float]]`

That shared contract is one of the repo's strongest design decisions because it makes retriever swapping simple in both the UI and the experiment harness.

### BM25

- Library: `rank_bm25`
- Tokenization: lowercase plus whitespace split
- Index persistence: `data/indices/bm25_index.pkl`
- Retrieval behavior: returns only positively scored documents

This is the sparse lexical baseline. It remains competitive here because PubMedQA questions often share biomedical vocabulary with their relevant context sections.

### Original DPR

- Models:
  - `facebook/dpr-question_encoder-single-nq-base`
  - `facebook/dpr-ctx_encoder-single-nq-base`
- Retrieval score: raw dot product
- Vector store: FAISS `IndexFlatIP`
- Cached embeddings: `data/embeddings/dpr_embeddings.npy`
- Cached index: `data/indices/dpr_faiss.index`

This implementation is deliberately the original dual-encoder DPR setup. It is not cosine similarity, and embeddings are not normalized. The code also includes stale-cache protection because an earlier MiniLM-based substitute existed during development and had to be explicitly excluded for experimental integrity.

### Contriever

- Model: `facebook/contriever`
- Encoding: custom wrapper around Hugging Face `AutoModel`
- Pooling: mean pooling over token embeddings with attention-mask weighting
- Normalization: L2 normalization
- Vector store: FAISS `IndexFlatIP`, which becomes cosine-equivalent after normalization
- Cached embeddings: `data/embeddings/contriever_embeddings.npy`
- Cached index: `data/indices/contriever_faiss.index`

Contriever is implemented to fail loudly if the real model cannot be loaded. That is a good research decision: it avoids silently substituting another encoder and mislabeling the experiment.

### ColBERTv2

- Library: `ragatouille`
- Model: `colbert-ir/colbertv2.0`
- Retrieval style: late interaction with token-level matching
- Index location: `.ragatouille/colbert/indexes/medical_rag_colbert`

This retriever has the heaviest implementation footprint. The code includes two compatibility workarounds:

- a monkey patch for a Transformers tied-weights compatibility issue
- a Windows workaround that injects a dummy `pwd` module before importing RAGatouille

This is a good example of where the repo reflects real research engineering rather than idealized architecture.

## 6. Generator Layer

File: `src/generator.py`

The fixed generator is `MakiGenerator`, which calls the University of Mannheim's OpenAI-compatible endpoint.

Key contract:

- `generate(query, context_docs, temperature=..., max_tokens=...)`
- `generate_streaming(query, context_docs, temperature=..., max_tokens=...)`

Key behavior:

- prompt is built from a fixed template in `src/config.py`
- context documents are concatenated into the prompt
- `/chat/completions` is called on the configured host
- retry logic is built into non-streaming generation
- streaming falls back to non-streaming if the endpoint does not support it

Important implementation reality:

- The repo still contains backward-compatibility naming from an older Groq/Ollama path, such as `GroqGenerator = MakiGenerator` and comments/messages that still mention Groq in places. That should be understood as migration residue, not the current system design.

## 7. Evaluation Layer

File: `src/evaluation/__init__.py`

The current evaluation contract is:

- `evaluate_single(prediction, ground_truth, retrieved_doc_ids, gold_doc_ids)`

This computes:

- `exact_match`
- `f1`
- `rouge_l`
- `recall_at_k`
- `mrr`

Interpretation matters here:

- `Recall@K` and `MRR` are the main Sprint 1 retrieval metrics
- `Exact Match` is effectively uninformative for this long-form generation task
- `F1` and `ROUGE-L` are weak lexical proxies, not faithfulness measures

The Sprint 1 report is explicit that faithfulness, hallucination, coverage, and diversity metrics are deferred. The code agrees with that report.

## 8. Experiment Configuration That Produced The Checked-In Results

The checked-in baseline experiment used a fixed configuration, as reflected in code, metadata, and result CSVs:

- dataset: PubMedQA-derived corpus
- questions: 1,000
- corpus size: 3,358 documents
- retrievers: BM25, Original DPR, Contriever, ColBERTv2
- generator provider: Mannheim Maki OpenAI-compatible API
- model: `ministral-3-14b`
- temperature: `0.0`
- max tokens: `512`
- top-k: `5`

The point of this setup is controlled comparison. Because the generator, prompt, top-k, and dataset are fixed, the main independent variable in Sprint 1 is the retriever.

## 9. Checked-In Results: What They Mean

The current summarized outputs show:

| Rank | Retriever | Recall@5 | MRR | F1 | ROUGE-L |
|---|---|---:|---:|---:|---:|
| 1 | ColBERTv2 | 0.750586 | 0.979450 | 0.233590 | 0.154276 |
| 2 | Contriever | 0.605600 | 0.952733 | 0.230435 | 0.154575 |
| 3 | BM25 | 0.599211 | 0.911517 | 0.224472 | 0.151236 |
| 4 | Original DPR | 0.267880 | 0.474600 | 0.168399 | 0.115600 |

Main conclusion:

- ColBERTv2 is the strongest Sprint 1 baseline retriever by a clear margin on retrieval quality.

Additional takeaways:

- Contriever and BM25 are relatively close on recall, with Contriever ranking relevant evidence earlier on average.
- DPR underperforms badly in this biomedical setting, likely because it is trained for open-domain QA rather than biomedical retrieval.
- Exact Match is 0.0 across retrievers and should not drive decisions here.
- F1 and ROUGE-L move only slightly across retrievers and should not be over-interpreted.

## 10. Research Interpretation

The project materials frame Sprint 1 correctly: it is a baseline-establishment sprint, not a claim about diversity-aware retrieval effectiveness yet.

That distinction is important for future contributors:

- The repo already answers "which baseline retriever should we build on?"
- The repo does not yet answer "does diversification improve grounding, coverage, or hallucination behavior?"

If Sprint 2 work is added on top of this codebase, the most natural insertion point is after candidate retrieval and before final top-k context selection. In practice, that means diversification should sit as a re-ranking or subset-selection layer operating on a candidate pool returned by one of the existing retrievers.

## 11. Practical Risks And Limitations

### Research limitations

- Gold evidence is intra-paper rather than cross-paper, which limits how strongly the dataset can test some notions of retrieval diversity.
- Current generation evaluation is mostly lexical and does not measure factual faithfulness.
- The experiment uses only `top_k=5`; there is no sweep yet over context size.

### Engineering limitations

- Dense retrievers and ColBERT depend on local model downloads and environment compatibility.
- ColBERT has Windows-specific and Transformers-version-specific workarounds.
- The Streamlit app mixes demo behavior, experimentation, and one-off utility paths, so it should not be mistaken for the canonical experiment driver.
- Some UI text and aliases still reflect older generator infrastructure.

## 12. Where Sprint 2 Should Attach

If a contributor is extending this repo for the next sprint, the most likely work items are:

1. Add a diversification layer that can rerank or subset-select from a larger candidate pool.
2. Extend the experiment harness so the independent variables include both retriever choice and diversification method.
3. Add metrics for faithfulness, hallucination, coverage, and retrieval diversity.
4. Run controlled sweeps over top-k, candidate-pool size, and diversification hyperparameters.

The baseline to beat is clear: ColBERTv2 under the fixed Maki generator setup.

## 13. Recommended Mental Model For New Contributors

Think of the current repository as a research baseline scaffold with three stable pieces:

- a deterministic data transformation from PubMedQA into corpus documents plus QA pairs
- a swappable retrieval layer
- a fixed-generator evaluation harness

Everything related to diversity-aware retrieval should be built as a controlled extension around that scaffold, not by changing multiple variables at once.
