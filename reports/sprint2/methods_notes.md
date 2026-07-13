# Sprint 2 Methods — Reference Notes

## Dataset
BEIR HotpotQA (BeIR/hotpotqa corpus/queries/qrels, HuggingFace).
Option B subsampling: 500 test queries (seed=42) sampled from the 7,405-query
BEIR test split. Corpus built from 997 unique gold passages (~2.0/query)
plus 50 random negatives per query (excluded from gold set, seed=42),
drawn from the full 5.23M-passage BEIR corpus. Final corpus: 25,997 documents.
BEIR HotpotQA qrels have no reference answers, so EM/F1/ROUGE-L were not
computed for Sprint 2 — evaluation relies on Recall@5, MRR, retrieval
diversity, and NLI faithfulness.

## Retrievers (unchanged from Sprint 1)
BM25, DPR (Karpukhin et al. 2020), Contriever, ColBERTv2 (via RAGatouille,
Stanford ColBERT backend, pinned to ragatouille<0.0.10 for Colab
compatibility).

## Diversification
MMR reranking. Candidate pool: top-20 by native retriever relevance score,
reranked via MMR to top-5. Diversity term: shared Contriever embedding
cosine similarity, consistent across all 4 retrievers. λ sweep:
{0.0, 0.25, 0.5, 0.75, 1.0} + "none" baseline = 6 conditions per retriever.
Sanity check: λ=1.0 confirmed mathematically identical to "none" in all
24 result rows (diversity term correctly zeroed).

## New Sprint 2 metrics
- Retrieval diversity: mean pairwise cosine distance among top-5 retrieved
  docs' Contriever embeddings.
- Faithfulness (NLI): entailment probability between generated answer
  (hypothesis) and each retrieved context doc (premise), facebook/bart-large-mnli,
  max score across the 5 context docs.

## Generator (fixed across all conditions)
qwen3.5-122b, Mannheim Maki API, temperature=0.0, max_tokens=512.

## Experimental grid
4 retrievers × 6 conditions × 500 questions = 24 combinations,
12,000 total generations. Total errors across all 24 combinations: 0.

## Core finding
Diversification (decreasing λ) monotonically reduces both Recall@5 and
faithfulness NLI score, consistently across all 4 retrievers. Relative
Recall@5 drop from λ=1.0 to λ=0.0: BM25 33.6%, DPR 38.7%, Contriever 33.1%,
ColBERTv2 40.2%.

## Cross-sprint comparison (Sprint 1 PubMedQA baseline vs Sprint 2 BEIR baseline)
DPR: Recall@5 rose from 0.268 (Sprint 1) to 0.648 (Sprint 2) — attributed to
domain mismatch in Sprint 1 (DPR's NQ-trained encoders vs PubMedQA's medical
text) rather than a general DPR weakness; BEIR HotpotQA's Wikipedia-based
text is closer to DPR's original training distribution. Other retrievers
(BM25, Contriever, ColBERTv2) maintained consistent relative rankings across
both sprints.
