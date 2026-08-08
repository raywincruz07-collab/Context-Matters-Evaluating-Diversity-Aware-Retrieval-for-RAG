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
DPR: Recall@5 rose from 0.268 (Sprint 1) to 0.648 (Sprint 2) -- attributed to
domain mismatch in Sprint 1 (DPR's NQ-trained encoders vs PubMedQA's medical
text) rather than a general DPR weakness; BEIR HotpotQA's Wikipedia-based
text is closer to DPR's original training distribution. Other retrievers
(BM25, Contriever, ColBERTv2) maintained consistent relative rankings across
both sprints.

## Clustering diversification (deck Sub-task 2.2)
Two methods: k-means and agglomerative (sklearn). Candidates are grouped into
n_clusters clusters based on their Contriever L2-normalised embeddings. The
most relevant document from each cluster is selected first, then remaining
slots are filled by round-robin across clusters. n_clusters=1 degenerates to
pure relevance top-k. Conditions evaluated: kmeans_k2, kmeans_k3, kmeans_k5,
agglo_k3, agglo_k5 (retrieval-only grid); kmeans_k3 also received
generation + faithfulness evaluation. Implementation: src/diversification/clustering.py.

## DPP diversification (deck Sub-task 2.3)
Kernel: L = Diag(q) * S * Diag(q), where q_i = exp(theta*(rel_i - max(rel))/2)
are quality scores in (0,1] and S is the cosine Gram matrix of L2-normalised
Contriever embeddings. Reference: Kulesza & Taskar (2012), "Determinantal
Point Processes for Machine Learning."

Two modes:
- map (greedy MAP): iterative Cholesky update; deterministic, no RNG required.
  At each step selects the item maximising the Schur complement diagonal (= log-det
  increment). O(nk^2).
- sample (exact k-DPP): eigendecomposition of L, eigenvector selection with
  probability lambda/(1+lambda), item sampling from the projected span via
  sequential Gram-Schmidt orthogonalisation. Requires seed for reproducibility.

Conditions evaluated: dpp_map (retrieval-only + generation), dpp_seed1/2/3
(retrieval-only only). dpp_seed1 was the worst-performing condition across the
full 60-combination grid (recall delta up to -0.66 in absolute terms for one
retriever), making stochastic k-DPP the least cost-effective method tested.
Implementation: src/diversification/dpp.py.

## Dispatch layer (src/diversification/dispatch.py)
parse_condition(condition) -> (family, kwargs) maps condition strings to reranker
calls. Grammar: none | mmr_<lambda> | kmeans_k<k> | agglo_k<k> | dpp_map | dpp_seed<n>.
is_diversified(condition) returns False only for "none". rerank() dispatches to
the appropriate module. Allows eval_all_retrievers_safe.py to accept --conditions
flags for any subset of the full condition grammar without per-method if-else chains.

## k-DPP sampling: bug and fix
The initial k-DPP sample implementation attempted to pre-select k eigenvectors by
a single Bernoulli draw per eigenvector (prob = lambda/(1+lambda)), which is correct
only in expectation over many samples but does not guarantee exactly k vectors per
call. On small candidate sets (n=12 in tests, n=20 in production) this produced
k_selected != k frequently, causing index errors or wrong-length results. Fixed by
retrying the Bernoulli draw until exactly k eigenvectors are selected (expected
retries < 2 for typical lambda distributions). The test suite's 12-candidate fixture
with rigged cluster structure made this deterministic enough to catch the failure
reliably on every run.
