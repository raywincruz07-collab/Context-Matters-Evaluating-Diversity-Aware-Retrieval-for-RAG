# Sprint 3 Research Questions

## Project

**Context Matters: Evaluating Diversity-Aware Retrieval for RAG**

Sprint 3 is the final experimental and analytical phase of the project.

The central objective is not to prove that diversification is beneficial.
The objective is to determine **when, where, and under what information-need
conditions diversity-aware retrieval helps or harms downstream RAG performance.**

---

## Primary Research Question

### RQ1

**Does diversity-aware retrieval improve downstream RAG performance compared
with standard relevance-only retrieval, and how does this depend on the type
of information need?**

The comparison must consider both:

- retrieval quality, and
- generated-answer quality.

A diversification method is not considered successful merely because retrieved
documents are more diverse.

---

## Dataset Roles

The datasets represent different evidence structures.

### PubMedQA — single-source control

Role:

- control condition from Sprint 1,
- limited opportunity for useful retrieval diversification,
- helps test whether diversification adds unnecessary noise when a question
  primarily depends on focused evidence.

Sprint 3 should use existing validated Sprint 1 results where appropriate
rather than rerunning experiments without a methodological reason.

### BEIR HotpotQA — multi-hop / complementary evidence

Role:

- Sprint 2 corpus-level evaluation,
- questions often require combining evidence from multiple documents,
- useful for studying complementarity between retrieved documents.

Important interpretation boundary:

**Multi-hop complementarity is not automatically the same thing as aspect
diversity.**

Therefore, HotpotQA results must not be generalized directly to all
diversity-oriented RAG settings.

### ASQA — aspect-diversity evaluation

Planned Sprint 3 role:

- evaluate ambiguous or multi-aspect information needs,
- provide a setting where retrieving multiple relevant aspects may be directly
  beneficial,
- support aspect-aware metrics such as coverage and alpha-nDCG where the
  dataset annotations permit them.

The exact ASQA split and sampling protocol must be fixed before final
experiments.

---

## Secondary Research Questions

### RQ2 — Diversity–relevance trade-off

**How much retrieval relevance is lost or preserved as retrieval diversity
increases?**

Evaluate the relationship between diversity and metrics such as:

- Recall@k,
- MRR@k,
- NDCG@k,
- retrieval diversity.

The analysis should examine the trade-off rather than optimize diversity alone.

---

### RQ3 — Diversity–faithfulness relationship

**Does increasing retrieval diversity improve or reduce answer faithfulness?**

Possible outcomes include:

- useful complementary evidence improves support,
- irrelevant diversity introduces noise,
- excessive diversification increases unsupported generation.

Faithfulness conclusions must be based on measured evaluation rather than
retrieval diversity alone.

---

### RQ4 — Aspect coverage

**For information needs containing multiple valid aspects, does
diversification improve coverage of those aspects?**

This question is especially important for ASQA.

Candidate metrics include, where supported by the dataset:

- alpha-nDCG,
- aspect/subtopic coverage,
- answer coverage.

Metric definitions must be fixed in `METRICS_PROTOCOL.md` before final runs.

---

### RQ5 — Answer quality

**Does improved retrieval diversity or aspect coverage translate into better
answers?**

Candidate answer-level measures may include:

- reference-answer correctness,
- lexical or semantic answer metrics,
- faithfulness,
- coverage.

The final metric set must be defined before final evaluation.

---

### RQ6 — Method comparison

**Which diversification approaches provide the best trade-off between
retrieval diversity, relevance, coverage, and downstream answer quality?**

Candidate method families currently include:

- MMR,
- clustering-based diversification,
- DPP-based diversification,
- relevance-only retrieval baseline.

Method selection must use a pre-defined evidence-based rule.

A method must not be selected simply because it has the highest score on a
single observed metric.

---

### RQ7 — Context-size sensitivity

**Do the effects of diversification change when retrieval depth or candidate
pool size changes?**

Planned factors may include:

- top-k,
- candidate-pool size.

Only configurations approved in the experiment protocol should be run.

---

### RQ8 — Output diversity

**Does diversity in retrieved context produce meaningful diversity in generated
answers?**

Possible measures may include:

- Self-BLEU,
- lexical diversity,
- answer novelty.

Output diversity is not automatically beneficial and must be interpreted
together with correctness and faithfulness.

---

## Claim Boundaries

Sprint 3 conclusions must distinguish between:

1. **Single-source evidence**
   - PubMedQA

2. **Multi-hop complementary evidence**
   - BEIR HotpotQA

3. **Multi-aspect / ambiguous information needs**
   - ASQA

A result observed on one regime should not automatically be claimed to apply
to the others.

---

## Statistical Principle

Final comparisons should report uncertainty where appropriate.

Preferred analyses include:

- paired comparisons at the question level,
- confidence intervals,
- paired bootstrap confidence intervals where suitable,
- Pareto/trade-off analysis for competing objectives.

Statistical procedures must be defined before final result interpretation.

---

## Method-Selection Principle

Sprint 3 must separate:

- method development,
- method selection,
- final evaluation.

Selection criteria should be fixed before examining final evaluation results.

The final method-selection procedure will be documented separately in:

`docs/sprint3/METHOD_SELECTION_PROTOCOL.md`

---

## Current Status

This document defines the research-question framework.

It does **not** yet finalize:

- ASQA split selection,
- ASQA sampling procedure,
- final metric definitions,
- method-selection thresholds,
- final experiment matrix,
- statistical testing procedure.

Those decisions must be documented separately before large Sprint 3 runs.
