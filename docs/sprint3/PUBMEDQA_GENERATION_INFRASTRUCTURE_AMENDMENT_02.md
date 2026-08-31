# PubMedQA Generation Infrastructure Amendment 02

Date: 2026-08-31

## Scope

This infrastructure defect was discovered before PubMedQA generation Blocks
13--15. The canonical ColBERTv2 PLAID index is a directory, while the existing
generation lineage helper represented only single-file artifacts. That behavior
was correct for DPR and Contriever `.faiss` indexes but could not represent the
canonical ColBERTv2 index without selecting a non-canonical file.

The 1,000 ColBERT candidate artifacts were already valid. They consistently
declare scientific index fingerprint
`a9999827368d090d305fcd9e4c7c7f4f100b290caa6c57ba2e6e0c2fcb2c3ef9` and
whole-directory physical SHA-256
`02f34ecd51a8f742442ce69642e23714d894b042cba2e54c3f7c8362701c8528`.
The candidate-declared physical SHA was independently verified against the
canonical 10-file PLAID directory using the frozen whole-directory fingerprint
procedure.

## Amendment

Generation provenance now represents DPR and Contriever indexes as before:
one repository-relative file with its file SHA-256. For ColBERTv2, it represents
the repository-relative canonical PLAID directory and records the SHA-256 from
`fingerprint_colbert_index_directory()`. In all cases, `artifact_id` remains
based on the candidate-declared scientific `index_fingerprint_sha256`; the
`sha256` field remains physical artifact integrity. Candidate-declared physical
integrity must match the supplied artifact before generation inputs are built.

This changes only generation provenance representation. Blocks 1--12 remain
governed by commit `ddca16635519470931047864d64a8e8a65d20c39` and are not
rerun. Blocks 13--15 will run prospectively from the later amendment commit.
There is no change to retrieval or generation methodology, candidates,
selected contexts, prompts, model bindings, decoding, sampling, or evaluation.
