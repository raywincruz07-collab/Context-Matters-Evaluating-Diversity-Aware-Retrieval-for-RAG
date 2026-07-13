"""
Data preparation for HotpotQA (distractor config) — Sprint 2.

Loads from disk (dataset must be pre-downloaded via download_hotpotqa.py).
Samples exactly HOTPOT_SAMPLE_SIZE questions from validation split with HOTPOT_SEED.
Builds corpus: each (title, sentences) pair in context becomes one document.
No deduplication across questions even when titles repeat.
Gold doc ids = docs whose title appears in supporting_facts["title"].

Output schema is intentionally parallel to data_prep.py so existing code paths
that call load_data() / prepare_data() are unchanged.

BEFORE RUNNING ON COLAB: the dataset must be inside medical-rag-maki-colab/data/.
If you downloaded it to the parent RAG-Project/ directory, move it first:
    mv data/hotpotqa_distractor medical-rag-maki-colab/data/
or, if running download_hotpotqa.py from the project root:
    python download_hotpotqa.py   # saves to ./data/hotpotqa_distractor — correct path
"""

import json
import os
import random
from typing import Dict, List, Tuple

from config import DATA_DIR, HOTPOT_SAMPLE_SIZE, HOTPOT_SEED

HOTPOT_DATASET_PATH = os.path.join(DATA_DIR, "hotpotqa_distractor")
HOTPOT_CORPUS_PATH = os.path.join(DATA_DIR, "hotpot_corpus.json")
HOTPOT_QA_PATH = os.path.join(DATA_DIR, "hotpot_qa_pairs.json")


def build_hotpot_data(dataset_path: str = HOTPOT_DATASET_PATH) -> Tuple[List[Dict], List[Dict]]:
    """
    Build corpus and qa_pairs from HotpotQA distractor validation split.

    Uses HOTPOT_SEED for reproducible sampling.
    Raises FileNotFoundError (with an actionable message) if the dataset is missing.
    Raises AssertionError if any gold_doc_id is missing from the corpus.
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"HotpotQA dataset not found at: {dataset_path}\n"
            "Fix: make sure download_hotpotqa.py has been run from the project root "
            "(medical-rag-maki-colab/) so the dataset lands in "
            "medical-rag-maki-colab/data/hotpotqa_distractor.\n"
            "If you downloaded it to the parent directory, move it with:\n"
            "    mv data/hotpotqa_distractor medical-rag-maki-colab/data/"
        )

    from datasets import load_from_disk

    print(f"[seed={HOTPOT_SEED}] Loading HotpotQA dataset from {dataset_path} ...")
    dataset = load_from_disk(dataset_path)

    val_split = dataset["validation"]
    print(f"Validation split size: {len(val_split)} examples")

    if HOTPOT_SAMPLE_SIZE > len(val_split):
        raise ValueError(
            f"HOTPOT_SAMPLE_SIZE={HOTPOT_SAMPLE_SIZE} exceeds validation split size {len(val_split)}"
        )

    # Fixed-seed sampling — reproducible across runs
    rng = random.Random(HOTPOT_SEED)
    all_indices = list(range(len(val_split)))
    sampled_indices = sorted(rng.sample(all_indices, HOTPOT_SAMPLE_SIZE))
    print(f"Sampled {len(sampled_indices)} questions (seed={HOTPOT_SEED})")

    corpus: List[Dict] = []
    qa_pairs: List[Dict] = []
    doc_id_counter = 0

    for qa_idx, orig_idx in enumerate(sampled_indices):
        example = val_split[orig_idx]

        question = example["question"]
        answer = example["answer"]
        context = example["context"]          # {"title": [...], "sentences": [[...]...]}
        supporting_facts = example["supporting_facts"]  # {"title": [...], "sent_id": [...]}

        gold_titles = set(supporting_facts["title"])

        titles = context["title"]
        sentences_list = context["sentences"]

        local_gold_doc_ids: List[int] = []

        for title, sentences in zip(titles, sentences_list):
            doc = {
                "doc_id": doc_id_counter,
                "title": title,
                "text": " ".join(sentences),
                "source_question_idx": qa_idx,
            }
            corpus.append(doc)

            if title in gold_titles:
                local_gold_doc_ids.append(doc_id_counter)

            doc_id_counter += 1

        qa_pairs.append(
            {
                "qa_id": qa_idx,
                "question": question,
                "long_answer": answer,
                "gold_doc_ids": local_gold_doc_ids,
            }
        )

    # Integrity check — fail loudly, never silently skip
    corpus_id_set = {doc["doc_id"] for doc in corpus}
    for qa in qa_pairs:
        for gid in qa["gold_doc_ids"]:
            if gid not in corpus_id_set:
                raise AssertionError(
                    f"Integrity violation: gold_doc_id={gid} in qa_id={qa['qa_id']} "
                    f"not found in corpus (corpus size={len(corpus)})"
                )

    print(f"Corpus size : {len(corpus)} docs")
    print(f"QA pairs   : {len(qa_pairs)}")

    return corpus, qa_pairs


def save_hotpot_data(corpus: List[Dict], qa_pairs: List[Dict]) -> None:
    with open(HOTPOT_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)
    print(f"Saved corpus to {HOTPOT_CORPUS_PATH}")

    with open(HOTPOT_QA_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"Saved qa_pairs to {HOTPOT_QA_PATH}")


def load_hotpot_data() -> Tuple[List[Dict], List[Dict]]:
    for path in (HOTPOT_CORPUS_PATH, HOTPOT_QA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"HotpotQA data not found at {path}. "
                "Run prepare_hotpot_data() or build_hotpot_data() first."
            )

    with open(HOTPOT_CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(HOTPOT_QA_PATH, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    return corpus, qa_pairs


def prepare_hotpot_data(dataset_path: str = HOTPOT_DATASET_PATH) -> Tuple[List[Dict], List[Dict]]:
    """Load cached data if it exists, otherwise build and cache it."""
    if os.path.exists(HOTPOT_CORPUS_PATH) and os.path.exists(HOTPOT_QA_PATH):
        print("Loading cached HotpotQA data from disk ...")
        return load_hotpot_data()

    corpus, qa_pairs = build_hotpot_data(dataset_path)
    save_hotpot_data(corpus, qa_pairs)
    return corpus, qa_pairs


if __name__ == "__main__":
    corpus, qa_pairs = prepare_hotpot_data()
    print(f"\nDone. Corpus: {len(corpus)} docs | QA pairs: {len(qa_pairs)}")
