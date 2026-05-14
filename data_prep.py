"""
Data preparation: Download PubMedQA and prepare corpus for retrieval.
"""

import json
import os
import pickle
from typing import List, Dict, Tuple

from datasets import load_dataset
from tqdm import tqdm

from config import DATA_DIR, DATASET_NAME, DATASET_SUBSET


def download_pubmedqa() -> Dict:
    """Download PubMedQA labeled subset."""
    print("Downloading PubMedQA dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, trust_remote_code=True)
    print(f"Downloaded {len(dataset['train'])} examples")
    return dataset["train"]


def prepare_corpus(dataset) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare retrieval corpus and QA pairs from PubMedQA.

    Each PubMedQA example has:
    - pubid: PubMed ID
    - question: the research question
    - context: dict with 'contexts' (list of text sections) and 'labels' (list of section labels)
    - long_answer: detailed answer
    - final_decision: yes/no/maybe

    We create:
    - corpus: list of documents (each context section becomes a document)
    - qa_pairs: list of {question, answer, gold_doc_ids} for evaluation
    """
    corpus = []
    qa_pairs = []
    doc_id = 0

    for idx, example in enumerate(tqdm(dataset, desc="Preparing corpus")):
        pubid = example["pubid"]
        question = example["question"]
        contexts = example["context"]["contexts"]
        labels = example["context"]["labels"]
        long_answer = example["long_answer"]
        final_decision = example["final_decision"]

        gold_doc_ids = []

        for sec_idx, (text, label) in enumerate(zip(contexts, labels)):
            doc = {
                "doc_id": doc_id,
                "pubid": pubid,
                "section_label": label,
                "text": text.strip(),
                "source_question_idx": idx,
            }
            corpus.append(doc)
            gold_doc_ids.append(doc_id)
            doc_id += 1

        qa_pairs.append({
            "qa_id": idx,
            "pubid": pubid,
            "question": question,
            "long_answer": long_answer,
            "final_decision": final_decision,
            "gold_doc_ids": gold_doc_ids,
        })

    return corpus, qa_pairs


def save_data(corpus: List[Dict], qa_pairs: List[Dict]):
    """Save processed data to disk."""
    corpus_path = os.path.join(DATA_DIR, "corpus.json")
    qa_path = os.path.join(DATA_DIR, "qa_pairs.json")

    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2)
    with open(qa_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Saved {len(corpus)} documents to {corpus_path}")
    print(f"Saved {len(qa_pairs)} QA pairs to {qa_path}")


def load_data() -> Tuple[List[Dict], List[Dict]]:
    """Load processed data from disk."""
    corpus_path = os.path.join(DATA_DIR, "corpus.json")
    qa_path = os.path.join(DATA_DIR, "qa_pairs.json")

    if not os.path.exists(corpus_path) or not os.path.exists(qa_path):
        raise FileNotFoundError("Data not found. Run prepare_data() first.")

    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    with open(qa_path, "r") as f:
        qa_pairs = json.load(f)

    return corpus, qa_pairs


def prepare_data():
    """Full data preparation pipeline."""
    corpus_path = os.path.join(DATA_DIR, "corpus.json")
    if os.path.exists(corpus_path):
        print("Data already prepared. Loading from disk...")
        return load_data()

    dataset = download_pubmedqa()
    corpus, qa_pairs = prepare_corpus(dataset)
    save_data(corpus, qa_pairs)
    return corpus, qa_pairs


if __name__ == "__main__":
    corpus, qa_pairs = prepare_data()
    print(f"\nCorpus size: {len(corpus)} documents")
    print(f"QA pairs: {len(qa_pairs)}")
    print(f"\nSample document:\n{corpus[0]}")
    print(f"\nSample QA pair:\n{qa_pairs[0]}")
