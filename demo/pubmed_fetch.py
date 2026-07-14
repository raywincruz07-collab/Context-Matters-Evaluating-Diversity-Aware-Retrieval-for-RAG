"""
Live PubMed abstract fetcher using NCBI E-utilities API.
No API key required. Free to use.

Flow:
  query string
    -> esearch (get PMIDs)
    -> efetch  (get abstracts for those PMIDs)
    -> list of doc dicts compatible with corpus format
"""

import time
from typing import Dict, List, Optional

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _clean_query(query: str) -> str:
    """
    Convert a natural language question into a PubMed-friendly search term.
    Strips question words, punctuation, and keeps medical keywords.
    """
    import re

    # Remove question/conversational words
    stopwords = {
        "i",
        "am",
        "have",
        "having",
        "give",
        "me",
        "a",
        "an",
        "the",
        "is",
        "are",
        "does",
        "do",
        "what",
        "how",
        "why",
        "can",
        "my",
        "please",
        "prescription",
        "tell",
        "about",
        "and",
        "or",
        "for",
        "with",
        "severe",
        "some",
        "any",
        "also",
        "been",
        "been",
    }
    words = re.sub(r"[^a-zA-Z0-9 ]", " ", query.lower()).split()
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    cleaned = " ".join(keywords) if keywords else query
    return cleaned


def search_pubmed(query: str, max_results: int = 10) -> List[str]:
    """Return up to max_results PMIDs for a query."""
    cleaned = _clean_query(query)
    params = {
        "db": "pubmed",
        "term": cleaned,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        pmids = resp.json()["esearchresult"]["idlist"]
        # If cleaned query returns nothing, try original query
        if not pmids and cleaned != query:
            params["term"] = query
            resp = requests.get(ESEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            pmids = resp.json()["esearchresult"]["idlist"]
        return pmids
    except Exception:
        return []


def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """
    Fetch abstracts for a list of PMIDs.
    Returns list of doc dicts matching the corpus format.
    """
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    resp = requests.get(EFETCH_URL, params=params, timeout=15)
    resp.raise_for_status()

    return _parse_xml(resp.text, pmids)


def _parse_xml(xml_text: str, pmids: List[str]) -> List[Dict]:
    """Parse PubMed XML response into doc dicts."""
    import xml.etree.ElementTree as ET

    docs = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else "unknown"

        # Title
        title_el = article.find(".//ArticleTitle")
        title = title_el.text or "" if title_el is not None else ""

        # Abstract — may have multiple AbstractText sections (background, methods, etc.)
        abstract_sections = article.findall(".//AbstractText")
        if not abstract_sections:
            continue

        for i, section in enumerate(abstract_sections):
            label = section.get("Label", f"Section {i+1}")
            text = (section.text or "").strip()
            if not text:
                continue

            docs.append(
                {
                    "doc_id": f"pubmed_live_{pmid}_{i}",
                    "pubid": int(pmid) if pmid.isdigit() else pmid,
                    "section_label": label,
                    "text": text,
                    "source": "live_pubmed",
                    "title": title,
                }
            )

    return docs


def fetch_for_query(
    query: str,
    max_results: int = 10,
    sleep: float = 0.35,
) -> List[Dict]:
    """
    Full pipeline: query → PMIDs → abstracts → doc list.

    Args:
        query: Natural language medical question
        max_results: How many papers to fetch (each paper → multiple docs)
        sleep: Polite delay between NCBI requests (respect rate limit)

    Returns:
        List of doc dicts ready for BM25/dense retrieval
    """
    try:
        pmids = search_pubmed(query, max_results=max_results)
        if not pmids:
            return []
        time.sleep(sleep)
        docs = fetch_abstracts(pmids)
        return docs
    except requests.exceptions.Timeout:
        return []
    except requests.exceptions.RequestException:
        return []
    except Exception:
        return []
