"""
Live PubMed abstract fetcher using NCBI E-utilities API.
No API key required. Free to use.
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    """Search PubMed and return list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["esearchresult"]["idlist"]
    except Exception as e:
        print(f"PubMed search error: {e}")
        return []


def fetch_abstracts(pmids: List[str]) -> List[Dict]:
    """
    Fetch abstracts for given PMIDs.
    Returns list of dicts with keys: pubid, title, abstract, sections
    """
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        return _parse_abstracts_xml(resp.text)
    except Exception as e:
        print(f"PubMed fetch error: {e}")
        return []


def _parse_abstracts_xml(xml_text: str) -> List[Dict]:
    """Parse PubMed XML response into structured dicts."""
    results = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        try:
            # PMID
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else "unknown"

            # Title
            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()) if title_el is not None else ""

            # Abstract — may have multiple labeled sections
            abstract_el = article.find(".//Abstract")
            sections = []
            if abstract_el is not None:
                for text_el in abstract_el.findall("AbstractText"):
                    label = text_el.get("Label", "ABSTRACT")
                    text = "".join(text_el.itertext()).strip()
                    if text:
                        sections.append({"label": label, "text": text})

            # If no structured sections, treat as one block
            if not sections:
                full_text = "".join(
                    "".join(t.itertext())
                    for t in (
                        abstract_el.findall("AbstractText") if abstract_el else []
                    )
                ).strip()
                if full_text:
                    sections = [{"label": "ABSTRACT", "text": full_text}]

            if sections:
                results.append(
                    {
                        "pubid": pmid,
                        "title": title,
                        "sections": sections,
                    }
                )

        except Exception:
            continue

    return results


def fetch_docs_for_query(
    query: str,
    max_results: int = 20,
    doc_id_offset: int = 0,
) -> List[Dict]:
    """
    Full pipeline: search PubMed → fetch abstracts → return corpus-compatible docs.

    Each section of each abstract becomes one document, matching the PubMedQA
    corpus format so the same retriever interface works seamlessly.

    Args:
        query: User's natural language question
        max_results: Number of PubMed papers to fetch
        doc_id_offset: Start doc_id from this value (avoids collision with static corpus)

    Returns:
        List of dicts with keys: doc_id, pubid, section_label, text, title, source
    """
    pmids = search_pubmed(query, max_results)
    if not pmids:
        return []

    # NCBI rate limit: max 3 requests/sec without API key
    time.sleep(0.34)

    articles = fetch_abstracts(pmids)

    docs = []
    doc_id = doc_id_offset
    for article in articles:
        for section in article["sections"]:
            docs.append(
                {
                    "doc_id": doc_id,
                    "pubid": article["pubid"],
                    "title": article["title"],
                    "section_label": section["label"],
                    "text": section["text"],
                    "source": "pubmed_live",
                }
            )
            doc_id += 1

    return docs
