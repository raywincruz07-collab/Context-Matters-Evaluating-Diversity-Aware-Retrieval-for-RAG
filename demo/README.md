# Sprint 1 Streamlit Demo

`app.py` is an optional interactive demo built during Sprint 1. It lets you run
single queries through the RAG pipeline via a Streamlit UI and optionally fetch
live PubMed abstracts (`pubmed_fetch.py`) for questions outside the static corpus.

**This is not part of the evaluated pipeline.** It is not included in the Colab
deployment zip (`create_colab_zip.py` walks `src/` and `tests/` only) and is not
referenced by any Sprint 1 or Sprint 2 evaluation script.

To run locally (requires `streamlit` and a valid `MAKI_API_KEY`):

```bash
cd demo
streamlit run app.py
```
