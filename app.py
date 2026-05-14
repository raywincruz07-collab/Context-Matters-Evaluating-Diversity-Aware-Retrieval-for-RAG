"""
Streamlit UI for Medical RAG Pipeline.
Run with: streamlit run app.py
"""

import json
import os
import sys
import time

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (MAKI_API_KEY, MAKI_HOST, MAKI_MODEL, RETRIEVER_CHOICES,
                    TOP_K)
from data_prep import load_data, prepare_data
from evaluation import evaluate_single
from generator import MakiGenerator
from pubmed_fetch import fetch_for_query
from retrievers.factory import get_retriever

# ─── Page Config ───
st.set_page_config(
    page_title="MedRAG — Medical RAG Pipeline",
    page_icon="🏥",
    layout="wide",
)

# ─── Custom CSS ───
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5d6d7e;
        margin-bottom: 1.5rem;
    }
    .doc-card {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    .doc-score {
        color: #27ae60;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .doc-meta {
        color: #7f8c8d;
        font-size: 0.8rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .status-ok { color: #27ae60; }
    .status-err { color: #e74c3c; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State Init ───
def init_state():
    defaults = {
        "corpus": None,
        "qa_pairs": None,
        "retriever": None,
        "retriever_name": None,
        "generator": None,
        "data_loaded": False,
        "retriever_ready": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


# ─── Data Loading ───
@st.cache_data(show_spinner=False)
def cached_load_data():
    return prepare_data()


# ─── Sidebar ───
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # Data status
    st.markdown("### 📊 Dataset")
    if st.button("Load PubMedQA", use_container_width=True):
        with st.spinner("Downloading & preparing PubMedQA..."):
            corpus, qa_pairs = cached_load_data()
            st.session_state.corpus = corpus
            st.session_state.qa_pairs = qa_pairs
            st.session_state.data_loaded = True

    if st.session_state.data_loaded:
        st.success(
            f"✓ {len(st.session_state.corpus)} docs | {len(st.session_state.qa_pairs)} QA pairs"
        )
    else:
        st.info("Click above to load dataset")

    st.markdown("---")

    # Retriever selection
    st.markdown("### 🔍 Retriever")
    selected_retriever = st.selectbox(
        "Choose retriever",
        RETRIEVER_CHOICES,
        help="BM25 is fastest. Dense models need embedding computation on first run.",
    )

    if st.session_state.data_loaded:
        if st.button("Build Index", use_container_width=True):
            with st.spinner(f"Building {selected_retriever} index..."):
                retriever = get_retriever(selected_retriever)
                retriever.index(st.session_state.corpus)
                st.session_state.retriever = retriever
                st.session_state.retriever_name = selected_retriever
                st.session_state.retriever_ready = True

    if st.session_state.retriever_ready:
        st.success(f"✓ {st.session_state.retriever_name} ready")

    st.markdown("---")

    # Generator status
    st.markdown("### 🤖 Generator")
    maki_key = st.text_input(
        "University GPU API Key",
        type="password",
        placeholder="sk_...",
        help="Use the API key for https://maki.uni-mannheim.de/v1",
    )
    if maki_key:
        import os

        os.environ["MAKI_API_KEY"] = maki_key
    api_key = maki_key or MAKI_API_KEY

    maki_host = st.text_input(
        "API Host", value=MAKI_HOST, help="OpenAI-compatible /v1 endpoint"
    )
    generator = MakiGenerator(api_key=api_key, host=maki_host)
    maki_model = st.selectbox(
        "Model",
        [
            "ministral-3-14b",
            "magistral:24b_6k",
        ],
        index=0,
        help="Use the model name provided by the university endpoint",
    )
    generator.model = maki_model
    if api_key:
        st.markdown(
            f'<span class="status-ok">✓ University GPU API ready ({maki_model})</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-err">✗ No API key set</span>', unsafe_allow_html=True
        )
        st.caption('Enter key above or set: $env:MAKI_API_KEY = "your_key"')
    st.session_state.generator = generator

    st.markdown("---")

    # Parameters
    st.markdown("### 🎛️ Parameters")
    top_k = st.slider("Top-K documents", 1, 20, TOP_K)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)


# ─── Main Content ───
st.markdown('<div class="main-header">🏥 MedRAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Medical Retrieval-Augmented Generation — Diversity-Aware Retrieval Evaluation</div>',
    unsafe_allow_html=True,
)

# ─── Tabs ───
tab_query, tab_eval, tab_explore = st.tabs(
    ["💬 Query", "📈 Evaluate", "📚 Explore Data"]
)

# ─── Query Tab ───
with tab_query:
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.markdown("### Ask a Medical Question")

        # Sample questions from PubMedQA
        sample_questions = ["Type your own question..."]
        if st.session_state.qa_pairs:
            sample_questions += [
                qa["question"] for qa in st.session_state.qa_pairs[:20]
            ]

        selected_q = st.selectbox("Sample questions from PubMedQA", sample_questions)

        if selected_q == "Type your own question...":
            query = st.text_area(
                "Your question:",
                height=100,
                placeholder="e.g., Does aspirin reduce heart attack risk?",
            )
            matching_qa = None
            is_custom = True
        else:
            query = st.text_area("Your question:", value=selected_q, height=100)
            matching_qa = next(
                (
                    qa
                    for qa in (st.session_state.qa_pairs or [])
                    if qa["question"] == selected_q
                ),
                None,
            )
            is_custom = False

        # Mode selector
        if is_custom:
            st.info(
                "💡 Custom question detected — will fetch live PubMed abstracts as context."
            )
            use_gold_docs = False
        else:
            use_gold_docs = st.checkbox(
                "Use gold documents (debug mode)",
                value=False,
                help="Bypasses the retriever and feeds the ground-truth documents directly to the LLM.",
            )

        can_run = bool(query.strip()) and st.session_state.generator is not None

        run_clicked = st.button(
            "🔍 Run RAG Pipeline", disabled=not can_run, use_container_width=True
        )

        if not query.strip():
            st.warning("Enter a question to continue")

    with col_output:
        if run_clicked and can_run:
            retriever = st.session_state.retriever
            generator = st.session_state.generator

            # Retrieve — three modes:
            # 1. Gold docs (debug): use ground-truth docs directly
            # 2. Custom question: live PubMed fetch + BM25
            # 3. PubMedQA sample: static corpus retrieval
            if use_gold_docs and matching_qa:
                gold_ids = matching_qa["gold_doc_ids"]
                context_docs = [st.session_state.corpus[i] for i in gold_ids]
                retrieved = [(doc, 1.0) for doc in context_docs]
                ret_time = 0.0
                all_fetched = []
                st.info(
                    f"Debug mode: using {len(context_docs)} gold documents directly"
                )

            elif is_custom:
                with st.spinner(
                    "Searching PubMed for relevant abstracts... (this takes ~5s)"
                ):
                    t0 = time.time()
                    live_docs = fetch_for_query(query, max_results=15)
                    ret_time = time.time() - t0

                if not live_docs:
                    st.error(
                        "No PubMed abstracts found. "
                        "Try using medical keywords instead of full sentences — "
                        "e.g. 'fever cough treatment' instead of 'I have fever and cough'."
                    )
                    st.stop()

                # BM25 over the live docs to rank them
                from rank_bm25 import BM25Okapi

                texts = [d["text"] for d in live_docs]
                tokenized = [t.lower().split() for t in texts]
                bm25_live = BM25Okapi(tokenized)
                scores = bm25_live.get_scores(query.lower().split())
                top_indices = scores.argsort()[-top_k:][::-1]
                retrieved = [
                    (live_docs[i], float(scores[i]))
                    for i in top_indices
                    if scores[i] > 0
                ]
                if not retrieved:
                    retrieved = [
                        (live_docs[i], 1.0) for i in range(min(top_k, len(live_docs)))
                    ]

                st.success(
                    f"✓ Fetched {len(live_docs)} sections from PubMed, showing top {len(retrieved)}"
                )
                all_fetched = live_docs

            else:
                with st.spinner("Retrieving documents from static corpus..."):
                    t0 = time.time()
                    retrieved = (
                        st.session_state.retriever.retrieve(query, top_k=top_k)
                        if st.session_state.retriever_ready
                        else []
                    )
                    ret_time = time.time() - t0
                all_fetched = []

                # Warn if retrieved docs look off-topic
                if matching_qa and retrieved:
                    gold_ids = set(matching_qa["gold_doc_ids"])
                    retrieved_ids = set(doc["doc_id"] for doc, _ in retrieved)
                    overlap = len(gold_ids & retrieved_ids)
                    if overlap == 0:
                        st.warning(
                            f"⚠️ None of the gold documents retrieved. Try increasing Top-K or enable debug mode."
                        )
                    else:
                        st.success(
                            f"✓ {overlap}/{len(gold_ids)} gold documents retrieved"
                        )

            src_label = "live PubMed" if is_custom else "static corpus"
            st.markdown(
                f"### Retrieved Documents ({len(retrieved)} docs from {src_label}, {ret_time:.2f}s)"
            )
            for i, (doc, score) in enumerate(retrieved, 1):
                is_gold = matching_qa and doc["doc_id"] in matching_qa["gold_doc_ids"]
                label = (
                    f"Doc {i} — Score: {score:.4f} | PubMed: {doc.get('pubid', 'N/A')}"
                )
                if is_gold:
                    label += " ✓ gold"
                with st.expander(label, expanded=(i <= 2)):
                    if doc.get("title"):
                        st.markdown(f"**{doc['title']}**")
                    st.markdown(
                        f'<div class="doc-meta">Section: {doc.get("section_label", "N/A")} | Source: {doc.get("source", "static")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.write(doc["text"])

            st.markdown("---")

            # Generate
            st.markdown("### Generated Answer")
            # context_docs may already be set (gold mode); set from retrieved if not
            if not use_gold_docs or not matching_qa:
                context_docs = [doc for doc, _ in retrieved]

            if generator.is_available():
                t0 = time.time()
                answer_placeholder = st.empty()
                full_answer = ""
                for token in generator.generate_streaming(
                    query, context_docs, temperature=temperature
                ):
                    full_answer += token
                    answer_placeholder.markdown(full_answer)
                gen_time = time.time() - t0

                st.caption(f"Generation time: {gen_time:.2f}s")

                # Show ground truth if from PubMedQA
                if matching_qa:
                    st.markdown("---")
                    st.markdown("### Ground Truth")
                    st.markdown(f"**Decision:** {matching_qa['final_decision']}")
                    st.write(matching_qa["long_answer"])

                    # Quick metrics
                    metrics = evaluate_single(
                        prediction=full_answer,
                        ground_truth=matching_qa["long_answer"],
                        retrieved_doc_ids=[d["doc_id"] for d, _ in retrieved],
                        gold_doc_ids=matching_qa["gold_doc_ids"],
                    )
                    st.markdown("### Metrics")
                    mcols = st.columns(len(metrics))
                    for col, (name, val) in zip(mcols, metrics.items()):
                        col.metric(name.replace("_", " ").title(), f"{val:.3f}")
            else:
                st.error("Groq API key not set. Enter it in the sidebar.")
                st.code(
                    "export GROQ_API_KEY=your_key_here\n# Get free key at https://console.groq.com",
                    language="bash",
                )


# ─── Evaluate Tab ───
with tab_eval:
    st.markdown("### Batch Evaluation")
    st.markdown("Run the full pipeline on multiple QA pairs and compare retrievers.")

    if not st.session_state.data_loaded:
        st.info("Load dataset first to run evaluation.")
    elif not st.session_state.retriever_ready:
        st.info("Build a retriever index first.")
    else:
        eval_col1, eval_col2 = st.columns([1, 2])

        with eval_col1:
            num_samples = st.number_input(
                "Number of samples", 1, len(st.session_state.qa_pairs), 10
            )
            eval_top_k = st.number_input("Top-K for eval", 1, 20, top_k)

            if st.button("▶️ Run Evaluation", use_container_width=True):
                retriever = st.session_state.retriever
                generator = st.session_state.generator

                if not generator.is_available():
                    st.error("University GPU API key not set. Enter it in the sidebar.")
                else:
                    progress = st.progress(0)
                    results = []

                    for i, qa in enumerate(st.session_state.qa_pairs[:num_samples]):
                        progress.progress(
                            (i + 1) / num_samples, f"Evaluating {i+1}/{num_samples}..."
                        )

                        retrieved = retriever.retrieve(qa["question"], top_k=eval_top_k)
                        context_docs = [doc for doc, _ in retrieved]
                        answer = generator.generate(
                            qa["question"], context_docs, temperature=temperature
                        )

                        metrics = evaluate_single(
                            prediction=answer,
                            ground_truth=qa["long_answer"],
                            retrieved_doc_ids=[d["doc_id"] for d, _ in retrieved],
                            gold_doc_ids=qa["gold_doc_ids"],
                        )
                        results.append(
                            {
                                "question": qa["question"][:80] + "...",
                                **metrics,
                            }
                        )

                    progress.empty()
                    st.session_state["eval_results"] = results

        with eval_col2:
            if "eval_results" in st.session_state and st.session_state["eval_results"]:
                results = st.session_state["eval_results"]
                import pandas as pd

                df = pd.DataFrame(results)

                # Averages
                st.markdown("#### Aggregated Metrics")
                avg_cols = st.columns(5)
                for col, metric in zip(
                    avg_cols, ["exact_match", "f1", "rouge_l", "recall_at_k", "mrr"]
                ):
                    if metric in df.columns:
                        col.metric(
                            metric.replace("_", " ").title(), f"{df[metric].mean():.3f}"
                        )

                # Per-example table
                st.markdown("#### Per-Example Results")
                st.dataframe(
                    df.style.format(
                        {c: "{:.3f}" for c in df.columns if c != "question"}
                    ),
                    use_container_width=True,
                )

                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    f"eval_{st.session_state.retriever_name}_results.csv",
                    "text/csv",
                )


# ─── Explore Tab ───
with tab_explore:
    st.markdown("### Explore PubMedQA Dataset")

    if not st.session_state.data_loaded:
        st.info("Load dataset first.")
    else:
        import pandas as pd

        qa_df = pd.DataFrame(st.session_state.qa_pairs)
        st.markdown(
            f"**Total QA pairs:** {len(qa_df)} | **Total documents:** {len(st.session_state.corpus)}"
        )

        # Decision distribution
        if "final_decision" in qa_df.columns:
            st.markdown("#### Answer Distribution")
            dist = qa_df["final_decision"].value_counts()
            st.bar_chart(dist)

        # Browse QA pairs
        st.markdown("#### Browse QA Pairs")
        idx = st.number_input("QA Index", 0, len(qa_df) - 1, 0)
        qa = st.session_state.qa_pairs[idx]

        st.markdown(f"**Question:** {qa['question']}")
        st.markdown(f"**Decision:** {qa['final_decision']}")
        st.markdown(f"**Answer:** {qa['long_answer']}")

        st.markdown("**Gold Documents:**")
        for doc_id in qa["gold_doc_ids"]:
            doc = st.session_state.corpus[doc_id]
            with st.expander(f"Doc {doc_id} — {doc['section_label']}"):
                st.write(doc["text"])
