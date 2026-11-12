"""VeriRAG Streamlit app.   Run with:  streamlit run app/main.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # make `src` and `app` importable no matter where streamlit is launched from
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.components import ui
from app.components.services import get_pipeline
from app.components.styles import inject_css
from src.evaluation.dataset import load_dataset, parse_dataset, sample_dataset_path
from src.evaluation.metrics_info import METRIC_INFO
from src.evaluation.ragas_runner import run_evaluation, save_report
from src.utils.errors import VeriRAGError

st.set_page_config(page_title="VeriRAG", page_icon="🔎", layout="wide")
inject_css()
pipeline = get_pipeline()

# ------------------------------------------------------------------ session defaults
_s = pipeline.settings
st.session_state.setdefault("messages", [])
st.session_state.setdefault("chunk_size", _s.chunk_size)
st.session_state.setdefault("chunk_overlap", _s.chunk_overlap)
st.session_state.setdefault("top_k", _s.top_k)
st.session_state.setdefault("auto_verify", False)
st.session_state.setdefault("eval_report", None)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🔎 VeriRAG")
    page = st.radio("Navigate", ["Dashboard", "Documents", "Chat", "Evaluation"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Settings**")
    st.session_state.top_k = st.slider("Passages to retrieve (top-k)", 1, 15, st.session_state.top_k)
    st.session_state.chunk_size = st.number_input("Chunk size (chars)", 200, 4000, st.session_state.chunk_size, step=100)
    st.session_state.chunk_overlap = st.number_input("Chunk overlap (chars)", 0, 1000, st.session_state.chunk_overlap, step=50)
    st.caption("Chunk settings apply to documents indexed from now on.")
    st.session_state.auto_verify = st.toggle("Auto-verify citations", st.session_state.auto_verify,
                                             help="Costs one extra Gemini call per answer.")
    if not pipeline.has_api_key:
        key = st.text_input("Gemini API key", type="password", help="Kept in memory only. Prefer using .env.")
        if key:
            pipeline.set_api_key(key)
            st.rerun()
        st.warning("No GEMINI_API_KEY found. Retrieval works, but answers, verification and evaluation need a key.")
    else:
        st.success("Gemini API key loaded", icon="🔑")


def indexed_docs() -> list[dict]:
    try:
        return pipeline.list_documents()
    except VeriRAGError as exc:
        st.error(str(exc))
        return []


# ================================================================== Dashboard
if page == "Dashboard":
    st.markdown('<div class="vr-hero"><h1>VeriRAG</h1><p>Citation-aware document Q&A with claim-level verification '
                'and RAG evaluation.</p></div>', unsafe_allow_html=True)
    docs = indexed_docs()
    c1, c2, c3, c4 = st.columns(4)
    with c1: ui.card("Documents", str(len(docs)), "indexed PDFs")
    with c2: ui.card("Chunks", str(sum(d["chunks"] for d in docs)), "in ChromaDB")
    with c3: ui.card("Questions asked", str(sum(1 for m in st.session_state.messages if m["role"] == "user")), "this session")
    with c4: ui.card("LLM", pipeline.settings.gemini_model, "API key ✓" if pipeline.has_api_key else "API key missing")
    st.markdown("### How it works")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**1 · Upload**\n\nPDFs are read page by page and cut into overlapping chunks that remember their document and page.")
    with b:
        st.markdown("**2 · Ask**\n\nYour question is embedded, the closest chunks are retrieved, and Gemini answers using only those chunks with `[n]` citations.")
    with c:
        st.markdown("**3 · Verify & Evaluate**\n\nEach claim is checked against its cited passage, and RAGAS scores the whole pipeline.")
    if not docs:
        st.info("Start on the **Documents** page and upload a PDF.", icon="👉")

# ================================================================== Documents
elif page == "Documents":
    st.markdown("## 📄 Documents")
    files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Index uploaded PDFs", type="primary"):
        bar = st.progress(0.0)
        for i, f in enumerate(files):
            with st.spinner(f"Processing {f.name} (first run downloads the embedding model)..."):
                try:
                    rep = pipeline.ingest(f.name, f.getvalue(), st.session_state.chunk_size, st.session_state.chunk_overlap)
                    if rep.already_indexed:
                        st.info(f"**{f.name}** was already indexed — nothing new embedded.")
                    else:
                        msg = f"**{f.name}**: {rep.total_pages} pages → {rep.chunks_total} chunks ({rep.chunks_added} newly embedded)."
                        if rep.empty_pages:
                            msg += f" Skipped {len(rep.empty_pages)} page(s) without text: {rep.empty_pages[:10]}."
                        st.success(msg)
                except VeriRAGError as exc:
                    st.error(f"{f.name}: {exc}")
                except Exception as exc:  # last-resort guard so one bad file never crashes the app
                    st.error(f"{f.name}: unexpected error — {exc}")
            bar.progress((i + 1) / len(files))
    st.markdown("### Indexed documents")
    docs = indexed_docs()
    if not docs:
        st.caption("Nothing indexed yet.")
    for d in docs:
        col1, col2, col3 = st.columns([5, 3, 1])
        col1.markdown(f"**{d['document']}**  \n<span class='vr-note'>id {d['doc_id']}</span>", unsafe_allow_html=True)
        col2.write(f"{d['pages']} pages with text · {d['chunks']} chunks")
        if col3.button("🗑️", key=f"del_{d['doc_id']}", help="Remove from index"):
            try:
                pipeline.delete_document(d["doc_id"])
                st.rerun()
            except VeriRAGError as exc:
                st.error(str(exc))

# ================================================================== Chat
elif page == "Chat":
    st.markdown("## 💬 Ask your documents")
    docs = indexed_docs()
    if not docs:
        st.info("Upload and index a PDF first (Documents page).", icon="👉")
    names = {d["doc_id"]: d["document"] for d in docs}
    selected = st.multiselect("Restrict to documents (empty = all)", list(names), format_func=names.get)

    def run_verify(idx: int) -> None:
        msg = st.session_state.messages[idx]
        try:
            with st.spinner("Verifying claims against cited passages..."):
                msg["verdicts"] = pipeline.verify(msg["result"])
                msg["verify_error"] = None
        except VeriRAGError as exc:
            msg["verify_error"] = str(exc)

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
                continue
            res = msg["result"]
            ui.render_answer_text(res)
            for w in res.warnings:
                st.warning(w)
            tabs = st.tabs(["Sources", "Retrieved context", "Citation verification"])
            with tabs[0]:
                ui.render_citations(res)
            with tabs[1]:
                ui.render_retrieved(res)
            with tabs[2]:
                if not res.found:
                    st.caption("Nothing to verify: the answer was not found in the documents.")
                elif msg.get("verdicts") is not None:
                    ui.render_verification(msg["verdicts"])
                else:
                    if msg.get("verify_error"):
                        st.error(msg["verify_error"])
                    if st.button("Verify citations", key=f"verify_{i}"):
                        run_verify(i)
                        st.rerun()

    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Retrieving passages and generating a grounded answer..."):
                result = pipeline.ask(question, doc_ids=selected or None, top_k=st.session_state.top_k)
            st.session_state.messages.append({"role": "assistant", "result": result, "verdicts": None, "verify_error": None})
            if st.session_state.auto_verify and result.found:
                run_verify(len(st.session_state.messages) - 1)
            st.rerun()
        except VeriRAGError as exc:
            st.session_state.messages.pop()  # drop the unanswered question so history stays consistent
            st.error(str(exc))

    if st.session_state.messages and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ================================================================== Evaluation
else:
    st.markdown("## 📊 RAG Evaluation (RAGAS)")
    st.caption("Scores are computed live by RAGAS using your real indexed documents — nothing is pre-filled.")
    with st.expander("What do these metrics mean?", expanded=False):
        for info in METRIC_INFO.values():
            st.markdown(f"**{info['title']}** — {info['what']}  \n{info['how']}  \n*{info['low']}*")

    source = st.radio("Dataset", ["Sample dataset", "Upload JSON"], horizontal=True)
    items = None
    try:
        if source == "Sample dataset":
            items = load_dataset(sample_dataset_path())
            st.info("The sample questions are about **“Attention Is All You Need”** (Vaswani et al., 2017). "
                    "Upload that PDF on the Documents page first, or use your own dataset.", icon="📘")
        else:
            up = st.file_uploader("JSON list of {question, reference}", type=["json"])
            if up:
                items = parse_dataset(up.getvalue())
    except VeriRAGError as exc:
        st.error(str(exc))
    if items:
        st.dataframe(pd.DataFrame([{"question": i.question, "reference": i.reference} for i in items]), width="stretch")

    docs = indexed_docs()
    names = {d["doc_id"]: d["document"] for d in docs}
    selected = st.multiselect("Restrict to documents (empty = all)", list(names), format_func=names.get, key="eval_docs")

    if st.button("Run evaluation", type="primary", disabled=not items):
        bar = st.progress(0.0, "Starting...")
        try:
            report = run_evaluation(pipeline, items, selected or None, st.session_state.top_k,
                                    progress=lambda f, t: bar.progress(min(f, 1.0), t))
            st.session_state.eval_report = report
            save_report(report, pipeline.settings.eval_dir)
            bar.progress(1.0, "Done")
        except VeriRAGError as exc:
            bar.empty()
            st.session_state.eval_report = None
            st.error(f"Evaluation could not be completed: {exc}")
        except Exception as exc:
            bar.empty()
            st.session_state.eval_report = None
            st.error(f"Unexpected evaluation error: {exc}")

    report = st.session_state.eval_report
    if report:
        st.markdown("### Results")
        cols = st.columns(4)
        for col, (key, info) in zip(cols, METRIC_INFO.items()):
            val = report.scores.get(key)
            with col:
                ui.card(info["title"], "n/a" if val is None else f"{val:.2f}", "mean over questions")
        for e in report.errors:
            st.warning(e)
        st.markdown("#### Per-question scores")
        st.dataframe(report.per_sample, width="stretch")
        st.caption(report.meta.get("judge_note", ""))
        d1, d2 = st.columns(2)
        d1.download_button("Download CSV", report.to_csv(), "verirag_eval.csv", "text/csv")
        d2.download_button("Download JSON", report.to_json(), "verirag_eval.json", "application/json")
