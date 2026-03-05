"""
app.py — HandNotes RAG System — Complete Streamlit UI
Run: streamlit run app.py
"""

import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="HandNotes RAG",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #080c14; color: #dde3ee; }
[data-testid="stSidebar"] { background: #0d1220 !important; border-right: 1px solid #1c2540; }
.hero {
    background: linear-gradient(130deg, #091428 0%, #0c1e3a 45%, #0a1228 100%);
    border: 1px solid #1e3358; border-radius: 18px;
    padding: 30px 40px 26px; margin-bottom: 26px;
    position: relative; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(56,139,255,0.12) 0%, transparent 65%);
}
.hero h1 { font-size: 2.1rem; font-weight: 800; color: #f0f6ff; margin: 0; letter-spacing: -1px; }
.hero p  { color: #7a90b4; margin: 7px 0 0; font-size: 0.93rem; }
.bubble-user {
    background: #0f2044; border: 1px solid #1a3462;
    border-radius: 14px 14px 4px 14px; padding: 13px 17px;
    margin: 10px 0; margin-left: 12%; color: #d9e8ff; font-size: 0.97rem;
}
.bubble-bot {
    background: #0d1526; border: 1px solid #1c2a46;
    border-radius: 14px 14px 14px 4px; padding: 13px 17px;
    margin: 10px 0; margin-right: 8%; color: #dde3ee;
    font-size: 0.97rem; line-height: 1.75;
}
.src-card {
    background: #080e1c; border: 1px solid #1c2a46;
    border-left: 3px solid #2d6af5; border-radius: 8px;
    padding: 8px 13px; margin: 4px 0;
    font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; color: #7a90b4;
}
.src-card b { color: #a8c0e8; }
.badge { display: inline-block; padding: 3px 11px; border-radius: 20px;
         font-size: 0.77rem; font-weight: 700; margin: 7px 0 2px; letter-spacing: 0.3px; }
.badge-high { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.28); }
.badge-med  { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.28); }
.badge-low  { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.28); }
.lbl { font-size: 0.72rem; font-weight: 700; letter-spacing: 1.8px;
       text-transform: uppercase; color: #3d5070; margin: 18px 0 8px; }
.up-hint { text-align: center; color: #3d5070; padding: 28px;
           border: 2px dashed #1c2a46; border-radius: 14px; font-size: 0.9rem; }
.mpill { background: #0d1526; border: 1px solid #1c2a46; border-radius: 10px;
         padding: 13px 16px; text-align: center; }
.mpill .num  { font-size: 1.7rem; font-weight: 800; color: #2d6af5; }
.mpill .name { font-size: 0.78rem; color: #3d5070; margin-top: 3px; }
.stTextInput > div > div > input {
    background: #0d1526 !important; border: 1px solid #1c2a46 !important;
    color: #dde3ee !important; border-radius: 9px !important;
}
.stSelectbox > div > div { background: #0d1526 !important; border-color: #1c2a46 !important; color: #dde3ee !important; }
.stButton > button {
    background: #0d1526 !important; color: #6d9df0 !important;
    border: 1px solid #1c2a46 !important; border-radius: 9px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { background: #162240 !important; border-color: #2d6af5 !important; }
.stAlert { border-radius: 9px !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in {"pipeline": None, "qa_history": [], "sys_status": None,
              "ocr_provider": os.getenv("OCR_PROVIDER", "mistral"),
              "ocr_api_key": os.getenv("MISTRAL_API_KEY", ""),
              "llm_model": os.getenv("LLM_MODEL", "llama3.2")}.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource
def _build_pipeline(provider, key, model):
    from pipeline import NotesRAGPipeline
    return NotesRAGPipeline(ocr_provider=provider, ocr_api_key=key, llm_model=model)


def get_pipeline():
    if st.session_state.pipeline is None:
        st.session_state.pipeline = _build_pipeline(
            st.session_state.ocr_provider,
            st.session_state.ocr_api_key,
            st.session_state.llm_model,
        )
    return st.session_state.pipeline


def _badge(conf):
    if conf >= 0.75:
        return f'<span class="badge badge-high">● High Confidence {int(conf*100)}%</span>'
    if conf >= 0.45:
        return f'<span class="badge badge-med">◐ Medium Confidence {int(conf*100)}%</span>'
    return f'<span class="badge badge-low">○ Low / No Info {int(conf*100)}%</span>'


def _src_card(src):
    sc = src.get("score", 0)
    col = "#4ade80" if sc >= 0.75 else ("#fbbf24" if sc >= 0.45 else "#f87171")
    return (f'<div class="src-card">📄 <b>{src["pdf"]}</b> — Page {src["page"]} '
            f'<span style="color:{col}">▪ {int(sc*100)}% relevance</span>'
            f'<br><span style="opacity:0.7">{src.get("preview","")[:90]}…</span></div>')


def _render_qa(qa):
    st.markdown(f'<div class="bubble-user">🧑 <b>You:</b> {qa.get("question","")}</div>',
                unsafe_allow_html=True)
    ans = qa.get("answer", "").replace("\n", "<br>")
    srcs_html = "".join(_src_card(s) for s in qa.get("sources", [])[:3])
    if srcs_html:
        srcs_html = ("<div style='margin-top:10px'><span style='font-size:0.75rem;color:#3d5070;"
                     "font-weight:700;letter-spacing:1px;text-transform:uppercase'>📍 Sources</span>"
                     + srcs_html + "</div>")
    st.markdown(
        f'<div class="bubble-bot">🤖 <b>Assistant:</b><br>{ans}<br>'
        f'{_badge(qa.get("confidence", 0))}{srcs_html}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="lbl">⚙️ Configuration</div>', unsafe_allow_html=True)

    new_prov = st.selectbox("OCR Provider", ["mistral", "google", "pymupdf"],
        index=["mistral","google","pymupdf"].index(st.session_state.ocr_provider))
    if new_prov != st.session_state.ocr_provider:
        st.session_state.ocr_provider = new_prov
        st.session_state.pipeline = None

    new_key = st.text_input("OCR API Key", type="password", value=st.session_state.ocr_api_key)
    if new_key != st.session_state.ocr_api_key:
        st.session_state.ocr_api_key = new_key
        st.session_state.pipeline = None

    model_opts = ["llama3.2","llama3.2:1b","mistral","mistral:7b","phi3","phi3:mini","gemma2:2b","qwen2.5:3b"]
    cur_idx = model_opts.index(st.session_state.llm_model) if st.session_state.llm_model in model_opts else 0
    new_model = st.selectbox("Local LLM (Ollama)", model_opts, index=cur_idx)
    if new_model != st.session_state.llm_model:
        st.session_state.llm_model = new_model
        st.session_state.pipeline = None

    st.markdown('<div class="lbl">📊 System Status</div>', unsafe_allow_html=True)
    if st.button("🔄 Check System", use_container_width=True):
        with st.spinner("Checking …"):
            try:
                st.session_state.sys_status = get_pipeline().system_check()
            except Exception as e:
                st.error(str(e))

    if st.session_state.sys_status:
        s = st.session_state.sys_status
        st.markdown(
            f"{'🟢' if s['ollama_running'] else '🔴'} Ollama: {'Running' if s['ollama_running'] else '**Not running** — `ollama serve`'}  \n"
            f"{'🟢' if s['model_available'] else '🟡'} Model: `{s['llm_model']}`  \n"
            f"{'🟢' if s['ocr_key_set'] else '🟡'} OCR Key: {'Set ✓' if s['ocr_key_set'] else 'Not set'}  \n"
            f"📦 Chunks in DB: **{s['db_stats'].get('total_chunks', 0)}**"
        )
        if s.get("available_models"):
            with st.expander("Available models"):
                for m in s["available_models"]: st.code(m)

    st.markdown('<div class="lbl">📚 Indexed PDFs</div>', unsafe_allow_html=True)
    try:
        db = get_pipeline().get_db_stats()
        pdfs = db.get("pdfs", [])
        if pdfs:
            for p in pdfs:
                pg = db.get("pages_per_pdf", {}).get(p, "?")
                st.markdown(f"📄 `{p}` — {pg} page(s)")
            st.markdown(f"**{db['total_chunks']} total chunks**")
        else:
            st.markdown('<div class="up-hint">No PDFs indexed yet</div>', unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.qa_history = []
            try: get_pipeline().clear_history()
            except Exception: pass
            st.rerun()
    with cb:
        if st.button("📥 Export PDF", use_container_width=True):
            if st.session_state.qa_history:
                try:
                    from report import generate_qa_report
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        rpath = f.name
                    generate_qa_report(st.session_state.qa_history, rpath)
                    with open(rpath, "rb") as f:
                        st.download_button("⬇️ Download Report", f,
                            file_name=f"qa_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf", use_container_width=True)
                    os.unlink(rpath)
                except Exception as e:
                    st.error(str(e))
            else:
                st.info("No Q&A to export yet.")

    try:
        pdfs_avail = get_pipeline().list_pdfs()
        if pdfs_avail:
            st.markdown('<div class="lbl">🗑️ Remove PDF</div>', unsafe_allow_html=True)
            del_p = st.selectbox("PDF to remove", ["— select —"] + pdfs_avail, label_visibility="collapsed")
            if del_p != "— select —" and st.button(f"Remove '{del_p}'", use_container_width=True):
                get_pipeline().delete_pdf(del_p)
                st.session_state.pipeline = None
                st.rerun()
    except Exception:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📝 HandNotes RAG</h1>
    <p>Upload handwritten notes · Ask anything · Get grounded answers with exact source citations</p>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_upload, tab_browse, tab_about = st.tabs(
    ["💬 Ask Questions", "📤 Upload PDF", "🗂️ Browse Chunks", "ℹ️ How It Works"])


# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown('<div class="lbl">📤 Upload & Index a PDF</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1: force = st.checkbox("Force re-OCR (bypass cache)", value=False)
    with c2: custom_name = st.text_input("Custom name (optional)", placeholder="e.g. biology_ch3")
    with c3: ingest_btn = st.button("🚀 Process & Index", type="primary", use_container_width=True)

    if uploaded and ingest_btn:
        # Always rebuild pipeline so latest sidebar API key is used
        st.session_state.pipeline = None
        with st.spinner(f"Processing '{uploaded.name}' … OCR may take 1–3 min for 19 pages (4s delay per page) …"):
            try:
                suffix = Path(uploaded.name).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                final_name = (custom_name.strip() or Path(uploaded.name).stem)
                named_path = os.path.join(os.path.dirname(tmp_path), final_name + ".pdf")
                if os.path.exists(named_path):
                    os.remove(named_path)
                os.rename(tmp_path, named_path)
                result = get_pipeline().ingest_pdf(named_path, force_reocr=force)
                os.unlink(named_path)
                st.success(f"✅ Indexed **{result['pdf_name']}** successfully!")
                m1,m2,m3,m4 = st.columns(4)
                for col, num, lbl in [(m1,result["pages"],"Pages"),
                                       (m2,result["chunks_created"],"Chunks"),
                                       (m3,result["chunks_stored"],"Stored"),
                                       (m4,f"{result['total_chars']:,}","Chars")]:
                    col.markdown(f'<div class="mpill"><div class="num">{num}</div>'
                                 f'<div class="name">{lbl}</div></div>', unsafe_allow_html=True)
                if result.get("stats"):
                    with st.expander("Chunking details"):
                        st.json(result["stats"])
            except Exception as exc:
                st.error(f"❌ {exc}")
                st.code(traceback.format_exc())
    elif not uploaded:
        st.markdown('<div class="up-hint"><div style="font-size:2.5rem">📂</div>'
                    '<div>Drag and drop a PDF, then click <b>Process &amp; Index</b></div>'
                    '<div style="font-size:0.8rem;margin-top:6px;color:#2a3a58">'
                    'Supports handwritten, printed, or mixed · Multiple PDFs supported</div></div>',
                    unsafe_allow_html=True)


# ── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    try:
        db_stats  = get_pipeline().get_db_stats()
        has_data  = db_stats.get("total_chunks", 0) > 0
        pdfs_list = db_stats.get("pdfs", [])
    except Exception:
        has_data, pdfs_list = False, []

    if not has_data:
        st.warning("⚠️ No notes indexed yet. Go to **Upload PDF** tab first.")
    else:
        if st.session_state.qa_history:
            st.markdown('<div class="lbl">💬 Conversation</div>', unsafe_allow_html=True)
            for qa in st.session_state.qa_history:
                _render_qa(qa)
        else:
            st.markdown('<div class="up-hint"><div style="font-size:2rem">💬</div>'
                        '<div>Ask your first question below!</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="lbl">💡 Ask a Question</div>', unsafe_allow_html=True)
        with st.form("q_form", clear_on_submit=True):
            question = st.text_input("q", placeholder="e.g. What is the formula on page 2?",
                                     label_visibility="collapsed")
            fa, fb, fc = st.columns([4, 2, 2])
            with fa: top_k = st.slider("Context chunks", 3, 10, 5)
            with fb:
                pdf_choice = st.selectbox("Search in", ["All PDFs"] + pdfs_list)
            with fc:
                submitted = st.form_submit_button("Ask →", type="primary", use_container_width=True)

        if submitted and question.strip():
            pdf_filter = None if pdf_choice == "All PDFs" else pdf_choice
            with st.spinner("🔍 Searching and generating …"):
                try:
                    result = get_pipeline().ask(question.strip(), top_k=top_k, pdf_filter=pdf_filter)
                    result["timestamp"] = datetime.now().isoformat()
                    st.session_state.qa_history.append(result)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
                    st.code(traceback.format_exc())

        if not st.session_state.qa_history and has_data:
            st.markdown('<div class="lbl">✨ Try these</div>', unsafe_allow_html=True)
            suggestions = ["What are the main topics?", "Summarise page 1",
                           "What does the diagram show?", "List all definitions",
                           "Are there any formulas?", "Key takeaways?"]
            for row_start in [0, 3]:
                cols = st.columns(3)
                for col, sug in zip(cols, suggestions[row_start:row_start+3]):
                    if col.button(sug, use_container_width=True):
                        with st.spinner("Thinking …"):
                            try:
                                r = get_pipeline().ask(sug, top_k=5)
                                r["timestamp"] = datetime.now().isoformat()
                                st.session_state.qa_history.append(r)
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))


# ── Browse tab ────────────────────────────────────────────────────────────────
with tab_browse:
    st.markdown('<div class="lbl">🗂️ Browse Indexed Chunks</div>', unsafe_allow_html=True)
    st.caption("Inspect what's stored in the vector database.")
    try:
        db = get_pipeline().get_db_stats()
        if not db.get("total_chunks"):
            st.info("No chunks indexed yet.")
        else:
            bf1, bf2 = st.columns(2)
            with bf1: browse_pdf = st.selectbox("Filter PDF", ["All"] + db.get("pdfs", []))
            with bf2: kw = st.text_input("Keyword filter", placeholder="search text …")
            if st.button("🔍 Load Chunks"):
                raw = get_pipeline().embedder.col.get()
                rows = [{"ID": raw["ids"][i], "PDF": raw["metadatas"][i]["pdf"],
                         "Page": raw["metadatas"][i]["page"],
                         "Chunk": raw["metadatas"][i]["chunk_index"],
                         "Text": raw["documents"][i]}
                        for i in range(len(raw["documents"]))]
                if browse_pdf != "All": rows = [r for r in rows if r["PDF"] == browse_pdf]
                if kw.strip(): rows = [r for r in rows if kw.lower() in r["Text"].lower()]
                st.markdown(f"Showing **{min(len(rows),50)}** of **{len(rows)}** chunks")
                for r in rows[:50]:
                    with st.expander(f"📄 {r['PDF']} — Page {r['Page']} — Chunk #{r['Chunk']}"):
                        st.write(r["Text"])
    except Exception as e:
        st.info(f"Index a PDF first. ({e})")


# ── About tab ─────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown('<div class="lbl">🏗️ Architecture</div>', unsafe_allow_html=True)
    st.code("""
PDF Upload
    │
    ▼
🔍 OCR  (Mistral Vision / Google Vision / PyMuPDF)   ← cloud allowed here only
    ▼
✂️  Chunking  — 400-char overlapping windows, sentence-aware
    ▼
🧮 Embeddings  — sentence-transformers all-MiniLM-L6-v2  (local)
    ▼
🗄  ChromaDB  — persistent local vector store (cosine similarity)
    │
    │         User asks a question
    │                │
    ▼                ▼
🔎 Retrieval — embed query → top-k cosine search
    ▼
🤖 Llama 3.2 via Ollama  (100% offline)
    ▼
📍 Answer + Source (PDF + page) + Confidence score
""", language="")

    st.markdown("""
**Stack:** Mistral Vision OCR · PyMuPDF · sentence-transformers · ChromaDB · Ollama Llama 3.2 · Streamlit · ReportLab

**Anti-hallucination design:**
- Score threshold gate: if top retrieval score < 0.30 → system returns "I don't know" *before* calling LLM
- Strict prompt: forbids outside knowledge, demands inline citations
- Temperature 0.1: deterministic, factual output
- Confidence badge: every answer labelled High / Medium / Low
""")