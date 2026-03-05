# 📝 HandNotes RAG — Gen AI Hackathon 2026

> **Build Smart. Retrieve Precisely. Answer Honestly.**

A fully local RAG system that reads handwritten PDF notes and answers questions based **only** on what is written — never hallucinating, always citing the exact source.

---

## 🏗️ Architecture

```
PDF Upload
    ↓
🔍 OCR  (Mistral OCR API — /v1/ocr endpoint)     ← cloud, allowed
    ↓
✂️  Smart Chunking  (400 chars + 80 char overlap)
    ↓
🧮 Embeddings  (all-MiniLM-L6-v2)                ← 100% local
    ↓
🗄️  ChromaDB  (cosine similarity search)          ← 100% local
    ↓
💬 User Question → top-k retrieval
    ↓
🤖 Llama 3.2 via Ollama                           ← 100% local
    ↓
📍 Answer + Source Citations + Confidence Score
```

---

## ⚙️ Requirements

- Python **3.10** (required — not 3.11+)
- [Ollama](https://ollama.com/download) installed
- A free [Mistral API key](https://console.mistral.ai) for OCR

---

## ⚡ Setup — Step by Step

### Step 1 — Clone / download the project

```
handnotes-rag/
├── app.py
├── pipeline.py
├── ocr.py
├── chunker.py
├── embedder.py
├── llm.py
├── report.py
├── config.py
├── requirements.txt
├── .env.example
└── README.md
```

### Step 2 — Install Python 3.10

Download from https://www.python.org/downloads/release/python-31011/

> ⚠️ During install, check **"Add Python to PATH"**

Verify:
```bash
py -3.10 --version
```

### Step 3 — Install dependencies

```bash
py -3.10 -m pip install -r requirements.txt
py -3.10 -m pip install tf-keras
```

### Step 4 — Install Ollama + pull model

Download Ollama from https://ollama.com/download/windows

Then pull the LLM model (~2 GB):
```bash
ollama pull llama3.2
```

Or lighter version (1.3 GB):
```bash
ollama pull llama3.2:1b
```

### Step 5 — Get a free Mistral API key

1. Go to https://console.mistral.ai
2. Sign up for a free account
3. Click **API Keys** → **Create new key**
4. Copy the key

### Step 6 — Create your .env file

Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

Open `.env` in Notepad and fill in:
```
MISTRAL_API_KEY=your_mistral_key_here
LLM_MODEL=llama3.2
```

### Step 7 — Run the app

Open **two terminals** in the project folder:

**Terminal 1** — start Ollama:
```bash
ollama serve
```

**Terminal 2** — start the app:
```bash
py -3.10 -m streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 🚀 How to Use

1. **Upload PDF** tab → drag and drop your handwritten notes PDF → click **Process & Index**
2. Wait for OCR + indexing to finish (takes ~30 seconds for a 20-page PDF)
3. **Ask Questions** tab → type any question → click **Ask →**
4. Get an answer with source citation (PDF name + page number)
5. **Sidebar → Generate Q&A Report** → download a PDF of the full session

---

## ✅ Features

| Feature | Status |
|---|---|
| Handwriting OCR (Mistral OCR API) | ✅ |
| Smart overlapping chunking | ✅ |
| Local vector embeddings (all-MiniLM-L6-v2) | ✅ |
| ChromaDB local vector store | ✅ |
| Local LLM answers (Llama 3.2 via Ollama) | ✅ |
| Source citations (PDF + page number) | ✅ |
| "I don't know" when answer not in notes | ✅ |
| Confidence scoring (High/Medium/Low) | ✅ (Bonus) |
| PDF Q&A report export | ✅ (Bonus) |
| Conversation memory (follow-up questions) | ✅ (Bonus) |
| OCR result caching | ✅ (Bonus) |
| Multi-PDF support | ✅ (Bonus) |
| CLI interface | ✅ (Bonus) |

---

## 📁 File Structure

```
app.py              → Streamlit UI (main entry point)
pipeline.py         → Orchestrates all 7 RAG steps
ocr.py              → PDF → text via Mistral OCR API
chunker.py          → Sentence-aware overlapping text splitting
embedder.py         → Embeddings + ChromaDB vector store
llm.py              → Ollama answer generation (strict grounding)
report.py           → PDF report generator (bonus)
config.py           → All settings in one place
cli.py              → Command-line interface
test_pipeline.py    → Unit tests
test_ocr.py         → OCR API diagnostic tool
requirements.txt    → Python dependencies
.env.example        → Environment variable template
README.md           → This file
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit not found` | Use `py -3.10 -m streamlit run app.py` |
| `No module named fitz` | Run `py -3.10 -m pip install PyMuPDF==1.24.14` |
| `Keras error` | Run `py -3.10 -m pip install tf-keras` |
| Ollama not running | Run `ollama serve` in a separate terminal |
| Model not found | Run `ollama pull llama3.2` |
| OCR returns 0 chunks | Check Mistral API key in `.env`, delete `ocr_cache/` folder and retry |
| WinError 183 on upload | Already fixed — re-download `app.py` |
| ChromaDB errors | Delete `chroma_db/` folder and re-ingest |

---

## 🏆 Scoring Alignment

| Criteria | Weight | Our Approach |
|---|---|---|
| Correct Answers | 40% | Mistral OCR + overlap chunking + strict RAG prompt |
| RAG Workflow | 20% | Clean 7-step pipeline, retrieval and generation separated |
| Bonus Features | 20% | PDF reports + confidence scores + memory + CLI |
| UI Design | 10% | Dark scholarly Streamlit theme |
| Presentation | 10% | See demo flow below |

---

## 🎤 Demo Flow (for judges)

1. Show **System Status** — all green lights
2. Upload PDF → show OCR + chunking stats
3. Ask a clear question → show answer with source citation
4. Ask a **trick question** → show "I don't know" (no hallucination)
5. Ask a **follow-up** → show conversation memory working
6. Upload a **second PDF** → ask a cross-PDF question
7. Click **Generate Q&A Report** → download PDF

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| OCR | Mistral OCR API (`mistral-ocr-latest`) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (local, persistent) |
| LLM | Llama 3.2 via Ollama (fully offline) |
| UI | Streamlit |
| PDF Reports | ReportLab |
| PDF Parsing | PyMuPDF |









<img width="1920" height="1020" alt="HandNotes RAG - Google Chrome 05-03-2026 15_51_12" src="https://github.com/user-attachments/assets/39bf7f69-701d-4fcf-977d-95766e66b618" />
<img width="1920" height="1020" alt="HandNotes RAG - Google Chrome 05-03-2026 15_51_25" src="https://github.com/user-attachments/assets/fd9311fa-9a10-40d6-bb1e-9e333547525f" />
<img width="1920" height="1020" alt="HandNotes RAG - Google Chrome 05-03-2026 15_51_33" src="https://github.com/user-attachments/assets/05ecb9b9-6b6e-4dfb-bb5b-f2e77dc9b31b" />
<img width="1920" height="1020" alt="HandNotes RAG - Google Chrome 05-03-2026 15_51_43" src="https://github.com/user-attachments/assets/abf15d4b-9584-42b0-8200-38cf466c16e0" />
