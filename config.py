"""
config.py — Central configuration for HandNotes RAG.
All tunable parameters live here. Edit this file to customise behaviour.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
CHROMA_DB_PATH  = str(BASE_DIR / "chroma_db")
OCR_CACHE_DIR   = str(BASE_DIR / "ocr_cache")
REPORTS_DIR     = str(BASE_DIR / "reports")

for _d in [CHROMA_DB_PATH, OCR_CACHE_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_PROVIDER        = os.getenv("OCR_PROVIDER", "mistral")   # mistral | google | pymupdf
MISTRAL_API_KEY     = os.getenv("MISTRAL_API_KEY", "")
GOOGLE_API_KEY      = os.getenv("GOOGLE_VISION_API_KEY", "")
OCR_RENDER_DPI      = 300

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE          = 400
CHUNK_OVERLAP       = 80

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
EMBEDDING_BATCH     = 32
CHROMA_COLLECTION   = "notes_collection"

# ── LLM (Ollama) ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL   = os.getenv("LLM_MODEL", "llama3.2")
LLM_TEMPERATURE     = 0.1
LLM_MAX_TOKENS      = 512
LLM_TOP_P           = 0.9

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K           = 5
MIN_SIMILARITY_SCORE    = 0.30
HIGH_CONFIDENCE_SCORE   = 0.75
MED_CONFIDENCE_SCORE    = 0.45

# ── Conversation memory ───────────────────────────────────────────────────────
MAX_HISTORY_TURNS   = 10   # total pairs kept; last 4 sent to LLM
HISTORY_SENT_TO_LLM = 4
