"""
pipeline.py — Orchestrates the complete RAG pipeline.

Steps:
  1. process_pdf  → OCR (with cache)
  2. chunk_all_pages → overlapping text chunks
  3. embed_and_store → embeddings into ChromaDB
  4. search         → cosine similarity retrieval
  5. generate_answer → Ollama LLM with strict grounding
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

from ocr       import process_pdf, cache_exists, save_cache, load_cache
from chunker   import chunk_all_pages, chunk_stats
from embedder  import NoteEmbedder
from llm       import generate_answer, is_ollama_running, list_models, model_exists, confidence_label
from config    import DEFAULT_LLM_MODEL, DEFAULT_TOP_K, MAX_HISTORY_TURNS


class NotesRAGPipeline:
    """Single entry-point for all RAG operations."""

    def __init__(
        self,
        ocr_provider: str  = "mistral",
        ocr_api_key:  str  = "",
        llm_model:    str  = DEFAULT_LLM_MODEL,
    ):
        self.ocr_provider = ocr_provider
        self.ocr_api_key  = ocr_api_key
        self.llm_model    = llm_model

        self.embedder: NoteEmbedder = NoteEmbedder()
        self.history:  List[Dict]   = []   # conversation memory

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: str, force_reocr: bool = False) -> Dict:
        """
        Full ingestion: OCR → chunk → embed → store.

        Args:
            pdf_path:    Absolute path to PDF on disk.
            force_reocr: Bypass the OCR cache and re-process.

        Returns:
            Stats dict with page count, chunk counts, char count.
        """
        pdf_name = Path(pdf_path).stem

        # ── Step 1: OCR ───────────────────────────────────────────────────────
        if cache_exists(pdf_name) and not force_reocr:
            print(f"📦 Loading cached OCR for '{pdf_name}' …")
            pages = load_cache(pdf_name)
        else:
            pages = process_pdf(pdf_path, self.ocr_provider, self.ocr_api_key)
            save_cache(pages, pdf_name)

        total_chars = sum(len(p.get("text", "")) for p in pages)
        print(f"  Extracted {total_chars:,} chars from {len(pages)} pages")

        # ── Step 2: Chunk ─────────────────────────────────────────────────────
        print("✂️  Chunking …")
        chunks = chunk_all_pages(pages, pdf_name)
        stats  = chunk_stats(chunks)

        # ── Step 3: Embed + Store ─────────────────────────────────────────────
        print("🧮 Embedding + storing …")
        stored = self.embedder.embed_and_store(chunks)

        return {
            "pdf_name":       pdf_name,
            "pages":          len(pages),
            "chunks_created": len(chunks),
            "chunks_stored":  stored,
            "total_chars":    total_chars,
            "stats":          stats,
        }

    # ── Q&A ───────────────────────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        top_k:    int           = DEFAULT_TOP_K,
        pdf_filter: Optional[str] = None,
    ) -> Dict:
        """
        Answer a question using the indexed notes.

        Args:
            question:   Natural-language question.
            top_k:      Number of context chunks to retrieve.
            pdf_filter: Restrict search to a specific PDF (None = all PDFs).

        Returns:
            Result dict: {question, answer, sources, confidence, …}
        """
        print(f"\n🔍 Question: {question}")

        # ── Step 4: Retrieve ──────────────────────────────────────────────────
        chunks = self.embedder.search(question, top_k=top_k, pdf_filter=pdf_filter)
        if chunks:
            print(f"  Top hit: score={chunks[0]['score']:.3f} | "
                  f"pdf={chunks[0]['pdf']} | page={chunks[0]['page']}")

        # ── Step 5: Generate ──────────────────────────────────────────────────
        result = generate_answer(
            query          = question,
            context_chunks = chunks,
            model          = self.llm_model,
            history        = self.history,
        )

        # Update conversation memory
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": result["answer"]})
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self.history = self.history[-(MAX_HISTORY_TURNS * 2):]

        result["question"] = question
        return result

    # ── Utilities ─────────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        self.history = []
        print("🗑  Conversation history cleared.")

    def get_db_stats(self) -> Dict:
        return self.embedder.get_stats()

    def list_pdfs(self) -> List[str]:
        return self.embedder.list_pdfs()

    def delete_pdf(self, pdf_name: str) -> None:
        self.embedder.delete_pdf(pdf_name)

    def system_check(self) -> Dict:
        """Return status of all components."""
        ollama_ok  = is_ollama_running()
        models     = list_models() if ollama_ok else []
        model_ok   = any(self.llm_model in m for m in models)
        db_stats   = self.get_db_stats()
        return {
            "ollama_running":  ollama_ok,
            "available_models": models,
            "llm_model":       self.llm_model,
            "model_available": model_ok,
            "ocr_provider":    self.ocr_provider,
            "ocr_key_set":     bool(self.ocr_api_key),
            "db_stats":        db_stats,
        }
