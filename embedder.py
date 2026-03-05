"""
embedder.py — Local embeddings (sentence-transformers) + ChromaDB vector store.
100% offline after the first model download (~80 MB).
"""

from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH,
)


class NoteEmbedder:
    """Handles embedding generation and vector store operations."""

    def __init__(self):
        print(f"🤖 Loading embedding model '{EMBEDDING_MODEL}' …")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"🗄  Connecting to ChromaDB at {CHROMA_DB_PATH} …")
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  Chunks in DB: {self.col.count()}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def embed_and_store(self, chunks: List[Dict]) -> int:
        """
        Embed chunks and upsert into ChromaDB.
        Skips chunks whose IDs already exist (idempotent).

        Returns number of newly stored chunks.
        """
        if not chunks:
            return 0

        # Find which chunk IDs are new
        existing_ids: set = set()
        if self.col.count() > 0:
            existing_ids = set(self.col.get()["ids"])

        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            print("  All chunks already indexed, nothing to add.")
            return 0

        print(f"  Embedding {len(new_chunks)} new chunks …")
        texts = [c["text"] for c in new_chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        self.col.add(
            ids       = [c["chunk_id"]    for c in new_chunks],
            embeddings= embeddings,
            documents = texts,
            metadatas = [
                {
                    "page":        c["page"],
                    "pdf":         c["pdf"],
                    "chunk_index": c["chunk_index"],
                    "preview":     c["preview"],
                }
                for c in new_chunks
            ],
        )
        print(f"  ✅ Stored {len(new_chunks)} chunks. DB total: {self.col.count()}")
        return len(new_chunks)

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        pdf_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Semantic search for the most relevant chunks.

        Args:
            query:      Natural-language question.
            top_k:      Max results to return.
            pdf_filter: If set, restrict search to this PDF name.

        Returns:
            List of chunk dicts with an added 'score' (0–1, higher = better).
        """
        if self.col.count() == 0:
            return []

        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        where = {"pdf": pdf_filter} if pdf_filter else None
        n     = min(top_k, self.col.count())

        res = self.col.query(
            query_embeddings=q_emb,
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(res["ids"][0])):
            dist  = res["distances"][0][i]
            score = round(1.0 - dist, 4)   # cosine distance → similarity
            meta  = res["metadatas"][0][i]
            hits.append({
                "chunk_id":    res["ids"][0][i],
                "text":        res["documents"][0][i],
                "page":        meta["page"],
                "pdf":         meta["pdf"],
                "chunk_index": meta["chunk_index"],
                "preview":     meta["preview"],
                "score":       score,
            })

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        count = self.col.count()
        if count == 0:
            return {"total_chunks": 0, "pdfs": [], "pages_per_pdf": {}}
        data  = self.col.get()
        pdfs  = {}
        for m in data["metadatas"]:
            pdfs.setdefault(m["pdf"], set()).add(m["page"])
        return {
            "total_chunks": count,
            "pdfs":         list(pdfs.keys()),
            "pages_per_pdf": {k: len(v) for k, v in pdfs.items()},
        }

    def list_pdfs(self) -> List[str]:
        return sorted(self.get_stats().get("pdfs", []))

    def delete_pdf(self, pdf_name: str) -> None:
        self.col.delete(where={"pdf": pdf_name})
        print(f"🗑  Deleted all chunks for '{pdf_name}'")
