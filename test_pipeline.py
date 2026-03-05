"""
test_pipeline.py — Smoke tests for the RAG pipeline.
Run: python test_pipeline.py

Tests work offline (uses PyMuPDF + a tiny synthetic PDF + Ollama).
"""

import os
import sys
import tempfile
import unittest

# Suppress noisy logs during tests
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class TestChunker(unittest.TestCase):
    def test_basic_chunking(self):
        from chunker import chunk_page, chunk_all_pages
        text = "This is the first sentence. " * 30
        chunks = chunk_page(1, text, "test_pdf")
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertIn("chunk_id", c)
            self.assertIn("page", c)
            self.assertEqual(c["page"], 1)
            self.assertEqual(c["pdf"], "test_pdf")

    def test_empty_page(self):
        from chunker import chunk_page
        self.assertEqual(chunk_page(1, "", "test"), [])
        self.assertEqual(chunk_page(1, "   \n  ", "test"), [])

    def test_chunk_all_pages(self):
        from chunker import chunk_all_pages
        pages = [
            {"page": 1, "text": "Topic A. " * 50},
            {"page": 2, "text": "Topic B. " * 50},
            {"page": 3, "text": ""},  # empty page
        ]
        chunks = chunk_all_pages(pages, "multi_page_test")
        pages_covered = set(c["page"] for c in chunks)
        self.assertIn(1, pages_covered)
        self.assertIn(2, pages_covered)
        self.assertNotIn(3, pages_covered)


class TestEmbedder(unittest.TestCase):
    def setUp(self):
        # Use a temp directory so tests don't pollute the real DB
        self._tmpdir = tempfile.mkdtemp()
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL, CHROMA_COLLECTION

        class _Embedder:
            def __init__(self, tmpdir):
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self.client = chromadb.PersistentClient(
                    path=tmpdir,
                    settings=Settings(anonymized_telemetry=False),
                )
                self.col = self.client.get_or_create_collection(
                    CHROMA_COLLECTION + "_test",
                    metadata={"hnsw:space": "cosine"},
                )

            def embed_and_store(self, chunks):
                texts = [c["text"] for c in chunks]
                embs = self.model.encode(texts, normalize_embeddings=True).tolist()
                self.col.add(
                    ids=[c["chunk_id"] for c in chunks],
                    embeddings=embs,
                    documents=texts,
                    metadatas=[{"page": c["page"], "pdf": c["pdf"],
                                "chunk_index": c["chunk_index"], "preview": c["preview"]}
                               for c in chunks],
                )
                return len(chunks)

            def search(self, query, top_k=3):
                q_emb = self.model.encode([query], normalize_embeddings=True).tolist()
                res = self.col.query(query_embeddings=q_emb, n_results=top_k,
                                     include=["documents","metadatas","distances"])
                return [{"text": res["documents"][0][i],
                         "page": res["metadatas"][0][i]["page"],
                         "pdf":  res["metadatas"][0][i]["pdf"],
                         "score": round(1.0 - res["distances"][0][i], 4)}
                        for i in range(len(res["ids"][0]))]

        self.embedder = _Embedder(self._tmpdir)

    def _make_chunk(self, cid, text, page=1, pdf="test"):
        return {"chunk_id": cid, "text": text, "page": page, "pdf": pdf,
                "chunk_index": 0, "preview": text[:50]}

    def test_store_and_retrieve(self):
        chunks = [
            self._make_chunk("c1", "The heart pumps blood through the body."),
            self._make_chunk("c2", "Photosynthesis converts sunlight to energy in plants."),
            self._make_chunk("c3", "Newton's laws describe motion and force."),
        ]
        stored = self.embedder.embed_and_store(chunks)
        self.assertEqual(stored, 3)

        results = self.embedder.search("How does the heart work?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("heart", results[0]["text"].lower())

    def test_semantic_search(self):
        chunks = [
            self._make_chunk("s1", "The mitochondria is the powerhouse of the cell."),
            self._make_chunk("s2", "Brazil is a country in South America."),
            self._make_chunk("s3", "DNA carries genetic information."),
        ]
        self.embedder.embed_and_store(chunks)
        # "cell energy factory" should match mitochondria, not geography
        results = self.embedder.search("cell energy production", top_k=1)
        self.assertGreater(results[0]["score"], 0.3)


class TestOCRClean(unittest.TestCase):
    def test_clean(self):
        from ocr import _clean
        raw = "hello   world\n\n\n\nextra spaces"
        cleaned = _clean(raw)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertNotIn("   ", cleaned)


class TestLLMFallback(unittest.TestCase):
    def test_empty_context_returns_no_info(self):
        from llm import generate_answer, NO_INFO_REPLY
        result = generate_answer("What is X?", context_chunks=[], model="llama3.2")
        self.assertEqual(result["answer"], NO_INFO_REPLY)
        self.assertFalse(result["is_grounded"])

    def test_low_score_returns_no_info(self):
        from llm import generate_answer, NO_INFO_REPLY
        low_chunks = [{"chunk_id": "x", "text": "irrelevant text", "page": 1,
                       "pdf": "p", "chunk_index": 0, "preview": "...", "score": 0.1}]
        result = generate_answer("What is X?", context_chunks=low_chunks, model="llama3.2")
        self.assertEqual(result["answer"], NO_INFO_REPLY)

    def test_confidence_label(self):
        from llm import confidence_label
        self.assertEqual(confidence_label(0.9)[0], "High Confidence")
        self.assertEqual(confidence_label(0.6)[0], "Medium Confidence")
        self.assertEqual(confidence_label(0.2)[0], "Low / No Info")


if __name__ == "__main__":
    print("🧪 Running HandNotes RAG unit tests …\n")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
