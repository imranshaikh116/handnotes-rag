"""
cli.py — Command-line interface for HandNotes RAG.
Useful for quick testing without the Streamlit UI.

Usage:
    python cli.py ingest path/to/notes.pdf
    python cli.py ask "What is the main topic?"
    python cli.py ask "Explain the diagram" --top-k 7
    python cli.py stats
    python cli.py clear
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def cmd_ingest(args):
    from pipeline import NotesRAGPipeline
    pipeline = NotesRAGPipeline(
        ocr_provider=os.getenv("OCR_PROVIDER", "mistral"),
        ocr_api_key=os.getenv("MISTRAL_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "llama3.2"),
    )
    result = pipeline.ingest_pdf(args.pdf, force_reocr=args.force)
    print(f"\n✅ Ingested '{result['pdf_name']}'")
    print(f"   Pages:  {result['pages']}")
    print(f"   Chunks: {result['chunks_created']} created, {result['chunks_stored']} new")
    print(f"   Chars:  {result['total_chars']:,}")


def cmd_ask(args):
    from pipeline import NotesRAGPipeline
    pipeline = NotesRAGPipeline(
        ocr_provider=os.getenv("OCR_PROVIDER", "mistral"),
        ocr_api_key=os.getenv("MISTRAL_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "llama3.2"),
    )
    result = pipeline.ask(args.question, top_k=args.top_k)
    print(f"\n🤖 Answer:\n{result['answer']}")
    print(f"\n📊 Confidence: {result['confidence_label']} ({int(result['confidence']*100)}%)")
    if result.get("sources"):
        print("\n📍 Sources:")
        seen = set()
        for s in result["sources"][:3]:
            key = f"{s['pdf']}_{s['page']}"
            if key not in seen:
                seen.add(key)
                print(f"   • {s['pdf']} — Page {s['page']}  (score: {s['score']:.3f})")


def cmd_stats(args):
    from pipeline import NotesRAGPipeline
    pipeline = NotesRAGPipeline()
    stats = pipeline.get_db_stats()
    if not stats.get("total_chunks"):
        print("⚠  Database is empty. Ingest a PDF first.")
        return
    print(f"\n📦 Database stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   PDFs indexed: {', '.join(stats['pdfs'])}")
    for pdf, pages in stats.get("pages_per_pdf", {}).items():
        print(f"   • {pdf}: {pages} page(s)")


def cmd_clear(args):
    from pipeline import NotesRAGPipeline
    if args.pdf:
        NotesRAGPipeline().delete_pdf(args.pdf)
    else:
        import shutil
        from config import CHROMA_DB_PATH
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        print("🗑  ChromaDB cleared. All chunks deleted.")


def main():
    parser = argparse.ArgumentParser(description="HandNotes RAG CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into the vector store")
    p_ingest.add_argument("pdf", help="Path to PDF file")
    p_ingest.add_argument("--force", action="store_true", help="Force re-OCR")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--top-k", type=int, default=5, help="Context chunks to retrieve")
    p_ask.set_defaults(func=cmd_ask)

    p_stats = sub.add_parser("stats", help="Show database statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_clear = sub.add_parser("clear", help="Clear the database")
    p_clear.add_argument("--pdf", default=None, help="Only remove a specific PDF")
    p_clear.set_defaults(func=cmd_clear)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
