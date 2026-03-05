"""
chunker.py — Split OCR text into overlapping, searchable chunks.
Each chunk carries page number and PDF name for exact source citations.
"""

import re
from typing import List, Dict

from config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-level units, respecting newlines and bullet points."""
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_page(page_num: int, text: str, pdf_name: str) -> List[Dict]:
    """
    Produce overlapping chunks for a single page.

    Returns list of chunk dicts with metadata.
    """
    if not text.strip():
        return []

    sentences = _split_sentences(text)
    chunks: List[Dict] = []
    current = ""
    sent_buf: List[str] = []
    idx = 0

    for sent in sentences:
        if len(current) + len(sent) > CHUNK_SIZE and current:
            chunks.append(_make_chunk(pdf_name, page_num, idx, current.strip()))
            idx += 1
            # overlap: keep last N chars worth of sentences
            overlap_sents = []
            overlap_len = 0
            for s in reversed(sent_buf):
                if overlap_len + len(s) <= CHUNK_OVERLAP:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = " ".join(overlap_sents) + " " + sent
            sent_buf = overlap_sents + [sent]
        else:
            current = (current + " " + sent).strip()
            sent_buf.append(sent)

    if current.strip():
        chunks.append(_make_chunk(pdf_name, page_num, idx, current.strip()))

    return chunks


def _make_chunk(pdf_name: str, page: int, idx: int, text: str) -> Dict:
    return {
        "chunk_id":    f"{pdf_name}__p{page}__c{idx}",
        "text":        text,
        "page":        page,
        "pdf":         pdf_name,
        "chunk_index": idx,
        "preview":     text[:100],
    }


def chunk_all_pages(pages: List[Dict], pdf_name: str) -> List[Dict]:
    """
    Chunk all pages from OCR output.

    Args:
        pages:    List of {page, text} from ocr.py
        pdf_name: Identifier used in chunk IDs and metadata

    Returns:
        Flat list of all chunk dicts
    """
    all_chunks: List[Dict] = []
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data.get("text", "")
        if not text.strip():
            print(f"  ⚠  Page {page_num} empty, skipping.")
            continue
        page_chunks = chunk_page(page_num, text, pdf_name)
        all_chunks.extend(page_chunks)
        print(f"  Page {page_num}: {len(page_chunks)} chunks")

    print(f"  ✅ Total chunks: {len(all_chunks)}")
    return all_chunks


def chunk_stats(chunks: List[Dict]) -> Dict:
    if not chunks:
        return {}
    lengths = [len(c["text"]) for c in chunks]
    pages   = sorted(set(c["page"] for c in chunks))
    return {
        "total_chunks":     len(chunks),
        "pages_covered":    len(pages),
        "avg_chunk_chars":  int(sum(lengths) / len(lengths)),
        "min_chunk_chars":  min(lengths),
        "max_chunk_chars":  max(lengths),
    }
