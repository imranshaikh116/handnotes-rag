"""
ocr.py — PDF → clean text.

Providers (in order of recommendation):
  1. mistral_ocr  — Mistral's dedicated OCR endpoint (truly free, no rate limits)
  2. google       — Google Cloud Vision (needs billing enabled)
  3. pymupdf      — Local fallback (typed PDFs only)
"""

from __future__ import annotations

import base64, json, os, re, time
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF

from config import OCR_CACHE_DIR, OCR_RENDER_DPI, MISTRAL_API_KEY, GOOGLE_API_KEY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_page_b64(page: fitz.Page, dpi: int = OCR_RENDER_DPI) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return base64.b64encode(pix.tobytes("png")).decode()


def _clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ── Provider 1: Mistral dedicated OCR API (free, no vision chat) ──────────────

def _mistral_ocr_api(pdf_path: str, api_key: str) -> List[Dict]:
    """Use Mistral's dedicated /v1/ocr endpoint — free tier, no rate limits."""
    import requests

    print("  Using Mistral dedicated OCR API (/v1/ocr) ...")

    # Upload PDF as file first
    try:
        with open(pdf_path, "rb") as f:
            upload_resp = requests.post(
                "https://api.mistral.ai/v1/files",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (Path(pdf_path).name, f, "application/pdf")},
                data={"purpose": "ocr"},
                timeout=120,
            )

        if upload_resp.status_code != 200:
            print(f"  ⚠ File upload failed: {upload_resp.status_code} — {upload_resp.text[:200]}")
            return _fallback_to_pymupdf(pdf_path)

        file_id = upload_resp.json()["id"]
        print(f"  ✅ PDF uploaded, file_id={file_id}")

        # Get signed URL
        url_resp = requests.get(
            f"https://api.mistral.ai/v1/files/{file_id}/url",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if url_resp.status_code != 200:
            print(f"  ⚠ Failed to get signed URL: {url_resp.text[:200]}")
            return _fallback_to_pymupdf(pdf_path)

        signed_url = url_resp.json()["url"]

        # Run OCR
        ocr_resp = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "mistral-ocr-latest",
                "document": {"type": "document_url", "document_url": signed_url},
                "include_image_base64": False,
            },
            timeout=300,
        )

        if ocr_resp.status_code == 200:
            pages_data = ocr_resp.json().get("pages", [])
            results = []
            for p in pages_data:
                text = p.get("markdown", "") or p.get("text", "")
                page_num = p.get("index", len(results)) + 1
                results.append({"page": page_num, "text": _clean(text), "raw": text})
            print(f"  ✅ Mistral OCR API success: {len(results)} pages extracted")

            # Clean up uploaded file
            requests.delete(
                f"https://api.mistral.ai/v1/files/{file_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            return results
        else:
            print(f"  ⚠ OCR request failed: {ocr_resp.status_code} — {ocr_resp.text[:300]}")
            return _fallback_to_pymupdf(pdf_path)

    except Exception as e:
        print(f"  ⚠ Mistral OCR API error: {e}")
        return _fallback_to_pymupdf(pdf_path)


# ── Provider 2: Google Cloud Vision ───────────────────────────────────────────

def _google_ocr(pdf_path: str, api_key: str) -> List[Dict]:
    import requests
    doc = fitz.open(pdf_path)
    results = []
    for i, page in enumerate(doc):
        print(f"  [Google Vision OCR] page {i+1}/{len(doc)} …")
        b64 = _render_page_b64(page, dpi=200)
        try:
            r = requests.post(
                f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
                json={"requests": [{"image": {"content": b64},
                                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]},
                timeout=45,
            )
            if r.status_code == 200:
                resp_data = r.json()["responses"][0]
                if "error" in resp_data:
                    print(f"  ⚠ Google error: {resp_data['error']}")
                    results.append({"page": i+1, "text": "", "raw": ""})
                else:
                    text = resp_data.get("fullTextAnnotation", {}).get("text", "")
                    results.append({"page": i+1, "text": _clean(text), "raw": text})
            else:
                print(f"  ⚠ Google OCR page {i+1}: HTTP {r.status_code}")
                results.append({"page": i+1, "text": "", "raw": ""})
        except Exception as e:
            print(f"  ⚠ Google OCR page {i+1}: {e}")
            results.append({"page": i+1, "text": "", "raw": ""})
    doc.close()
    return results


# ── Provider 3: PyMuPDF local (typed PDFs only) ────────────────────────────────

def _pymupdf_extract(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    results = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        results.append({"page": i+1, "text": _clean(text), "raw": text})
    doc.close()
    return results


def _fallback_to_pymupdf(pdf_path: str) -> List[Dict]:
    print("  Falling back to PyMuPDF (typed text only) …")
    return _pymupdf_extract(pdf_path)


# ── Public API ─────────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str, provider: str = "mistral", api_key: str = "") -> List[Dict]:
    """
    Extract text from every page of a PDF.
    provider: 'mistral' | 'google' | 'pymupdf'
    """
    print(f"📄 OCR: {Path(pdf_path).name}  [provider={provider}]")
    key = api_key.strip() if api_key else ""

    if provider == "mistral":
        key = key or MISTRAL_API_KEY.strip()
        if not key:
            print("  No Mistral key — falling back to PyMuPDF")
            return _pymupdf_extract(pdf_path)
        return _mistral_ocr_api(pdf_path, key)

    if provider == "google":
        key = key or GOOGLE_API_KEY.strip()
        if not key:
            print("  No Google key — falling back to PyMuPDF")
            return _pymupdf_extract(pdf_path)
        return _google_ocr(pdf_path, key)

    return _pymupdf_extract(pdf_path)


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(pdf_name: str) -> str:
    return os.path.join(OCR_CACHE_DIR, f"{pdf_name}_ocr.json")

def cache_exists(pdf_name: str) -> bool:
    return os.path.exists(_cache_path(pdf_name))

def save_cache(results: List[Dict], pdf_name: str) -> None:
    with open(_cache_path(pdf_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  💾 OCR cached → {_cache_path(pdf_name)}")

def load_cache(pdf_name: str) -> List[Dict]:
    with open(_cache_path(pdf_name), "r", encoding="utf-8") as f:
        return json.load(f)

# backward compat
def save_ocr_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_ocr_results(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)