"""
test_ocr.py — Run this to test your OCR API keys directly.
Usage:  python test_ocr.py
"""

import base64
import sys
import fitz  # PyMuPDF

PDF_PATH = "notes.pdf"   # change if your PDF has a different name

def render_first_page(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(2.0, 2.0)  # 144 DPI — smaller = faster test
    pix = page.get_pixmap(matrix=mat)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    doc.close()
    print(f"  Page rendered: {len(b64)//1024} KB")
    return b64


def test_mistral(api_key, b64):
    import requests
    print("\n=== Testing Mistral ===")
    print(f"  Key starts with: {api_key[:8]}...")
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "pixtral-12b-2409",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "Extract all text from this image. Output only the text."},
            ]}],
            "max_tokens": 500,
        },
        timeout=60,
    )
    print(f"  HTTP Status: {r.status_code}")
    if r.status_code == 200:
        text = r.json()["choices"][0]["message"]["content"]
        print(f"  ✅ SUCCESS! Extracted {len(text)} chars")
        print(f"  Preview: {text[:200]}")
        return True
    elif r.status_code == 429:
        print("  ❌ RATE LIMITED (429) — Mistral free tier exhausted for today")
        print("     Try again tomorrow or use Google Vision instead")
    elif r.status_code == 401:
        print("  ❌ UNAUTHORIZED (401) — API key is wrong")
        print("     Get a new key at: https://console.mistral.ai")
    else:
        print(f"  ❌ ERROR: {r.text[:300]}")
    return False


def test_google(api_key, b64):
    import requests
    print("\n=== Testing Google Vision ===")
    print(f"  Key starts with: {api_key[:8]}...")
    r = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
        json={"requests": [
            {"image": {"content": b64},
             "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}
        ]},
        timeout=30,
    )
    print(f"  HTTP Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if "error" in data.get("responses", [{}])[0]:
            err = data["responses"][0]["error"]
            print(f"  ❌ API Error: {err}")
            return False
        text = data["responses"][0].get("fullTextAnnotation", {}).get("text", "")
        print(f"  ✅ SUCCESS! Extracted {len(text)} chars")
        print(f"  Preview: {text[:200]}")
        return True
    elif r.status_code == 400:
        print("  ❌ BAD REQUEST (400) — likely causes:")
        print("     1. Vision API not enabled in Google Cloud Console")
        print("     2. API key has restrictions that block Vision API")
        print("     Fix: go to console.cloud.google.com → APIs → Enable 'Cloud Vision API'")
        print(f"  Response: {r.text[:400]}")
    elif r.status_code == 403:
        print("  ❌ FORBIDDEN (403) — billing not enabled or API key restricted")
        print("     Go to console.cloud.google.com and enable billing (free $300 credit)")
    else:
        print(f"  ❌ ERROR {r.status_code}: {r.text[:300]}")
    return False


def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 50)
    print("  HandNotes RAG — OCR API Tester")
    print("=" * 50)

    # Check PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"\n❌ PDF not found: {PDF_PATH}")
        print("   Make sure notes.pdf is in the same folder as this script")
        sys.exit(1)

    print(f"\n📄 Rendering first page of {PDF_PATH} ...")
    b64 = render_first_page(PDF_PATH)

    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    google_key  = os.getenv("GOOGLE_VISION_API_KEY", "").strip()

    print(f"\n🔑 Keys found in .env:")
    print(f"   Mistral: {'✅ ' + mistral_key[:8] + '...' if mistral_key else '❌ NOT SET'}")
    print(f"   Google:  {'✅ ' + google_key[:8] + '...' if google_key else '❌ NOT SET'}")

    results = {}

    if mistral_key:
        results["mistral"] = test_mistral(mistral_key, b64)
    else:
        print("\n⚠️  Skipping Mistral — no key in .env")

    if google_key:
        results["google"] = test_google(google_key, b64)
    else:
        print("\n⚠️  Skipping Google — no GOOGLE_VISION_API_KEY in .env")

    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    for provider, ok in results.items():
        print(f"  {provider:10}: {'✅ WORKING' if ok else '❌ FAILED'}")

    working = [k for k, v in results.items() if v]
    if working:
        print(f"\n✅ Use this provider in the app: {working[0]}")
        print(f"   Set OCR Provider to '{working[0]}' in the sidebar")
    else:
        print("\n❌ No OCR provider is working!")
        print("   Options:")
        print("   1. Wait until tomorrow (Mistral rate limit resets at midnight)")
        print("   2. Fix Google Vision API setup (enable it in Google Cloud Console)")
        print("   3. Create a NEW Mistral account with a fresh email for a new API key")
    print()


if __name__ == "__main__":
    main()
