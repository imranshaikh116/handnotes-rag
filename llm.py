"""
llm.py — Local answer generation via Ollama.
Strict grounding: the model is NEVER allowed to answer outside the retrieved context.
"""

import json
import requests
from typing import List, Dict, Optional, Tuple, Iterator

from config import (
    OLLAMA_BASE_URL,
    DEFAULT_LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    MIN_SIMILARITY_SCORE,
    HIGH_CONFIDENCE_SCORE,
    MED_CONFIDENCE_SCORE,
    HISTORY_SENT_TO_LLM,
)

NO_INFO_REPLY = "I don't have enough information in the notes to answer this."

_DONT_KNOW_PHRASES = [
    "don't have enough",
    "not in the notes",
    "cannot find",
    "no information",
    "not mentioned",
    "not covered",
    "outside the scope",
]


# ─── Ollama helpers ───────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> List[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def model_exists(model: str) -> bool:
    available = list_models()
    return any(model in m for m in available)


# ─── Prompt builder ───────────────────────────────────────────────────────────

def _build_prompt(
    query: str,
    context_chunks: List[Dict],
    history: List[Dict],
) -> str:
    ctx = ""
    for i, c in enumerate(context_chunks):
        ctx += f"\n[SOURCE {i+1} | PDF: {c['pdf']} | Page {c['page']}]\n{c['text']}\n"

    hist = ""
    if history:
        for turn in history[-HISTORY_SENT_TO_LLM * 2:]:
            role = "User" if turn["role"] == "user" else "Assistant"
            hist += f"{role}: {turn['content']}\n"

    return f"""You are a precise study assistant. Answer questions ONLY from the provided notes.

STRICT RULES — follow every one of them:
1. Use ONLY the SOURCE sections below. Do not add any outside knowledge.
2. If the answer is not in the sources, reply exactly: "{NO_INFO_REPLY}"
3. Always cite sources as: (PDF: <pdf_name>, Page <number>)
4. Keep answers concise and accurate — do not pad or guess.
5. Never contradict or ignore what is written in the sources.

{f"RECENT CONVERSATION:{chr(10)}{hist}" if hist else ""}
NOTES CONTEXT:
{ctx}
QUESTION: {query}

ANSWER (cite sources inline):"""


# ─── Confidence ───────────────────────────────────────────────────────────────

def _confidence(top_score: float, answer: str, chunks: List[Dict]) -> float:
    if any(p in answer.lower() for p in _DONT_KNOW_PHRASES):
        return round(min(top_score * 0.4, 0.25), 2)
    avg = sum(c["score"] for c in chunks[:3]) / min(3, len(chunks))
    return round(min(top_score * 0.6 + avg * 0.4, 0.99), 2)


def confidence_label(score: float) -> Tuple[str, str]:
    """Return (human label, hex colour)."""
    if score >= HIGH_CONFIDENCE_SCORE:
        return "High Confidence", "#22c55e"
    if score >= MED_CONFIDENCE_SCORE:
        return "Medium Confidence", "#f59e0b"
    return "Low / No Info", "#ef4444"


# ─── Main answer generation ───────────────────────────────────────────────────

def generate_answer(
    query: str,
    context_chunks: List[Dict],
    model: str = DEFAULT_LLM_MODEL,
    history: Optional[List[Dict]] = None,
) -> Dict:
    """
    Generate a grounded answer using the local Ollama LLM.

    Returns:
        {answer, model, sources, confidence, confidence_label, confidence_color, is_grounded}
    """
    history = history or []

    # Fast-path: empty context or score too low
    if not context_chunks:
        return _no_info_result(model, [])

    top_score = max(c["score"] for c in context_chunks)
    if top_score < MIN_SIMILARITY_SCORE:
        return _no_info_result(model, context_chunks[:3], top_score)

    if not is_ollama_running():
        return {
            "answer": "❌ Ollama is not running. Start it with: `ollama serve`",
            "model": model, "sources": [], "confidence": 0.0,
            "confidence_label": "Error", "confidence_color": "#ef4444",
            "is_grounded": False,
        }

    prompt = _build_prompt(query, context_chunks, history)

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":   model,
                "prompt":  prompt,
                "stream":  False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "top_p":       LLM_TOP_P,
                    "num_predict": LLM_MAX_TOKENS,
                },
            },
            timeout=120,
        )

        if resp.status_code == 200:
            answer = resp.json()["response"].strip()
            conf   = _confidence(top_score, answer, context_chunks)
            label, color = confidence_label(conf)
            return {
                "answer":            answer,
                "model":             model,
                "sources":           context_chunks[:5],
                "confidence":        conf,
                "confidence_label":  label,
                "confidence_color":  color,
                "is_grounded":       True,
            }
        else:
            return _error_result(model, f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")

    except requests.exceptions.ConnectionError:
        return _error_result(model, "Cannot connect to Ollama. Run: `ollama serve`")
    except Exception as exc:
        return _error_result(model, str(exc))


def stream_answer(
    query: str,
    context_chunks: List[Dict],
    model: str = DEFAULT_LLM_MODEL,
    history: Optional[List[Dict]] = None,
) -> Iterator[str]:
    """Yield answer tokens one by one (for streaming UI)."""
    history = history or []
    prompt = _build_prompt(query, context_chunks, history)
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True,
                  "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS}},
            stream=True, timeout=120,
        )
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done"):
                    yield data.get("response", "")
    except Exception as exc:
        yield f"\n[Stream error: {exc}]"


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _no_info_result(model: str, sources: List[Dict], conf: float = 0.0) -> Dict:
    label, color = confidence_label(conf)
    return {
        "answer":           NO_INFO_REPLY,
        "model":            model,
        "sources":          sources,
        "confidence":       conf,
        "confidence_label": label,
        "confidence_color": color,
        "is_grounded":      False,
    }


def _error_result(model: str, msg: str) -> Dict:
    return {
        "answer":           f"❌ {msg}",
        "model":            model,
        "sources":          [],
        "confidence":       0.0,
        "confidence_label": "Error",
        "confidence_color": "#ef4444",
        "is_grounded":      False,
    }
