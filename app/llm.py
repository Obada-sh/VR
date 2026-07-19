"""LLM HTTP call plus the text-cleanup passes that wrap it."""

import re
from typing import List

import requests
from fastapi import HTTPException

from .config import API_KEY, API_URL, MODEL_NAME
from .prompts import TASHKEEL_PROMPT


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks the model may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_stage_directions(text: str) -> str:
    """Drop narrated actions/emotions so only the patient's spoken words remain.

    Removes anything wrapped in parentheses ( ) / （ ） or asterisks *...*,
    e.g. "(تتنهد بتعب) أهلين دكتور" -> "أهلين دكتور".
    """
    text = re.sub(r"[\(（][^)）]*[\)）]", "", text)  # ( ... ) and （ ... ）
    text = re.sub(r"\*[^*]*\*", "", text)            # *...*
    text = re.sub(r"[ \t]{2,}", " ", text)           # collapse leftover gaps
    return text.strip()


def call_llm(messages: List[dict], max_tokens: int, temperature: float) -> str:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENCODE_API_KEY environment variable is not set.",
        )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        detail = f"LLM request failed: {e}"
        resp_text = getattr(getattr(e, "response", None), "text", None)
        if resp_text:
            detail += f" | server said: {resp_text}"
        raise HTTPException(status_code=502, detail=detail)

    data = resp.json()
    return strip_think(data["choices"][0]["message"]["content"].strip())


def add_tashkeel(text: str) -> str:
    """Add full Arabic diacritics (تشكيل) to the reply for accurate TTS.

    Runs a dedicated LLM pass. If it fails or returns nothing usable, fall back
    to the original (undiacritized) text so speech is never blocked.
    """
    try:
        diacritized = call_llm(
            [{"role": "user", "content": TASHKEEL_PROMPT.format(text=text)}],
            max_tokens=800,
            temperature=0.0,
        ).strip()
    except HTTPException:
        return text
    return diacritized or text
