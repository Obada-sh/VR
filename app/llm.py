"""LLM HTTP call plus the text-cleanup passes that wrap it."""

import re
from typing import List

import requests
from fastapi import HTTPException

from .config import API_KEY, API_URL, MODEL_NAME
from .prompts import TASHKEEL_PROMPT


# Extra token budget so the model's (unavoidable) reasoning doesn't crowd out
# the actual answer. See the payload comment in call_llm.
REASONING_HEADROOM = 1024


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
        # mimo-v2.5 is a reasoning model and IGNORES the "thinking: disabled"
        # flag below — it still emits reasoning tokens, and they're billed
        # against max_tokens. With no headroom the reasoning eats the whole
        # budget, the answer never gets written, and `content` comes back null.
        # `max_tokens` from callers means "room for the ANSWER", so add to it.
        "max_tokens": max_tokens + REASONING_HEADROOM,
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
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail=f"LLM returned no choices: {str(data)[:500]}")

    message = choice.get("message") or {}
    content = message.get("content")
    if content:
        return strip_think(content.strip())

    # `content` is null. Say WHY instead of blowing up on .strip() — a reasoning
    # model that spent its whole budget thinking looks identical to a refusal
    # unless you check finish_reason.
    finish = choice.get("finish_reason")
    if message.get("refusal"):
        raise HTTPException(status_code=502, detail=f"LLM refused the request: {message['refusal']}")
    if finish == "length":
        raise HTTPException(
            status_code=502,
            detail=(
                f"LLM ({MODEL_NAME}) used its entire token budget on reasoning without "
                f"writing an answer. Raise max_tokens / REASONING_HEADROOM in app/llm.py."
            ),
        )
    raise HTTPException(
        status_code=502,
        detail=f"LLM ({MODEL_NAME}) returned empty content (finish_reason={finish!r}).",
    )


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
