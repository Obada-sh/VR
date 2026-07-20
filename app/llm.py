"""LLM HTTP call plus the text-cleanup passes that wrap it."""

import re
from typing import List

import requests
from fastapi import HTTPException

import logging

from .config import CHAIN, MISSING_KEYS, Provider

log = logging.getLogger(__name__)
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


def _post_once(
    provider: Provider, messages: List[dict], max_tokens: int, temperature: float
) -> str:
    """One attempt against one provider. Raises on any failure."""
    payload = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider.supports_thinking_flag:
        payload["thinking"] = {"type": "disabled"}

    resp = requests.post(
        provider.url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.key}",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return strip_think(resp.json()["choices"][0]["message"]["content"].strip())


def call_llm(messages: List[dict], max_tokens: int, temperature: float) -> str:
    """Try each configured provider in turn; return the first usable reply.

    Any failure moves on to the next provider — free tiers go down (5xx), hit
    their daily cap (429), and retire model IDs (400/404), and none of those are
    worth failing a consultation over. Only when every provider is exhausted do
    we raise, reporting what each one said.
    """
    if not CHAIN:
        raise HTTPException(
            status_code=500,
            detail=(
                "No LLM provider is configured. Set one of these keys in .env: "
                f"{', '.join(MISSING_KEYS)} — see .env.example."
            ),
        )

    failures = []
    for provider in CHAIN:
        try:
            return _post_once(provider, messages, max_tokens, temperature)
        except (requests.RequestException, KeyError, ValueError) as e:
            # KeyError/ValueError: a 200 whose body isn't the shape we expect.
            reason = str(e)
            resp_text = getattr(getattr(e, "response", None), "text", None)
            if resp_text:
                reason += f" | server said: {resp_text[:200]}"
            log.warning("LLM provider %s failed, trying next: %s", provider, reason)
            failures.append(f"{provider}: {reason}")

    raise HTTPException(
        status_code=502,
        detail="All LLM providers failed. " + " || ".join(failures),
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
