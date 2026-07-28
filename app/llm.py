"""LLM HTTP calls (blocking + streaming) plus the text-cleanup passes.

Two ways to call the model:
  call_llm()   — blocking, returns the whole reply. Used by the REST endpoints.
  stream_llm() — yields text deltas as they arrive. Used by the voice WebSocket,
                 where the point is to start speaking before generation finishes.

iter_sentences() sits on top of stream_llm() and regroups the deltas into
TTS-sized speakable chunks, which is what makes the voice turn feel fast.
"""

import json
import re
from typing import Iterable, Iterator, List

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


def _headers() -> dict:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENCODE_API_KEY environment variable is not set.",
        )
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }


def _payload(messages: List[dict], max_tokens: int, temperature: float) -> dict:
    return {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }


def call_llm(messages: List[dict], max_tokens: int, temperature: float) -> str:
    headers = _headers()
    payload = _payload(messages, max_tokens, temperature)
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


# --- Streaming ---------------------------------------------------------------

def stream_llm(messages: List[dict], max_tokens: int, temperature: float) -> Iterator[str]:
    """Yield the reply as text deltas, as the model produces them.

    Falls back to one blocking call (yielding the whole reply as a single chunk)
    if the endpoint rejects or mishandles `stream: true`, so a server-side
    change can never take voice chat down — it only makes it slower.
    """
    headers = _headers()
    payload = _payload(messages, max_tokens, temperature) | {"stream": True}

    try:
        with requests.post(
            API_URL, headers=headers, json=payload, stream=True, timeout=(10, 120)
        ) as resp:
            resp.raise_for_status()
            # `requests` picks the decode encoding from Content-Type. SSE responses
            # are "text/event-stream" with no charset param, and requests' own rule
            # for a bare "text/*" type defaults that to ISO-8859-1 (not UTF-8) — so
            # every Arabic byte pair gets decoded one byte at a time as Latin-1,
            # producing exactly the "Ã¢Ã£Ã¢..." mojibake this line fixes.
            resp.encoding = "utf-8"
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    choices = json.loads(data).get("choices") or []
                except json.JSONDecodeError:
                    continue                      # keep-alive / comment line
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta
    except requests.RequestException:
        yield call_llm(messages, max_tokens, temperature)


def _strip_think_stream(deltas: Iterable[str]) -> Iterator[str]:
    """Drop <think>...</think> spans from a delta stream without buffering it all.

    Thinking is disabled in the payload, so this should never fire — but a leaked
    reasoning block would be spoken aloud by the patient, so guard anyway.
    """
    open_tag, close_tag = "<think>", "</think>"
    buf, in_think = "", False

    for delta in deltas:
        buf += delta
        out = ""
        while buf:
            if in_think:
                i = buf.find(close_tag)
                if i == -1:
                    # Discard, but keep a tail in case the tag is split across deltas.
                    buf = buf[-(len(close_tag) - 1):]
                    break
                buf, in_think = buf[i + len(close_tag):], False
            else:
                i = buf.find(open_tag)
                if i == -1:
                    hold = len(open_tag) - 1     # a partial opening tag may follow
                    if len(buf) > hold:
                        out, buf = out + buf[:-hold], buf[-hold:]
                    break
                out, buf, in_think = out + buf[:i], buf[i + len(open_tag):], True
        if out:
            yield out

    if buf and not in_think:
        yield buf


# Where it is always safe to break for speech, and where it is only worth it
# once the chunk is long enough to sound natural on its own.
_STRONG_BREAKS = ".!?؟\n"
_WEAK_BREAKS = "،؛,:"
_MIN_WEAK_CHARS = 60      # don't cut on a comma until the chunk has some body
_MAX_CHARS = 200          # force a break if the model never punctuates


def _find_break(buf: str) -> int | None:
    """Index just past the best place to cut `buf`, or None to keep accumulating."""
    for i, ch in enumerate(buf):
        if ch in _STRONG_BREAKS:
            # A '.' after a digit may be a decimal point — "حرارتي 37.5" must
            # stay in one piece or the TTS reads it as two fragments.
            if ch == "." and i > 0 and buf[i - 1].isdigit():
                if i == len(buf) - 1:
                    # The next character hasn't streamed in yet, so we cannot
                    # tell a decimal from a full stop. Wait rather than guess:
                    # deltas arrive token-sized, so "37." and "5" routinely land
                    # in separate chunks.
                    return None
                if buf[i + 1].isdigit():
                    continue
            return i + 1
        if ch in _WEAK_BREAKS and i + 1 >= _MIN_WEAK_CHARS:
            return i + 1

    if len(buf) >= _MAX_CHARS:
        space = buf.rfind(" ", 0, _MAX_CHARS)
        return space + 1 if space > 0 else _MAX_CHARS
    return None


# Stage directions the patient must never "say". Stripped per-character rather
# than per-sentence so live text can be displayed without a direction flashing
# on screen before a later pass removes it. '*' both opens and closes.
# Written as escapes, not literals: the fullwidth forms get silently flattened
# to ASCII by some editors/linters, which would drop （ ） support unnoticed.
_DIRECTION_OPEN = "(（*"
_DIRECTION_CLOSE = ")）*"


def _clean_stream(deltas: Iterable[str]) -> Iterator[str]:
    """Yield display-ready text: no <think> blocks, no stage directions.

    Safe to append verbatim to a UI as it arrives — nothing yielded here will
    need to be retracted later.
    """
    in_direction = False
    for delta in _strip_think_stream(deltas):
        out = []
        for ch in delta:
            if in_direction:
                if ch in _DIRECTION_CLOSE:
                    in_direction = False
            elif ch in _DIRECTION_OPEN:
                in_direction = True
            else:
                out.append(ch)
        if out:
            yield "".join(out)


def iter_reply_stream(deltas: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Split one LLM stream into the two things the voice turn needs at once.

    Yields ('delta', text) the instant text arrives — for live character-by-
    character display — and ('sentence', text) at each speech boundary, which is
    the unit the TTS can actually synthesize with correct prosody.

    Every character appears in exactly one 'delta'; the 'sentence' events repeat
    that same text regrouped, so a consumer should render one or the other, not
    both.
    """
    buf = ""
    for clean in _clean_stream(deltas):
        yield "delta", clean
        buf += clean
        while (cut := _find_break(buf)) is not None:
            piece, buf = buf[:cut], buf[cut:]
            if piece := re.sub(r"\s{2,}", " ", piece).strip():
                yield "sentence", piece

    if tail := re.sub(r"\s{2,}", " ", buf).strip():
        yield "sentence", tail


def iter_sentences(deltas: Iterable[str]) -> Iterator[str]:
    """Just the speakable chunks — the sentence half of iter_reply_stream()."""
    for kind, text in iter_reply_stream(deltas):
        if kind == "sentence":
            yield text


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
