"""Client for the whisper STT microservice + the dialect-fix LLM pass."""

import requests
from fastapi import HTTPException, UploadFile

from .config import WHISPER_SERVICE_URL
from .llm import call_llm
from .prompts import STT_FIX_PROMPT


def transcribe_and_fix(file: UploadFile) -> dict:
    """Send an uploaded audio file to the whisper STT service, then run the
    dialect-fix LLM pass on the returned transcript.

    Returns {text, raw_text, language, duration}: `text` is the corrected
    transcript, `raw_text` is whisper's exact output.
    """
    audio_bytes = file.file.read()
    filename = file.filename or "audio.wav"
    content_type = file.content_type or "application/octet-stream"

    try:
        resp = requests.post(
            f"{WHISPER_SERVICE_URL}/transcribe",
            files={"file": (filename, audio_bytes, content_type)},
            timeout=300,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Whisper STT service unreachable at {WHISPER_SERVICE_URL}. "
                f"Start it first:  python -m uvicorn whisper_service:app --port 8001  ({e})"
            ),
        )

    stt = resp.json()
    raw_text = stt["raw_text"]

    if not raw_text:
        raise HTTPException(status_code=400, detail="No speech detected in the audio.")

    corrected = call_llm(
        [{"role": "user", "content": STT_FIX_PROMPT.format(raw_text=raw_text)}],
        max_tokens=800,
        temperature=0.2,
    )

    return {
        "text": corrected,
        "raw_text": raw_text,
        "language": stt["language"],
        "duration": stt["duration"],
    }
