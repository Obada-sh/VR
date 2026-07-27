"""Client for the STT microservice, with and without the dialect-fix LLM pass.

Two entry points:
  transcribe_and_fix() — upload a file, then run the dialect-fix LLM pass on the
                         transcript. Used by the REST endpoints.
  transcribe_pcm()     — send raw audio, return the transcript as-is. Used by the
                         voice WebSocket, which skips the fix pass entirely: the
                         patient prompt now tells the model to read past STT
                         errors itself (see rule 7 in BASE_SYSTEM_PROMPT), which
                         saves a whole LLM round trip on the latency-critical path.
"""

import io

import numpy as np
import requests
import soundfile as sf
from fastapi import HTTPException, UploadFile

from .config import WHISPER_SERVICE_URL
from .llm import call_llm
from .prompts import STT_FIX_PROMPT


def _post_audio(filename: str, audio_bytes: bytes, content_type: str) -> dict:
    """POST audio to the STT service and return its JSON response."""
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
                f"STT service unreachable at {WHISPER_SERVICE_URL}. "
                f"Start it first:  python -m uvicorn whisper_service:app --port 8001  ({e})"
            ),
        )
    return resp.json()


def transcribe_pcm(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe a float32 mono array. Returns the raw transcript ('' if silent).

    No dialect-fix LLM pass — the patient prompt handles STT errors itself.
    Wraps the samples in an in-memory WAV because the STT service decodes from a
    real audio file.
    """
    if audio.size == 0:
        return ""

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    stt = _post_audio("utterance.wav", buf.getvalue(), "audio/wav")
    return (stt.get("raw_text") or "").strip()


def transcribe_and_fix(file: UploadFile) -> dict:
    """Send an uploaded audio file to the whisper STT service, then run the
    dialect-fix LLM pass on the returned transcript.

    Returns {text, raw_text, language, duration}: `text` is the corrected
    transcript, `raw_text` is whisper's exact output.
    """
    stt = _post_audio(
        file.filename or "audio.wav",
        file.file.read(),
        file.content_type or "application/octet-stream",
    )
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
