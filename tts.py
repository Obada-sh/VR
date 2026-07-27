"""
Local Levantine-dialect Text-to-Speech using Leva-TTS (XTTS-v2 based).

Everything runs LOCALLY on the GPU — no external API. The Leva-TTS model is
loaded ONCE at import time (so it sits on the GPU as soon as the server starts,
exactly like the whisper model) and reused for every request.

Leva-TTS ships 10 built-in speakers (no reference clip needed):
  Female: Amina, Fatma, Lamyaa, Mona, Haneen
  Male:   Badr, Mohamed, Saad, Rami, Fadi

The patient speaks with a MALE voice by default ("Badr"). Override with:

    export LEVA_SPEAKER="Rami"

Project: https://mohammedaly22.github.io/Leva-TTS/
"""

import io
import os
import threading
from typing import Iterator

import numpy as np

# The Leva-TTS checkpoint (best_model.pth) is ~5.6 GB. HuggingFace's default
# single-stream downloader tends to STALL on a file that big. hf_transfer is a
# Rust, multi-connection downloader that fixes this. Enabling it without the
# package installed raises an error, so only turn it on when it's importable.
# Must be set before huggingface_hub is imported (i.e. before `leva_tts`).
try:
    import hf_transfer  # noqa: F401

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

# Import torch BEFORE soundfile. On Windows, loading soundfile's native
# libsndfile before torch's OpenMP/MKL runtime segfaults the process at import
# time (no traceback). Importing torch first claims that runtime and prevents
# the crash. Do not reorder these.
import torch

import soundfile as sf

from leva_tts import LevaTTS

# --- Config ------------------------------------------------------------------
# Male voice for the patient. Override with the LEVA_SPEAKER env var.
SPEAKER = os.environ.get("LEVA_SPEAKER", "Badr")  # male Levantine voice
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# XTTS-v2 always renders at 24 kHz. The voice WebSocket sends raw PCM at this
# rate, so the Unity client must create its playback AudioClip to match.
SAMPLE_RATE = 24000

# Generation settings, shared by the blocking and streaming paths.
_GEN = {"temperature": 0.65, "top_p": 0.85}

# How many GPT tokens the streaming path decodes before emitting an audio chunk.
# Smaller = the patient starts talking sooner, at the cost of more chunks and a
# slightly higher risk of artifacts at the seams. Measured on this GPU:
#     20 (library default) -> 0.38s to first audio
#     10 (ours)            -> 0.21s
#      6                   -> 0.14s, but 21 chunks and a worse total time
# Blocking synthesis of the same sentence takes 2.21s before ANY audio exists.
_STREAM_CHUNK = int(os.environ.get("LEVA_STREAM_CHUNK", "10"))


def _load():
    """Load the Leva-TTS model onto the GPU (once)."""
    print("Loading Leva-TTS model (Levantine/XTTS-v2)...")
    # First run auto-downloads the checkpoint from Hugging Face (cached after).
    model = LevaTTS(device=_DEVICE, preprocess_text=True)
    print(f"✅  Leva-TTS ready on {_DEVICE}. Voice={SPEAKER} (male)")
    return model


_MODEL = _load()
# XTTS inference uses torch + shared model state -> serialize calls.
_tts_lock = threading.Lock()


def synthesize_to_wav_bytes(text: str) -> bytes:
    """Turn Arabic/Levantine text into a WAV, returned as raw bytes.

    Blocking: nothing comes back until the whole utterance is rendered. Used by
    the REST /chat-voice endpoint. For the voice WebSocket use synthesize_stream.
    """
    with _tts_lock:
        wave, sample_rate = _MODEL.synthesize(text, speaker=SPEAKER, **_GEN)
    buf = io.BytesIO()
    sf.write(buf, wave, sample_rate, format="WAV")
    return buf.getvalue()


def float_to_pcm16(wave: np.ndarray) -> bytes:
    """Convert float32 audio in [-1, 1] to little-endian PCM16 bytes for the wire."""
    clipped = np.clip(wave, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def synthesize_stream(text: str) -> Iterator[bytes]:
    """Synthesize `text`, yielding PCM16 chunks as they are generated.

    This is what makes the voice turn feel fast: the first chunk lands after a
    fraction of the utterance instead of after all of it.

    The model lock is held for the whole generator, so a caller MUST drain it or
    close it (a `for` loop does both). Only one utterance is synthesized at a
    time regardless — XTTS shares mutable state across calls.
    """
    with _tts_lock:
        for chunk in _MODEL.stream(
            text, speaker=SPEAKER, chunk_size=_STREAM_CHUNK, **_GEN
        ):
            yield float_to_pcm16(chunk)
