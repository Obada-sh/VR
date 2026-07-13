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
    """Turn Arabic/Levantine text into a WAV, returned as raw bytes."""
    with _tts_lock:
        wave, sample_rate = _MODEL.synthesize(
            text,
            speaker=SPEAKER,
            temperature=0.65,
            top_p=0.85,
        )
    buf = io.BytesIO()
    sf.write(buf, wave, sample_rate, format="WAV")
    return buf.getvalue()
