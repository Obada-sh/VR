"""
Whisper STT microservice (runs in its OWN process).

Why this exists
---------------
faster-whisper (CTranslate2) and Habibi-TTS (PyTorch) each bundle their own copy
of the Intel OpenMP runtime. Loading BOTH into a single process segfaults on
Windows. So speech-to-text lives here, in a separate process, and the main API
(main.py) calls it over HTTP. This process holds ONLY whisper; the main process
holds ONLY PyTorch/Habibi-TTS. They never collide.

Run this FIRST, in its own terminal:
    python -m uvicorn whisper_service:app --port 8001

Then start the main API in another terminal:
    python -m uvicorn main:app --port 8000
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# huggingface_hub 1.x routes Xet-backed repos (this whisper model is one) through
# the hf_xet native client by default. On this machine that transfer hangs at
# 0 bytes forever. Force the classic HTTPS download path, which works. This MUST
# be set before huggingface_hub is imported (faster_whisper imports it lazily).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile

# Windows consoles default to cp1252, which can't print Arabic text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

WHISPER_MODEL_SIZE = "large-v3"


def _resolve_model_path():
    """Return a local model dir if the weights are already on disk, else the
    model name.

    huggingface_hub 1.x downloads this repo's model.bin over the Xet transfer,
    which hangs at 0 bytes on this machine. Once model.bin has been fetched
    manually (curl) into the HF cache snapshot folder, load straight from that
    directory so startup never touches the hub download path again.
    """
    snapshots = (
        Path.home()
        / ".cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots"
    )
    if snapshots.is_dir():
        for snap in snapshots.iterdir():
            if (snap / "model.bin").exists():
                return str(snap)
    return WHISPER_MODEL_SIZE


def _pick_device():
    """Return (device, compute_type) preferring the GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅  Using GPU: {torch.cuda.get_device_name(0)}")
            return "cuda", "float16"
    except ImportError:
        pass

    print("⚠️  CUDA not available — running whisper on CPU with int8 (slower).")
    return "cpu", "int8"


def _load_whisper():
    """Load faster-whisper once, at service startup."""
    from faster_whisper import WhisperModel

    device, compute_type = _pick_device()
    model_path = _resolve_model_path()
    where = "local dir" if model_path != WHISPER_MODEL_SIZE else "hub"
    print(f"Loading faster-whisper {WHISPER_MODEL_SIZE} ({device}, {compute_type}) from {where}...")
    start = time.time()
    model = WhisperModel(model_path, device=device, compute_type=compute_type)
    print(f"Whisper model ready in {time.time() - start:.1f}s")
    return model


WHISPER_MODEL = _load_whisper()
# Whisper inference is not guaranteed thread-safe -> serialize transcriptions.
_whisper_lock = threading.Lock()

app = FastAPI(title="Whisper STT service", version="1.0.0")


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file. Returns {raw_text, language, duration}.

    This is the RAW whisper output — the dialect-fix LLM pass happens back in
    main.py, which has the LLM credentials.
    """
    # Whisper decodes via ffmpeg/av from a real file path, so persist the upload.
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        start = time.time()
        with _whisper_lock:
            segments, info = WHISPER_MODEL.transcribe(
                tmp_path,
                language="ar",            # force Arabic
                task="transcribe",
                beam_size=5,
                temperature=0.0,          # greedy for best accuracy
                vad_filter=True,          # skip silence -> faster + cleaner output
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            raw_text = "".join(s.text for s in segments).strip()
        print(
            f"Transcribed {info.duration:.1f}s of audio in {time.time() - start:.1f}s: {raw_text}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)

    return {
        "raw_text": raw_text,
        "language": info.language,
        "duration": info.duration,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": WHISPER_MODEL_SIZE}
