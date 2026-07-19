"""
Speech-to-text microservice (runs in its OWN process AND its own venv).

Model: CohereLabs/cohere-transcribe-arabic-07-2026 — a 2B Conformer encoder / Transformer
decoder ASR model (transformers-based). It supports Arabic, which is what we force
here for the Damascus-dialect doctor audio.

Why a separate process AND a separate venv
------------------------------------------
This model needs `transformers>=5.4.0`, but Leva-TTS (in the main venv) pins
`transformers<5`. The two can't coexist in one environment. Speech-to-text
therefore lives here, in its OWN venv (`sttenv`), in its OWN process, and the
main API (main.py) calls it over HTTP. The main process holds ONLY PyTorch/
Leva-TTS (transformers<5); this process holds ONLY the Cohere ASR model
(transformers>=5.4). They never collide.

Setup (once), in this service's venv:
    python -m venv sttenv
    source sttenv/Scripts/activate
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    pip install "transformers>=5.4.0" accelerate sentencepiece protobuf soundfile librosa
    pip install fastapi "uvicorn[standard]" python-multipart huggingface_hub hf_transfer
    hf auth login                       # model is gated
    # accept the license at https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026
    HF_HUB_DISABLE_XET=1 hf download CohereLabs/cohere-transcribe-arabic-07-2026

Run this FIRST, in its own terminal (with sttenv active):
    python -m uvicorn whisper_service:app --port 8001

Then start the main API in another terminal (with myenv active):
    python -m uvicorn main:app --port 8000
"""

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# huggingface_hub 1.x routes some repos through the hf_xet native client, whose
# transfer hangs at 0 bytes forever on this machine. Force the classic HTTPS
# download path. MUST be set before huggingface_hub is imported (transformers
# imports it lazily).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile

# Windows consoles default to cp1252, which can't print Arabic text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
LANGUAGE = "ar"            # force Arabic for the Damascus-dialect audio
SAMPLE_RATE = 16000        # the model expects 16 kHz audio


def _pick_dtype():
    """Return (device_map, torch_dtype) preferring the GPU with float16."""
    import torch

    if torch.cuda.is_available():
        print(f"✅  Using GPU: {torch.cuda.get_device_name(0)}")
        return "auto", torch.float16

    print("⚠️  CUDA not available — running Cohere ASR on CPU (much slower).")
    return "cpu", torch.float32


def _load_model():
    """Load the Cohere ASR model + processor once, at service startup."""
    import torch  # noqa: F401  (ensures torch is imported before transformers)
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    device_map, torch_dtype = _pick_dtype()
    print(f"Loading {MODEL_ID} (dtype={torch_dtype})...")
    start = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    model.eval()
    print(f"Cohere ASR model ready in {time.time() - start:.1f}s")
    return model, processor


MODEL, PROCESSOR = _load_model()
# Generation is not guaranteed thread-safe -> serialize transcriptions.
_infer_lock = threading.Lock()

app = FastAPI(title="Cohere ASR STT service", version="2.0.0")


def _load_audio(path: str):
    """Load an audio file as a mono 16 kHz float array."""
    from transformers.audio_utils import load_audio

    return load_audio(path, sampling_rate=SAMPLE_RATE)


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file. Returns {raw_text, language, duration}.

    This is the RAW model output — the dialect-fix LLM pass happens back in
    main.py, which has the LLM credentials.
    """
    import torch

    # The audio loader decodes from a real file path, so persist the upload.
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        start = time.time()
        audio = _load_audio(tmp_path)
        duration = len(audio) / SAMPLE_RATE

        with _infer_lock:
            inputs = PROCESSOR(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                language=LANGUAGE,
            )
            inputs = inputs.to(MODEL.device, dtype=MODEL.dtype)
            with torch.no_grad():
                outputs = MODEL.generate(**inputs, max_new_tokens=256)
            raw_text = PROCESSOR.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        print(
            f"Transcribed {duration:.1f}s of audio in {time.time() - start:.1f}s: {raw_text}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)

    return {
        "raw_text": raw_text,
        "language": LANGUAGE,
        "duration": duration,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
