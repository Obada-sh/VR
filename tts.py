"""
Local Text-to-Speech using VoxCPM2 (openbmb/VoxCPM2).

Everything runs LOCALLY on the GPU — no external API. The VoxCPM2 model is
loaded ONCE at import time (so it sits on the GPU as soon as the server starts,
exactly like the whisper model) and reused for every request. VoxCPM2 is
multilingual (30+ languages, Arabic included) and needs no language tag.

Unlike the previous Leva-TTS engine, VoxCPM2 has NO built-in named speakers.
It produces a voice one of two ways:

  1. Voice cloning (DEFAULT) — it mimics a short reference clip + that clip's
     transcript. By default it clones the bundled 1.wav (a male Damascene
     sample) so the patient keeps a consistent male voice out of the box.
     Override with VOXCPM_PROMPT_WAV / VOXCPM_PROMPT_TEXT to clone a different
     voice.

  2. Seed-based voice — set VOXCPM_PROMPT_WAV to an empty/missing path and the
     voice is generated from a FIXED seed (VOXCPM_SEED) instead. NOTE: voxcpm
     2.0.3 (PyPI) has no `seed` parameter — it only exists on git main — so on
     that build this fallback yields a DIFFERENT voice per request. Cloning is
     the reliable option; the seed is passed only if the install supports it.

Env overrides:
    VOXCPM_MODEL_DIR    dir holding the weights (default: ./models/VoxCPM2)
    VOXCPM_PROMPT_WAV   path to a reference voice clip (default: bundled 1.wav)
    VOXCPM_PROMPT_TEXT  transcript of that clip (default: 1.wav's transcript)
    VOXCPM_SEED         integer seed, used only if this voxcpm build supports it
    VOXCPM_CFG          guidance value (default 2.0)
    VOXCPM_TIMESTEPS    inference timesteps (default 10)

Model: https://huggingface.co/openbmb/VoxCPM2
"""

import inspect
import io
import os
import threading

# VoxCPM2's checkpoint is large. HuggingFace's default single-stream downloader
# tends to STALL on big files. hf_transfer is a Rust, multi-connection downloader
# that fixes this. Enabling it without the package installed raises an error, so
# only turn it on when it's importable. Must be set before huggingface_hub is
# imported (i.e. before `voxcpm`).
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

from voxcpm import VoxCPM

# Where the model weights live. HuggingFace's own downloader (and hf_transfer)
# STALL on this network, so the weights are fetched manually with aria2c from
# the hf-mirror.com mirror into a project-local dir — see README/justfile.
# Kept OFF %LOCALAPPDATA% on purpose: this venv's Microsoft Store Python
# virtualizes AppData reads and can shadow files written by other tools.
# If the dir isn't populated we fall back to a normal HuggingFace download.

# --- Config ------------------------------------------------------------------
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.environ.get("VOXCPM_MODEL_DIR", os.path.join(_HERE, "models", "VoxCPM2"))

# Reference clip for voice cloning (see module docstring). Defaults to the
# bundled 1.wav (a male Damascene sample) + its transcript, so the patient
# clones that voice out of the box. Override either with the env vars.
PROMPT_WAV = os.environ.get("VOXCPM_PROMPT_WAV", os.path.join(_HERE, "1.wav"))
PROMPT_TEXT = os.environ.get(
    "VOXCPM_PROMPT_TEXT",
    "شلونك دكتور، شو أخبارك؟ حلقي والله عم يوجعني، لوزاتي ملتهبين. "
    "صرلي 3 أيام بالتخت، مالي حسنان، فز من التخت من السخونة. "
    "أخدت دوا الالتهاب يلي وصفتلي ياه المرة الماضية بس ما طلع شغال، "
    "فبتوقع بدي دوا التهاب تاني.",
)

# Fixed seed => the generated voice stays identical across every request, so the
# patient doesn't change voice mid-conversation.
SEED = int(os.environ.get("VOXCPM_SEED", "20240115"))
CFG_VALUE = float(os.environ.get("VOXCPM_CFG", "2.0"))
INFERENCE_TIMESTEPS = int(os.environ.get("VOXCPM_TIMESTEPS", "10"))


def _resolve_prompt():
    """Return (prompt_wav_path, prompt_text) if a usable reference clip exists."""
    if PROMPT_WAV and os.path.exists(PROMPT_WAV):
        return PROMPT_WAV, PROMPT_TEXT
    if PROMPT_WAV:
        print(f"⚠️  VOXCPM_PROMPT_WAV={PROMPT_WAV!r} not found; using seed voice.")
    return None, None


def _sample_rate(model) -> int:
    """VoxCPM2 outputs 48 kHz. Read it off the model rather than hard-coding."""
    for path in (("sample_rate",), ("tts_model", "sample_rate"), ("audio_vae", "sample_rate")):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, int):
            return obj
    print("⚠️  Could not read VoxCPM sample rate; assuming 16000 Hz.")
    return 16000


def _load():
    """Load the VoxCPM2 model onto the GPU (once)."""
    print("Loading VoxCPM2 model...")
    # Denoiser off either way: we feed clean text + a clean reference clip.
    if os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        print(f"   using local weights: {MODEL_DIR}")
        model = VoxCPM(voxcpm_model_path=MODEL_DIR, enable_denoiser=False, device=_DEVICE)
    else:
        print(f"   {MODEL_DIR} not populated -> downloading from HuggingFace (may stall; see justfile `model-tts`)")
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    prompt_wav, _ = _resolve_prompt()
    voice = f"cloned from {prompt_wav}" if prompt_wav else f"seed={SEED}"
    print(f"✅  VoxCPM2 ready on {_DEVICE}. Voice: {voice}")
    return model


_MODEL = _load()
_SAMPLE_RATE = _sample_rate(_MODEL)
# Inference uses torch + shared model state -> serialize calls.
_tts_lock = threading.Lock()

# generate() forwards **kwargs to _generate(), whose signature moves between
# releases (e.g. `seed` exists on git main but NOT in voxcpm 2.0.3 on PyPI, where
# passing it raises TypeError). Only send what this install actually accepts.
_GENERATE_PARAMS = set(inspect.signature(_MODEL._generate).parameters)

if "seed" not in _GENERATE_PARAMS and not _resolve_prompt()[0]:
    print("⚠️  This voxcpm build ignores `seed` and no reference clip resolved — "
          "the patient's voice will CHANGE between requests. Set VOXCPM_PROMPT_WAV.")


def synthesize_to_wav_bytes(text: str) -> bytes:
    """Turn Arabic/Levantine text into a WAV, returned as raw bytes."""
    prompt_wav, prompt_text = _resolve_prompt()
    kwargs = {
        "text": text,
        "cfg_value": CFG_VALUE,
        "inference_timesteps": INFERENCE_TIMESTEPS,
        "normalize": False,  # VoxCPM's text normalizer targets EN/ZH; skip for Arabic.
        "seed": SEED,  # dropped below if unsupported; the reference clip fixes the voice anyway
    }
    if prompt_wav:
        kwargs["prompt_wav_path"] = prompt_wav
        if prompt_text:
            kwargs["prompt_text"] = prompt_text

    kwargs = {k: v for k, v in kwargs.items() if k in _GENERATE_PARAMS}

    with _tts_lock:
        wave = _MODEL.generate(**kwargs)

    buf = io.BytesIO()
    sf.write(buf, wave, _SAMPLE_RATE, format="WAV")
    return buf.getvalue()
