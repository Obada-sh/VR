"""
Local Damascus-dialect Text-to-Speech using Habibi-TTS (F5-TTS based).

Everything runs LOCALLY on the GPU — no external API. The Unified Habibi
checkpoint is loaded ONCE at import time (so it sits on the GPU as soon as the
server starts, exactly like the whisper model) and reused for every request.

Habibi is a *voice-cloning* model, so the output voice has two ingredients:
  1. dialect_id = "LEV" (Levantine, code ⑩) — drives Damascene/Levantine
     phonology + prosody. The Unified model is required for LEV.
  2. A short REFERENCE CLIP (REF_AUDIO + its transcript REF_TEXT) — this is the
     voice/accent that gets cloned.

Habibi ships reference clips for Gulf/EGY/IRQ/... but NOT Levantine, so to make
the patient truly sound Damascene, drop a few-second clip of a Damascus speaker
next to this file and point the env vars at it:

    export HABIBI_REF_AUDIO="ref_damascus.wav"
    export HABIBI_REF_TEXT="the exact words spoken in that clip"

If you don't set them it still generates Levantine speech (dialect_id does the
heavy lifting), but the timbre comes from the bundled MSA sample.
"""

import io
import os
import threading
from importlib.resources import files

# Import torch BEFORE soundfile. On Windows, loading soundfile's native
# libsndfile before torch's OpenMP/MKL runtime segfaults the process at import
# time (no traceback). Importing torch first claims that runtime and prevents
# the crash. Do not reorder these.
import torch  # noqa: F401

import soundfile as sf

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from hydra.utils import get_class
from omegaconf import OmegaConf
from cached_path import cached_path

from habibi_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    nfe_step,
    speed,
    sway_sampling_coef,
    target_rms,
)
from habibi_tts.model.utils import dialect_id_map

# --- Config ------------------------------------------------------------------
DIALECT = "LEV"  # Levantine — covers the Damascus dialect. Needs the Unified model.

# Voice to clone. Override with a Damascene clip for the most authentic accent.
_DEFAULT_REF_AUDIO = str(files("habibi_tts").joinpath("assets/MSA.mp3"))
_DEFAULT_REF_TEXT = (
    "كان اللعيب حاضرًا في العديد من الأنشطة والفعاليات المرتبطة بكأس العالم، "
    "مما سمح للجماهير بالتفاعل معه والتقاط الصور التذكارية."
)
REF_AUDIO = os.environ.get("HABIBI_REF_AUDIO", _DEFAULT_REF_AUDIO)
REF_TEXT = os.environ.get("HABIBI_REF_TEXT", _DEFAULT_REF_TEXT)


def _load():
    """Load the Habibi Unified checkpoint + vocoder onto the GPU (once)."""
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    vocoder_name = model_cfg.model.mel_spec.mel_spec_type

    print("Loading Habibi-TTS Unified checkpoint (Damascus/Levantine)...")
    # First run downloads the checkpoint + vocab from Hugging Face (cached after).
    ckpt_file = str(cached_path("hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors"))
    vocab_file = str(cached_path("hf://SWivid/Habibi-TTS/Unified/vocab.txt"))
    ema_model = load_model(
        model_cls, model_arc, ckpt_file,
        mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device,
    )
    vocoder = load_vocoder(vocoder_name=vocoder_name, is_local=False, local_path="", device=device)

    # Preprocess the reference clip once (trims silence / normalizes).
    ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)
    print(f"✅  Habibi-TTS ready on {device}. Dialect={DIALECT}, voice ref={REF_AUDIO}")
    return ema_model, vocoder, vocoder_name, ref_audio, ref_text


_MODEL, _VOCODER, _VOCODER_NAME, _REF_AUDIO, _REF_TEXT = _load()
_DIALECT_ID = dialect_id_map[DIALECT]
# F5-TTS inference uses torch + shared model state -> serialize calls.
_tts_lock = threading.Lock()


def synthesize_to_wav_bytes(text: str) -> bytes:
    """Turn Arabic text into a Damascene-dialect WAV, returned as raw bytes."""
    with _tts_lock:
        wave, sample_rate, _ = infer_process(
            _REF_AUDIO,
            _REF_TEXT,
            text,
            _MODEL,
            _VOCODER,
            mel_spec_type=_VOCODER_NAME,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
            dialect_id=_DIALECT_ID,  # ⑩ Levantine -> Damascus pronunciation
        )
    buf = io.BytesIO()
    sf.write(buf, wave, sample_rate, format="WAV")
    return buf.getvalue()
