"""Central configuration: environment variables, file paths, LLM constants."""

import os
from pathlib import Path

# --- LLM ---------------------------------------------------------------------
API_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MODEL_NAME = "deepseek-v4-pro"
API_KEY = os.environ.get("OPENCODE_API_KEY", "")  # you pass this via the environment

# --- Data files (project root, next to main.py) ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = BASE_DIR / "scenarios.json"
TEST_CATEGORIES_PATH = BASE_DIR / "test_categories.json"
TESTS_PATH = BASE_DIR / "tests.json"
QUESTIONS_PATH = BASE_DIR / "questions.json"

# --- Speech-to-text (external ASR microservice) -------------------------------
# STT uses CohereLabs/cohere-transcribe-arabic-07-2026, which needs transformers>=5.4,
# while Leva-TTS pins transformers<5. They can't share a venv, so STT runs as a
# SEPARATE service (its own venv + process) that we call over HTTP.
# Start it FIRST:  just stt   (or: python -m uvicorn whisper_service:app --port 8001)
WHISPER_SERVICE_URL = os.environ.get("WHISPER_SERVICE_URL", "http://127.0.0.1:8001")

# --- Voice WebSocket ----------------------------------------------------------
# The REST /chat-voice path runs an LLM pass that adds Arabic diacritics (تشكيل)
# before speaking. On the WebSocket path that pass is OFF by default: it is a
# full LLM round trip sitting in front of the FIRST audio chunk, and Leva-TTS
# already applies Levantine diacritics from its own lexicon
# (leva_tts/text/processor.py). Listen to both and decide:
#     set VOICE_WS_TASHKEEL=1   -> better pronunciation, ~1-3s slower to first audio
VOICE_WS_TASHKEEL = os.environ.get("VOICE_WS_TASHKEEL", "0") == "1"
