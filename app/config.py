"""Central configuration: environment variables, file paths, LLM constants."""

import os
from pathlib import Path

# --- LLM ---------------------------------------------------------------------
API_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MODEL_NAME = "mimo-v2.5"
API_KEY = os.environ.get("OPENCODE_API_KEY", "")  # you pass this via the environment

# --- Data files (project root, next to main.py) ------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = BASE_DIR / "scenarios.json"
TEST_CATEGORIES_PATH = BASE_DIR / "test_categories.json"
TESTS_PATH = BASE_DIR / "tests.json"
QUESTIONS_PATH = BASE_DIR / "questions.json"

# --- Speech-to-text (external ASR microservice) -------------------------------
# STT uses CohereLabs/cohere-transcribe-arabic-07-2026, which needs transformers>=5.4,
# while the VoxCPM2 TTS engine pins an older transformers. They can't share a venv, so STT runs as a
# SEPARATE service (its own venv + process) that we call over HTTP.
# Start it FIRST:  just stt   (or: python -m uvicorn whisper_service:app --port 8001)
WHISPER_SERVICE_URL = os.environ.get("WHISPER_SERVICE_URL", "http://127.0.0.1:8001")
