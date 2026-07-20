"""Central configuration: environment variables, file paths, LLM constants."""

import os
from pathlib import Path

# --- LLM ---------------------------------------------------------------------
# Every provider below speaks the OpenAI /chat/completions dialect, so switching
# is just a URL + model + key swap. Free tiers are catalogued at
# https://github.com/cheahjs/free-llm-api-resources
#
# Pick one with LLM_PROVIDER in .env, then set that provider's key env var:
#
#   LLM_PROVIDER=google      GOOGLE_API_KEY=...     (aistudio.google.com/apikey)
#   LLM_PROVIDER=groq        GROQ_API_KEY=...       (console.groq.com/keys)
#   LLM_PROVIDER=cerebras    CEREBRAS_API_KEY=...   (cloud.cerebras.ai)
#   LLM_PROVIDER=openrouter  OPENROUTER_API_KEY=... (openrouter.ai/keys)
#   LLM_PROVIDER=mistral     MISTRAL_API_KEY=...    (console.mistral.ai)
#   LLM_PROVIDER=opencode    OPENCODE_API_KEY=...   (the original paid provider)
#
# Model IDs go stale as providers rotate their free line-up — `just models`
# lists what your key can actually reach, and LLM_MODEL overrides the default.
PROVIDERS = {
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.5-flash",
        "key_env": "GOOGLE_API_KEY",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    "cerebras": {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "gpt-oss-120b",
        "key_env": "CEREBRAS_API_KEY",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "key_env": "OPENROUTER_API_KEY",
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-large-latest",
        "key_env": "MISTRAL_API_KEY",
    },
    # Free, but it logs every request/response, and its better models require
    # opting into training use at https://logfare.ai/consent. deepseek-v4-flash
    # and minimax-m3 need no opt-in.
    "logfare": {
        "url": "https://logfare.ai/v1/chat/completions",
        "model": "deepseek-v4-flash",
        "key_env": "LOGFARE_API_KEY",
    },
    "opencode": {
        "url": "https://opencode.ai/zen/go/v1/chat/completions",
        "model": "deepseek-v4-flash",
        "key_env": "OPENCODE_API_KEY",
    },
}

# Gemini is the default: the patient replies and the تشكيل pass are Arabic, and
# it handles diacritics noticeably better than the free Llama/OSS models.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "google").strip().lower()
if LLM_PROVIDER not in PROVIDERS:
    raise SystemExit(
        f"LLM_PROVIDER={LLM_PROVIDER!r} is not one of: {', '.join(PROVIDERS)}"
    )

_provider = PROVIDERS[LLM_PROVIDER]
API_KEY_ENV = _provider["key_env"]
API_URL = os.environ.get("LLM_API_URL", "") or _provider["url"]
MODEL_NAME = os.environ.get("LLM_MODEL", "") or _provider["model"]
API_KEY = os.environ.get(API_KEY_ENV, "")
# The reasoning-off switch is opencode's own extension; other providers 400 on
# unknown top-level fields, so only send it there.
SUPPORTS_THINKING_FLAG = LLM_PROVIDER == "opencode"

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
