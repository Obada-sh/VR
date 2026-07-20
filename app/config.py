"""Central configuration: environment variables, file paths, LLM constants."""

import os
from dataclasses import dataclass
from pathlib import Path

# --- LLM ---------------------------------------------------------------------
# Every provider below speaks the OpenAI /chat/completions dialect, so switching
# is just a URL + model + key swap. Free tiers are catalogued at
# https://github.com/cheahjs/free-llm-api-resources
#
# LLM_PROVIDERS in .env is an ordered FAILOVER CHAIN, not a single choice: each
# request tries them left to right until one answers. Free tiers go down, run
# out of daily quota, and rotate their model line-ups, so one provider alone is
# not dependable — give the chain at least two.
#
#   LLM_PROVIDERS=google,groq,cerebras
#
# A provider is skipped entirely unless its key is set, so listing one you have
# no key for costs nothing:
#
#   GOOGLE_API_KEY=...      aistudio.google.com/apikey
#   GROQ_API_KEY=...        console.groq.com/keys
#   CEREBRAS_API_KEY=...    cloud.cerebras.ai
#   OPENROUTER_API_KEY=...  openrouter.ai/keys
#   MISTRAL_API_KEY=...     console.mistral.ai
#   OPENCODE_API_KEY=...    the original paid provider
#
# Model IDs go stale as providers rotate their free line-up — `just models`
# lists what a key can actually reach, and <PROVIDER>_MODEL overrides one
# provider's default (e.g. GROQ_MODEL=llama-3.1-8b-instant).
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
    "opencode": {
        "url": "https://opencode.ai/zen/go/v1/chat/completions",
        "model": "deepseek-v4-flash",
        "key_env": "OPENCODE_API_KEY",
    },
}

@dataclass(frozen=True)
class Provider:
    """One resolved link in the failover chain."""

    name: str
    url: str
    model: str
    key_env: str
    key: str

    @property
    def supports_thinking_flag(self) -> bool:
        # The reasoning-off switch is opencode's own extension; other providers
        # 400 on unknown top-level fields, so only send it there.
        return self.name == "opencode"

    def __str__(self) -> str:
        return f"{self.name}/{self.model}"


# Gemini leads: the patient replies and the تشكيل pass are Arabic, and it handles
# diacritics noticeably better than the free Llama/OSS models. Groq and Cerebras
# back it up — both are fast and their free quotas are generous.
LLM_PROVIDERS = os.environ.get("LLM_PROVIDERS", "google,groq,cerebras")
_names = [n.strip().lower() for n in LLM_PROVIDERS.split(",") if n.strip()]

_unknown = [n for n in _names if n not in PROVIDERS]
if _unknown:
    raise SystemExit(
        f"LLM_PROVIDERS has unknown provider(s): {', '.join(_unknown)}. "
        f"Valid: {', '.join(PROVIDERS)}"
    )

# Only providers we hold a key for are usable; the rest are silently skipped so
# the default chain works no matter which keys you have.
CHAIN = [
    Provider(
        name=n,
        url=PROVIDERS[n]["url"],
        model=os.environ.get(f"{n.upper()}_MODEL", "") or PROVIDERS[n]["model"],
        key_env=PROVIDERS[n]["key_env"],
        key=os.environ.get(PROVIDERS[n]["key_env"], ""),
    )
    for n in _names
    if os.environ.get(PROVIDERS[n]["key_env"], "")
]

# Names that were asked for but have no key — reported when the chain is empty.
MISSING_KEYS = [PROVIDERS[n]["key_env"] for n in _names if n not in {p.name for p in CHAIN}]

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
