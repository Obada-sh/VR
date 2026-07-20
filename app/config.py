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
# out of daily quota, and retire model IDs, so one provider alone is not
# dependable — give the chain at least two.
#
#   LLM_PROVIDERS=google:gemini-3.5-flash,google:gemini-3.1-flash-lite,groq
#
# A link is "provider" (its default model) or "provider:model". The same
# provider may appear more than once with different models — that is how you put
# a strong-but-busy model in front of a reliable smaller one.
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
        # flash-lite answers reliably; the bigger flash models 503 under free-tier
        # load, so the default chain puts one of those first and falls back here.
        "model": "gemini-3.1-flash-lite",
        "key_env": "GOOGLE_API_KEY",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
    },
    # Note: this account's key gets 402 Payment Required on chat/completions
    # (listing models still works), so cerebras is not in the default chain.
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


# Gemini leads: the patient replies and the تشكيل pass are Arabic, and measured
# on the تشكيل prompt it was the only family that kept the Damascene wording
# instead of drifting to Fusha (llama/allam both "corrected" كتير -> كثير).
#
# flash-lite leads rather than the bigger gemini-3.5-flash: on this free tier
# 3.5-flash 503s constantly and sometimes hangs to a read timeout, and every
# patient turn costs TWO calls (reply + تشكيل), so a slow first link is felt
# twice. flash-lite also scored best on the تشكيل probe. Put 3.5-flash in front
# if you want its quality and can absorb the stalls.
LLM_PROVIDERS = os.environ.get("LLM_PROVIDERS", "google:gemini-3.1-flash-lite,groq")

# A dead-but-listening provider is worse than a failing one: it burns the whole
# timeout before we can move on. Non-final links get a short leash; the last one
# gets the full budget because there is nothing to fall back to.
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))
LLM_FAILOVER_TIMEOUT = float(os.environ.get("LLM_FAILOVER_TIMEOUT", "20"))

_links = []
for item in LLM_PROVIDERS.split(","):
    item = item.strip()
    if not item:
        continue
    name, _, model = item.partition(":")
    _links.append((name.strip().lower(), model.strip()))

_unknown = sorted({n for n, _ in _links if n not in PROVIDERS})
if _unknown:
    raise SystemExit(
        f"LLM_PROVIDERS has unknown provider(s): {', '.join(_unknown)}. "
        f"Valid: {', '.join(PROVIDERS)}"
    )

# Only providers we hold a key for are usable; the rest are silently skipped so
# the default chain works no matter which keys you have. Model precedence:
# explicit provider:model > <PROVIDER>_MODEL env > the registry default.
CHAIN = [
    Provider(
        name=n,
        url=PROVIDERS[n]["url"],
        model=model or os.environ.get(f"{n.upper()}_MODEL", "") or PROVIDERS[n]["model"],
        key_env=PROVIDERS[n]["key_env"],
        key=os.environ.get(PROVIDERS[n]["key_env"], ""),
    )
    for n, model in _links
    if os.environ.get(PROVIDERS[n]["key_env"], "")
]

# Keys that were asked for but not set — reported when the chain is empty.
_have = {p.name for p in CHAIN}
MISSING_KEYS = sorted({PROVIDERS[n]["key_env"] for n, _ in _links if n not in _have})

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
