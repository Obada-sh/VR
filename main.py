"""
Patient Simulator Backend — entrypoint
======================================

Session-based FastAPI backend for the medical patient-simulation app.

Flow:
  1. Frontend calls GET /scenarios to show the list of cases (id + name).
  2. Frontend calls POST /start with a chosen scenario_id (and optionally its own
     session_id). The backend creates a session, injects that scenario's case
     text into the patient system prompt, stores it, and returns the session_id.
  3. On every turn the frontend calls POST /chat with { session_id, message } —
     ONLY the latest doctor message. The backend appends it to the stored
     history, calls the LLM, stores + returns the patient's reply.
  4. GET /test-categories and GET /test-categories/{id}/tests list the
     investigations; POST /test-result orders one and returns its result.
  5. GET /questions?session_id=... lists the final multiple-choice quiz for the
     session's case; POST /answer records one choice per question. The questions
     and the doctor's answers come back with GET /session/{id}.
  6. Optionally POST /evaluate with { session_id } to grade the doctor.
  7. POST /chat-voice with { session_id, file } (multipart) is the full VOICE
     version of /chat; POST /transcribe is speech-to-text only.

This file only bootstraps the process and assembles the app — all real code
lives in the app/ package (see app/__init__.py for the map). It stays at the
project root so `uvicorn main:app` keeps working.

Run:
    just main        (or: uvicorn main:app --port 8000)
"""

import os

# faster-whisper (CTranslate2) and Leva-TTS (PyTorch) each bundle their own
# copy of the Intel OpenMP runtime (libiomp5md.dll). Loading both into one
# process segfaults on Windows. This tells the loader to tolerate the duplicate.
# MUST be set before torch / ctranslate2 are imported (i.e. before anything
# that imports tts — which app.routes.voice does).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

# Windows consoles default to cp1252, which can't print emojis OR Arabic text.
# Must happen BEFORE importing app.routes.voice: it imports tts, which prints
# Arabic reference text while loading its model at import time.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import evaluation, questions, sessions, simulation, system, tests, voice

TAGS_METADATA = [
    {"name": "Simulation", "description": "Start a case and chat with the patient (text)."},
    {"name": "Voice", "description": "Speech-to-text and the full voice turn (audio in / audio out)."},
    {"name": "Tests", "description": "Order investigations (labs, imaging, ...) and read their results."},
    {"name": "Questions", "description": "The final multiple-choice quiz for the session's case."},
    {"name": "Evaluation", "description": "Grade the doctor's OSCE performance."},
    {"name": "Sessions", "description": "Inspect or delete stored conversation history."},
    {"name": "System", "description": "Health check."},
]

app = FastAPI(
    title="Patient Simulator Backend",
    version="1.0.0",
    description=(
        "Medical patient-simulation API for Syrian medical students.\n\n"
        "**Typical flow:** `GET /scenarios` → `POST /start` → `POST /chat` "
        "(or `POST /chat-voice` for voice) each turn → `POST /evaluate`.\n\n"
        "Speech runs locally on the GPU: **Cohere Transcribe** for speech-to-text "
        "and **Leva-TTS** for Damascene (Levantine) text-to-speech."
    ),
    openapi_tags=TAGS_METADATA,
)

# Allow the frontend / your friend (any origin on the LAN) to call these endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation.router)
app.include_router(voice.router)
app.include_router(tests.router)
app.include_router(questions.router)
app.include_router(evaluation.router)
app.include_router(sessions.router)
app.include_router(system.router)
