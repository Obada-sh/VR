"""Patient Simulator backend package.

The FastAPI app is assembled in the top-level main.py (kept there so
`uvicorn main:app` and the justfile keep working). This package holds
everything else:

    config.py       env vars, file paths, LLM constants
    prompts.py      every prompt template (patient role-play, STT fix, tashkeel, evaluator)
    schemas.py      pydantic request/response models
    data.py         loaders for scenarios.json / test_categories.json / tests.json
    llm.py          LLM HTTP call + reply cleanup + diacritization
    sessions.py     in-memory session store (history + ordered tests) and the chat turn
    stt_client.py   client for the whisper STT microservice (:8001)
    routes/         one APIRouter per OpenAPI tag
"""
