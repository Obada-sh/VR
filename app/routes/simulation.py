"""Simulation: list cases, start a session, text chat."""

import uuid

from fastapi import APIRouter, HTTPException

from ..data import load_scenarios
from ..prompts import BASE_SYSTEM_PROMPT
from ..schemas import ChatRequest, ChatResponse, StartRequest, StartResponse
from ..sessions import create_session, get_session, run_chat_turn

router = APIRouter(tags=["Simulation"])


@router.get("/scenarios", summary="List available cases")
def list_scenarios():
    """Return [{ id, name }] for the frontend to render a picker."""
    scenarios = load_scenarios()
    return [{"id": sid, "name": s["name"]} for sid, s in scenarios.items()]


@router.post("/start", response_model=StartResponse, summary="Start a session for a case")
def start(req: StartRequest):
    """Create a session for a chosen scenario and store its injected system prompt."""
    scenarios = load_scenarios()
    scenario = scenarios.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Unknown scenario_id: {req.scenario_id}")

    session_id = req.session_id or str(uuid.uuid4())
    system_prompt = BASE_SYSTEM_PROMPT.format(case_text=scenario["case_text"])
    create_session(session_id, req.scenario_id, system_prompt)

    return StartResponse(session_id=session_id)


@router.post("/chat", response_model=ChatResponse, summary="Send a text message, get the patient reply")
def chat(req: ChatRequest):
    """Append the latest doctor message to the session and reply as the patient."""
    session = get_session(req.session_id)
    reply = run_chat_turn(session, req.message)
    return ChatResponse(reply=reply)
