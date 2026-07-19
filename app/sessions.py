"""In-memory session store and the core chat turn.

session_id -> { "scenario_id": str,
                "messages":    [ {role, content}, ... ],   # [0] is the system prompt
                "tests":       [ {category_id, id, name}, ... ] }  # ordered investigations

Note: in-memory storage is per-process — it is cleared on restart and does not
work across multiple uvicorn workers. For production, back SESSIONS with Redis
or a database.
"""

import threading
from typing import Dict

from fastapi import HTTPException

from .llm import call_llm, strip_stage_directions

SESSIONS: Dict[str, dict] = {}
_lock = threading.Lock()  # sync endpoints run in a threadpool -> guard mutations


def create_session(session_id: str, scenario_id: str, system_prompt: str) -> None:
    with _lock:
        SESSIONS[session_id] = {
            "scenario_id": scenario_id,
            "messages": [{"role": "system", "content": system_prompt}],
            "tests": [],  # investigations the doctor ordered, in order
        }


def get_session(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown session_id: {session_id}. Call POST /start first.",
        )
    return session


def delete_session(session_id: str) -> bool:
    """Drop a session's stored history. Returns whether it existed."""
    with _lock:
        return SESSIONS.pop(session_id, None) is not None


def record_test(session: dict, category_id: str, test_id: str, name: str) -> None:
    """Remember that the doctor ordered this investigation in this session."""
    with _lock:
        session.setdefault("tests", []).append(
            {"category_id": category_id, "id": test_id, "name": name}
        )


def run_chat_turn(session: dict, message: str) -> str:
    """Append a doctor message to the session, call the LLM, store + return the reply."""
    with _lock:
        session["messages"].append({"role": "user", "content": message})
        # Snapshot the messages to send while holding the lock.
        messages_snapshot = list(session["messages"])

    reply = strip_stage_directions(call_llm(messages_snapshot, max_tokens=800, temperature=0.2))

    with _lock:
        session["messages"].append({"role": "assistant", "content": reply})

    return reply
