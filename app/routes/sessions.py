"""Sessions: inspect or delete stored conversation history."""

from fastapi import APIRouter

from .. import sessions as store

router = APIRouter(tags=["Sessions"])


@router.get("/session/{session_id}", summary="Get a session's history")
def get_session(session_id: str, include_system: bool = False):
    """Return the stored history for a session.

    By default the system prompt is hidden (it contains the hidden case).
    Pass ?include_system=true to see it too.
    """
    session = store.get_session(session_id)
    messages = session["messages"]
    if not include_system:
        messages = [m for m in messages if m["role"] != "system"]
    return {
        "session_id": session_id,
        "scenario_id": session["scenario_id"],
        "messages": messages,
        "tests": session.get("tests", []),
    }


@router.delete("/session/{session_id}", summary="Delete a session")
def delete_session(session_id: str):
    """Drop a session's stored history (e.g. when the doctor finishes/resets)."""
    return {"deleted": store.delete_session(session_id)}
