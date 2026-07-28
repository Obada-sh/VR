"""System: health check and the browser test page for the voice WebSocket."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import API_KEY, BASE_DIR
from ..sessions import SESSIONS

router = APIRouter(tags=["System"])


@router.get("/health", summary="Health check")
def health():
    return {"status": "ok", "api_key_set": bool(API_KEY), "active_sessions": len(SESSIONS)}


@router.get("/test-voice", include_in_schema=False, summary="Browser test page for /ws/voice")
def test_voice_page():
    """Serve test_voice.html — a mic-driven test client for the voice WebSocket.

    Served from the API on purpose. Browsers only grant microphone access in a
    secure context, and `localhost` counts as one while opening the same file
    over file:// does not. Open http://localhost:8000/test-voice.

    (A LAN address like http://192.168.x.x:8000 is NOT a secure context either,
    so the mic will be blocked there — test from the machine running the API.)
    """
    page = BASE_DIR / "test_voice.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="test_voice.html is not in the project root.")
    return FileResponse(page, media_type="text/html")
