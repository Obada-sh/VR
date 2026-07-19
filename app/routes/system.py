"""System: health check."""

from fastapi import APIRouter

from ..config import API_KEY
from ..sessions import SESSIONS

router = APIRouter(tags=["System"])


@router.get("/health", summary="Health check")
def health():
    return {"status": "ok", "api_key_set": bool(API_KEY), "active_sessions": len(SESSIONS)}
