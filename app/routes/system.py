"""System: health check."""

from fastapi import APIRouter

from ..config import CHAIN
from ..sessions import SESSIONS

router = APIRouter(tags=["System"])


@router.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "llm_chain": [str(p) for p in CHAIN],
        "active_sessions": len(SESSIONS),
    }
