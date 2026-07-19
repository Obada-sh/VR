"""Tests: order investigations (labs, imaging, ...) and read their results."""

from fastapi import APIRouter, HTTPException

from ..data import load_test_categories, load_tests
from ..schemas import TestResultRequest, TestResultResponse
from ..sessions import get_session, record_test

router = APIRouter(tags=["Tests"])


@router.get("/test-categories", summary="List investigation categories")
def list_test_categories():
    """Return [{ id, name }] for the frontend to render the category picker."""
    categories = load_test_categories()
    return [{"id": cid, "name": c["name"]} for cid, c in categories.items()]


@router.get("/test-categories/{category_id}/tests", summary="List the tests in a category")
def list_tests(category_id: str):
    """Return [{ id, name }] for a category — results are NOT included here.

    The doctor must actually order a test (POST /test-result) to see its result,
    so the frontend can't leak every answer just by opening the category.
    """
    tests = load_tests()
    category_tests = tests.get(category_id)
    if category_tests is None:
        raise HTTPException(status_code=404, detail=f"Unknown category_id: {category_id}")
    return [{"id": tid, "name": t["name"]} for tid, t in category_tests.items()]


@router.post("/test-result", response_model=TestResultResponse, summary="Order a test, get its result")
def test_result(req: TestResultRequest):
    """Order one investigation inside a session and return its result.

    The ordered test is recorded on the session (see GET /session/{id}) so the
    frontend can show what was already run.
    """
    session = get_session(req.session_id)

    tests = load_tests()
    category_tests = tests.get(req.category_id)
    if category_tests is None:
        raise HTTPException(status_code=404, detail=f"Unknown category_id: {req.category_id}")

    test = category_tests.get(req.test_id)
    if test is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown test_id: {req.test_id} in category {req.category_id}",
        )

    record_test(session, req.category_id, req.test_id, test["name"])

    return TestResultResponse(id=req.test_id, name=test["name"], result=test["result"])
