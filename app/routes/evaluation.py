"""Evaluation: grade the doctor's OSCE performance from the stored transcript."""

import json

from fastapi import APIRouter

from ..data import load_scenarios
from ..llm import call_llm
from ..prompts import EVALUATOR_PROMPT
from ..schemas import EvaluateRequest, EvaluateResponse
from ..sessions import get_session

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate", response_model=EvaluateResponse, summary="Grade the doctor (OSCE)")
def evaluate(req: EvaluateRequest):
    """Grade the doctor's OSCE performance using the stored session transcript."""
    session = get_session(req.session_id)

    scenarios = load_scenarios()
    scenario = scenarios.get(session["scenario_id"], {})
    gold = scenario.get("gold_standard")
    gold_json = json.dumps(gold, ensure_ascii=False, indent=2) if gold else "{}"

    # Build the transcript from stored messages, skipping the system prompt.
    transcript = ""
    for m in session["messages"]:
        if m["role"] == "user":
            transcript += f"الطبيب: {m['content']}\n"
        elif m["role"] == "assistant":
            transcript += f"المريض: {m['content']}\n"

    evaluation = call_llm(
        [{"role": "user", "content": EVALUATOR_PROMPT.format(gold_json=gold_json, transcript=transcript)}],
        max_tokens=2500,
        temperature=0.2,
    )
    return EvaluateResponse(evaluation=evaluation)
