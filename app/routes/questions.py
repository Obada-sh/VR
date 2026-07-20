"""Questions: the final multiple-choice quiz for the session's scenario.

This is the last level of a session. The questions belong to the scenario the
doctor picked at POST /start, so the frontend never passes a scenario_id here —
just the session_id. Each question has several choices and exactly ONE answer.

Answers are recorded on the session and returned by GET /session/{id}; nothing
in this module tells the doctor whether they were right.
"""

from fastapi import APIRouter, HTTPException

from ..data import load_questions
from ..schemas import AnswerRequest, AnswerResponse, QuestionOut
from ..sessions import get_session, record_answer

router = APIRouter(tags=["Questions"])


def _scenario_questions(session: dict) -> list:
    """The raw question list for this session's scenario (includes the answer key)."""
    questions = load_questions().get(session["scenario_id"])
    if not questions:
        raise HTTPException(
            status_code=404,
            detail=f"No questions defined for scenario_id: {session['scenario_id']}",
        )
    return questions


@router.get(
    "/questions",
    response_model=list[QuestionOut],
    summary="List the quiz questions for a session",
)
def list_questions(session_id: str):
    """Return the questions for this session's scenario, without the answer key."""
    session = get_session(session_id)
    return [
        QuestionOut(id=q["id"], text=q["text"], choices=q["choices"])
        for q in _scenario_questions(session)
    ]


@router.post("/answer", response_model=AnswerResponse, summary="Answer one question")
def answer(req: AnswerRequest):
    """Record the doctor's single choice for one question.

    Answering the same question again replaces the previous answer. The response
    deliberately does NOT say whether the choice was correct — see
    GET /session/{id} for the full review.
    """
    session = get_session(req.session_id)

    question = next(
        (q for q in _scenario_questions(session) if q["id"] == req.question_id), None
    )
    if question is None:
        raise HTTPException(status_code=404, detail=f"Unknown question_id: {req.question_id}")

    choice = next((c for c in question["choices"] if c["id"] == req.choice_id), None)
    if choice is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown choice_id: {req.choice_id} in question {req.question_id}",
        )

    correct_id = question.get("correct_choice_id")
    correct = next((c for c in question["choices"] if c["id"] == correct_id), None)

    record_answer(
        session,
        {
            "question_id": question["id"],
            "question": question["text"],
            "choice_id": choice["id"],
            "answer": choice["text"],
            "correct_choice_id": correct_id,
            "correct_answer": correct["text"] if correct else None,
            "is_correct": choice["id"] == correct_id if correct_id else None,
        },
    )

    return AnswerResponse(question_id=question["id"], choice_id=choice["id"], recorded=True)
