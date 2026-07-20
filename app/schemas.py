"""Pydantic request/response models for every endpoint."""

from typing import List, Optional

from pydantic import BaseModel


class StartRequest(BaseModel):
    scenario_id: str
    session_id: Optional[str] = None  # frontend may supply its own; else we generate one


class StartResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str  # ONLY the latest doctor message


class ChatResponse(BaseModel):
    reply: str


class TranscribeResponse(BaseModel):
    text: str       # dialect-corrected transcript (what you should show/use)
    raw_text: str   # exact whisper output, before LLM correction
    language: str
    duration: float


class TestResultRequest(BaseModel):
    session_id: str
    category_id: str
    test_id: str


class TestResultResponse(BaseModel):
    id: str
    name: str
    result: str


class ChoiceOut(BaseModel):
    id: str
    text: str


class QuestionOut(BaseModel):
    """A quiz question as the doctor sees it — correct_choice_id is never sent."""

    id: str
    text: str
    choices: List[ChoiceOut]


class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    choice_id: str  # exactly one choice


class AnswerResponse(BaseModel):
    question_id: str
    choice_id: str
    recorded: bool


class EvaluateRequest(BaseModel):
    session_id: str


class EvaluateResponse(BaseModel):
    evaluation: str
