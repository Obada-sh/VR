"""Voice: speech-to-text and the full voice turn (audio in / audio out).

NOTE: this module imports tts, which loads the VoxCPM2 model onto the GPU at
import time. main.py's bootstrap (KMP_DUPLICATE_LIB_OK + utf-8 stdout) must run
BEFORE this module is imported — which it does, because main.py sets those up
before including any router.
"""

from urllib.parse import quote

from fastapi import APIRouter, File, Form, Response, UploadFile

import tts  # local VoxCPM2; loads the model onto the GPU at import time

from ..llm import add_tashkeel
from ..schemas import TranscribeResponse
from ..sessions import get_session, run_chat_turn
from ..stt_client import transcribe_and_fix

router = APIRouter(tags=["Voice"])


@router.post(
    "/chat-voice",
    summary="Voice in, patient's spoken reply out (WAV)",
    responses={200: {"content": {"audio/wav": {}}, "description": "The patient's reply as a WAV file."}},
)
def chat_voice(session_id: str = Form(...), file: UploadFile = File(...)):
    """Full voice turn: audio in -> patient's spoken reply (WAV) out.

    Pipeline: Cohere Transcribe transcribes the doctor's audio -> LLM fixes the
    Damascus-dialect STT mistakes -> the patient LLM answers -> an LLM pass adds
    diacritics (تشكيل) to that answer -> VoxCPM2 speaks it in a
    Damascene (Levantine) voice.

    Send as multipart/form-data with fields `session_id` and `file`.
    The response BODY is the patient's reply as audio/wav. The reply text and
    the doctor's transcript are also returned (URL-encoded) in response headers
    `X-Patient-Reply` and `X-Doctor-Transcript` in case the frontend needs them.
    """
    session = get_session(session_id)
    stt = transcribe_and_fix(file)
    reply = run_chat_turn(session, stt["text"])

    # Diacritize (تشكيل) just for TTS so the Damascene words are pronounced
    # correctly; the returned/stored reply text stays clean.
    spoken = add_tashkeel(reply)
    wav_bytes = tts.synthesize_to_wav_bytes(spoken)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="reply.wav"',
            # Header values must be latin-1; URL-encode the Arabic text.
            "X-Patient-Reply": quote(reply),
            "X-Doctor-Transcript": quote(stt["text"]),
        },
    )


@router.post("/transcribe", response_model=TranscribeResponse, summary="Speech-to-text only (corrected transcript)")
def transcribe_audio(file: UploadFile = File(...)):
    """Speech-to-text only: upload audio, get back the corrected Arabic transcript.

    Pipeline: Cohere Transcribe (Arabic) -> LLM pass that fixes the
    Damascus-dialect words the model got wrong. Does NOT touch a chat session.
    """
    stt = transcribe_and_fix(file)
    return TranscribeResponse(
        text=stt["text"],
        raw_text=stt["raw_text"],
        language=stt["language"],
        duration=stt["duration"],
    )
