"""
Patient Simulator Backend
=========================

Session-based FastAPI backend for the medical patient-simulation app.

Flow:
  1. Frontend calls GET /scenarios to show the list of cases (id + name).
  2. Frontend calls POST /start with a chosen scenario_id (and optionally its own
     session_id). The backend creates a session, injects that scenario's case
     text into the patient system prompt, stores it, and returns the session_id.
  3. On every turn the frontend calls POST /chat with { session_id, message } —
     ONLY the latest doctor message. The backend appends it to the stored
     history, calls the LLM, stores + returns the patient's reply.
  4. Optionally POST /evaluate with { session_id } to grade the doctor.
  5. POST /chat-voice with { session_id, file } (multipart) is the full VOICE
     version of /chat: the doctor's audio is transcribed, the Damascus-dialect
     STT mistakes are fixed by the LLM, the patient LLM answers, and the answer
     is spoken back with Habibi-TTS. The response body is audio/wav (the reply
     text is also in the X-Patient-Reply response header).
  6. POST /transcribe with a multipart audio file to get ONLY a Damascus-dialect
     corrected Arabic transcript (no chat session involved).
Both the whisper (STT) and Habibi-TTS models are loaded onto the GPU once, at
server startup.

Conversation history lives HERE, keyed by session_id, in memory.
Note: in-memory storage is per-process — it is cleared on restart and does not
work across multiple uvicorn workers. For production, back SESSIONS with Redis
or a database.

Run:
    export OPENCODE_API_KEY="sk-..."        # (Windows PowerShell: $env:OPENCODE_API_KEY="sk-...")
    uvicorn main:app --reload --port 8000
"""

import os

# faster-whisper (CTranslate2) and Leva-TTS (PyTorch) each bundle their own
# copy of the Intel OpenMP runtime (libiomp5md.dll). Loading both into one
# process segfaults on Windows. This tells the loader to tolerate the duplicate.
# MUST be set before torch / ctranslate2 are imported (i.e. before `import tts`).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import re
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import requests
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Windows consoles default to cp1252, which can't print emojis OR Arabic text.
# Must happen BEFORE importing tts: it prints Arabic reference text while
# loading its model at import time, which would otherwise crash on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import tts  # local Leva-TTS; loads the model onto the GPU at import time

# --- LLM configuration -------------------------------------------------------
API_URL = "https://opencode.ai/zen/go/v1/chat/completions"
MODEL_NAME = "deepseek-v4-flash"
API_KEY = os.environ.get("OPENCODE_API_KEY", "")  # you pass this via the environment

SCENARIOS_PATH = Path(__file__).parent / "scenarios.json"

# --- Speech-to-text (external whisper microservice) ---------------------------
# faster-whisper (CTranslate2) can't share a process with Habibi-TTS (PyTorch)
# on Windows without segfaulting, so STT runs as a SEPARATE service that we call
# over HTTP. Start it FIRST:  python -m uvicorn whisper_service:app --port 8001
WHISPER_SERVICE_URL = os.environ.get("WHISPER_SERVICE_URL", "http://127.0.0.1:8001")

# Prompt used to clean up STT mistakes caused by the Damascus dialect.
STT_FIX_PROMPT = """I have this transcipt from an SST model
{raw_text}
as you can see there are some wrong words, that's because the voice is from Damascus dilect

I want you to fix the mistakes and return only the corrected transcript
Make sure each word is written correctly according to Damascus dilect


"""

# Prompt used to add full Arabic diacritics (تشكيل) to the patient's reply so
# the TTS pronounces the Damascene words correctly.
TASHKEEL_PROMPT = """أضِف التشكيل الكامل (الفتحة، الضمة، الكسرة، السكون، الشدّة، التنوين) لكل حرف في النص التالي.

قواعد صارمة:
- لا تُغيّر الكلمات إطلاقاً، ولا ترتيبها، ولا تحوّلها إلى الفصحى — اللهجة شامية دمشقية، احتفظ بها كما هي حرفاً بحرف.
- شكّل الكلمات كما تُنطق فعلاً باللهجة الشامية (مثال: "هلّق"، "شو"، "تعبانِة").
- لا تُضِف أي كلمة أو شرح أو علامات، أعِد النص نفسه مُشكّلاً فقط.

النص:
{text}
"""

# --- In-memory session store -------------------------------------------------
# session_id -> { "scenario_id": str, "messages": [ {role, content}, ... ] }
# messages[0] is always the system prompt; the rest is the running transcript.
SESSIONS: Dict[str, dict] = {}
_sessions_lock = threading.Lock()  # sync endpoints run in a threadpool -> guard mutations

# --- Patient role-play system prompt (NO scenario baked in) -------------------
# The specific case is injected at session start via {case_text}.
BASE_SYSTEM_PROMPT = """You are an AI playing the role of ONE SPECIFIC patient in a clinical training simulation for Syrian medical students. You must completely forget you are an AI. You have zero medical knowledge — you are simply a sick person who came to see a doctor.

Below this prompt you will receive a full clinical case write-up in formal medical Arabic (patient profile, complaint, history, findings — sometimes even the diagnosis name in the title). This entire write-up is for YOUR INTERNAL UNDERSTANDING ONLY. The patient has never read it, never seen a doctor's note about themselves, and does not know a single medical term in it.

**STEP 1 — BECOME THIS SPECIFIC PERSON:**
From the "بطاقة تعريف الحالة" (or equivalent) section, extract and fully adopt:
- Your name/initials — mention only if the doctor asks.
- Your exact age and gender. **This is critical and non-negotiable**: if the scenario describes a female patient, EVERY verb, adjective, and pronoun you use must be feminine ("تعبانة" not "تعبان", "حاسة" not "حاسس", "رحت/جيت" with feminine agreement, "إنتي" never "إنت" when addressed). If male, use masculine forms throughout.
- Your life situation (job, marital status, recent events like a recent birth, etc.) — this is part of who you are, not a symptom. Mention it only naturally, when relevant to what the doctor asks, never as a diagnostic hint.
You ARE this person. Speak entirely in first person ("أنا...", "عندي...", "صار معي...") — never describe yourself in the clinical third-person style of the write-up.

**STEP 2 — TRANSLATE EVERY CLINICAL DETAIL INTO LAYMAN'S FEELINGS:**
Nothing from the write-up may come out of your mouth in medical language. Convert it into how a non-medical person would actually describe it. Examples:
* *Write-up:* "التهاب مفاصل متعدد واسع النطاق" → *You say:* "مفاصلي كلها عم توجعني، إيديّ ورجليّ وركبي."
* *Write-up:* "طفح جلدي حساس للضوء على الوجه" → *You say:* "طلعلي شي حمرة بوجهي، وبتزيد لما بطلع عالشمس."
* *Write-up:* "آفات مؤلمة نخرية براحة اليدين وأخمص القدمين" → *You say:* "صاير في جروح بكف إيدي وتحت رجليّ، بتوجعني كتير وما عم قدر لمس الشي."
* *Write-up:* "فقدان ملحوظ في الوزن" → *You say:* "لاحظت إنو وزني نزل بشكل واضح من دون ما أحاول."
**NEVER** say the diagnosis name — not in Arabic, not in English, not even a piece of it — even though it may be written as the title of your scenario. You have genuinely never heard this word in your life.

**3. NATIVE DAMASCUS DIALECT ONLY (لهجة شامية دمشقية — مدينة دمشق تحديداً):**
You MUST speak in authentic, conversational **Damascene** Arabic specifically — not generic Levantine, not Aleppine, not coastal (لاذقية/طرطوس), not rural/Bedouin, and absolutely not Gulf/Saudi.
* **DO NOT** use Fusha (Modern Standard Arabic) or literal translations.
* **Use natural Damascene markers:** "والله يا دكتور...", "هلق...", "هيك...", "شو في...", "تعبانة والله...", "ما بعرف شو فيني...", "على راسي دكتور...", "خير إن شاء الله".

**4. THE "DRIP-FEED" RULE (NEVER DUMP SYMPTOMS):**
* When the doctor asks "كيفك" or "شو حاسة", **DO NOT list everything at once.** Mention ONLY your primary complaint, in one short sentence.
* Wait for the doctor to ask follow-up questions before revealing anything else from the scenario. Let the doctor extract information from you — don't volunteer it.

**5. REALISTIC, SHORT REACTIONS:**
Keep every answer to 1 short sentence (max 2).
* **Greeting:** If the doctor says "مرحبا", say "أهلين دكتور" or "يا هلا دكتور" — stop there, don't mention symptoms yet.
* **Reacting to a Diagnosis:** Sound mildly worried but clueless: "عن جد؟ خير إن شاء الله دكتور، شو هاد؟ بيخوف؟"
* **Reacting to Tests/Images:** Agree simply: "على راسي دكتور، اللي بتشوفه. وين بساويهم هدول؟"
* **Reacting to Medication/Injections:** "تكرم دكتور، إن شاء الله بطيب عليهم؟"

**6. SPOKEN WORDS ONLY — NO STAGE DIRECTIONS:**
Output ONLY the words the patient actually says out loud. NEVER describe actions, gestures, tone, or emotions. Do NOT write things like "(تتنهد)", "(بصوت متعب)", "(تدخل العيادة)", "*تبتسم*", or any narration between parentheses or asterisks. Reply with the plain spoken sentence and nothing else.

**EXAMPLE OF CORRECT BEHAVIOR (with a female patient):**
🧑‍⚕️ Doctor: مرحبا
👤 You: أهلين دكتور، يا هلا.
🧑‍⚕️ Doctor: شو في؟ شو عم تحسي؟
👤 You: والله يا دكتور، مفاصلي عم توجعني كتير من شي أسبوعين.
🧑‍⚕️ Doctor: وين بالتحديد؟
👤 You: بإيديّ ورجليّ وركبي، وحتى صعب عليي حمل طفلتي من الوجع.

**YOUR FULL CASE SCENARIO (INTERNAL ONLY — NEVER READ IT OUT LOUD, NEVER NAME THE DIAGNOSIS):**
{case_text}
"""


# --- Scenario loading --------------------------------------------------------
def load_scenarios() -> dict:
    """Read scenarios.json fresh each call so edits don't require a restart."""
    with open(SCENARIOS_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- Request / response models ----------------------------------------------
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


class EvaluateRequest(BaseModel):
    session_id: str


class EvaluateResponse(BaseModel):
    evaluation: str


# --- Helpers -----------------------------------------------------------------
def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks the model may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_stage_directions(text: str) -> str:
    """Drop narrated actions/emotions so only the patient's spoken words remain.

    Removes anything wrapped in parentheses ( ) / （ ） or asterisks *...*,
    e.g. "(تتنهد بتعب) أهلين دكتور" -> "أهلين دكتور".
    """
    text = re.sub(r"[\(（][^)）]*[\)）]", "", text)  # ( ... ) and （ ... ）
    text = re.sub(r"\*[^*]*\*", "", text)            # *...*
    text = re.sub(r"[ \t]{2,}", " ", text)           # collapse leftover gaps
    return text.strip()


def _call_llm(messages: List[dict], max_tokens: int, temperature: float) -> str:
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENCODE_API_KEY environment variable is not set.",
        )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        detail = f"LLM request failed: {e}"
        resp_text = getattr(getattr(e, "response", None), "text", None)
        if resp_text:
            detail += f" | server said: {resp_text}"
        raise HTTPException(status_code=502, detail=detail)

    data = resp.json()
    return _strip_think(data["choices"][0]["message"]["content"].strip())


def _get_session(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown session_id: {session_id}. Call POST /start first.",
        )
    return session


# --- App ---------------------------------------------------------------------
TAGS_METADATA = [
    {"name": "Simulation", "description": "Start a case and chat with the patient (text)."},
    {"name": "Voice", "description": "Speech-to-text and the full voice turn (audio in / audio out)."},
    {"name": "Evaluation", "description": "Grade the doctor's OSCE performance."},
    {"name": "Sessions", "description": "Inspect or delete stored conversation history."},
    {"name": "System", "description": "Health check."},
]

app = FastAPI(
    title="Patient Simulator Backend",
    version="1.0.0",
    description=(
        "Medical patient-simulation API for Syrian medical students.\n\n"
        "**Typical flow:** `GET /scenarios` → `POST /start` → `POST /chat` "
        "(or `POST /chat-voice` for voice) each turn → `POST /evaluate`.\n\n"
        "Speech runs locally on the GPU: **faster-whisper** for speech-to-text "
        "and **Habibi-TTS** for Damascene (Levantine) text-to-speech."
    ),
    openapi_tags=TAGS_METADATA,
)

# Allow the frontend / your friend (any origin on the LAN) to call these endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/scenarios", tags=["Simulation"], summary="List available cases")
def list_scenarios():
    """Return [{ id, name }] for the frontend to render a picker."""
    scenarios = load_scenarios()
    return [{"id": sid, "name": s["name"]} for sid, s in scenarios.items()]


@app.post("/start", response_model=StartResponse, tags=["Simulation"], summary="Start a session for a case")
def start(req: StartRequest):
    """Create a session for a chosen scenario and store its injected system prompt."""
    scenarios = load_scenarios()
    scenario = scenarios.get(req.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Unknown scenario_id: {req.scenario_id}")

    session_id = req.session_id or str(uuid.uuid4())
    system_prompt = BASE_SYSTEM_PROMPT.format(case_text=scenario["case_text"])

    with _sessions_lock:
        SESSIONS[session_id] = {
            "scenario_id": req.scenario_id,
            "messages": [{"role": "system", "content": system_prompt}],
        }

    return StartResponse(session_id=session_id)


def _run_chat_turn(session: dict, message: str) -> str:
    """Append a doctor message to the session, call the LLM, store + return the reply."""
    with _sessions_lock:
        session["messages"].append({"role": "user", "content": message})
        # Snapshot the messages to send while holding the lock.
        messages_snapshot = list(session["messages"])

    reply = _strip_stage_directions(_call_llm(messages_snapshot, max_tokens=800, temperature=0.2))

    with _sessions_lock:
        session["messages"].append({"role": "assistant", "content": reply})

    return reply


def _add_tashkeel(text: str) -> str:
    """Add full Arabic diacritics (تشكيل) to the reply for accurate TTS.

    Runs a dedicated LLM pass. If it fails or returns nothing usable, fall back
    to the original (undiacritized) text so speech is never blocked.
    """
    try:
        diacritized = _call_llm(
            [{"role": "user", "content": TASHKEEL_PROMPT.format(text=text)}],
            max_tokens=800,
            temperature=0.0,
        ).strip()
    except HTTPException:
        return text
    return diacritized or text


def _transcribe_and_fix(file: UploadFile) -> dict:
    """Send an uploaded audio file to the whisper STT service, then run the
    dialect-fix LLM pass on the returned transcript.

    Returns {text, raw_text, language, duration}: `text` is the corrected
    transcript, `raw_text` is whisper's exact output.
    """
    audio_bytes = file.file.read()
    filename = file.filename or "audio.wav"
    content_type = file.content_type or "application/octet-stream"

    try:
        resp = requests.post(
            f"{WHISPER_SERVICE_URL}/transcribe",
            files={"file": (filename, audio_bytes, content_type)},
            timeout=300,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Whisper STT service unreachable at {WHISPER_SERVICE_URL}. "
                f"Start it first:  python -m uvicorn whisper_service:app --port 8001  ({e})"
            ),
        )

    stt = resp.json()
    raw_text = stt["raw_text"]

    if not raw_text:
        raise HTTPException(status_code=400, detail="No speech detected in the audio.")

    corrected = _call_llm(
        [{"role": "user", "content": STT_FIX_PROMPT.format(raw_text=raw_text)}],
        max_tokens=800,
        temperature=0.2,
    )

    return {
        "text": corrected,
        "raw_text": raw_text,
        "language": stt["language"],
        "duration": stt["duration"],
    }


@app.post("/chat", response_model=ChatResponse, tags=["Simulation"], summary="Send a text message, get the patient reply")
def chat(req: ChatRequest):
    """Append the latest doctor message to the session and reply as the patient."""
    session = _get_session(req.session_id)
    reply = _run_chat_turn(session, req.message)
    return ChatResponse(reply=reply)


@app.post(
    "/chat-voice",
    tags=["Voice"],
    summary="Voice in, patient's spoken reply out (WAV)",
    responses={200: {"content": {"audio/wav": {}}, "description": "The patient's reply as a WAV file."}},
)
def chat_voice(session_id: str = Form(...), file: UploadFile = File(...)):
    """Full voice turn: audio in -> patient's spoken reply (WAV) out.

    Pipeline: faster-whisper transcribes the doctor's audio -> LLM fixes the
    Damascus-dialect STT mistakes -> the patient LLM answers -> an LLM pass adds
    diacritics (تشكيل) to that answer -> Leva-TTS speaks it in a female
    Damascene (Levantine) voice.

    Send as multipart/form-data with fields `session_id` and `file`.
    The response BODY is the patient's reply as audio/wav. The reply text and
    the doctor's transcript are also returned (URL-encoded) in response headers
    `X-Patient-Reply` and `X-Doctor-Transcript` in case the frontend needs them.
    """
    session = _get_session(session_id)
    stt = _transcribe_and_fix(file)
    reply = _run_chat_turn(session, stt["text"])

    # Diacritize (تشكيل) just for TTS so the Damascene words are pronounced
    # correctly; the returned/stored reply text stays clean.
    spoken = _add_tashkeel(reply)
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


@app.post("/transcribe", response_model=TranscribeResponse, tags=["Voice"], summary="Speech-to-text only (corrected transcript)")
def transcribe_audio(file: UploadFile = File(...)):
    """Speech-to-text only: upload audio, get back the corrected Arabic transcript.

    Pipeline: faster-whisper (large-v3, Arabic) -> LLM pass that fixes the
    Damascus-dialect words whisper got wrong. Does NOT touch a chat session.
    """
    stt = _transcribe_and_fix(file)
    return TranscribeResponse(
        text=stt["text"],
        raw_text=stt["raw_text"],
        language=stt["language"],
        duration=stt["duration"],
    )


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"], summary="Grade the doctor (OSCE)")
def evaluate(req: EvaluateRequest):
    """Grade the doctor's OSCE performance using the stored session transcript."""
    session = _get_session(req.session_id)

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

    evaluator_prompt = f"""أنت أستاذ طب استشاري (Consultant Examiner) صارم جداً، تقيّم أداء طبيب في امتحان سريري عملي (OSCE). أنت معروف بأنك مُمتحِن قاسٍ لا يجامل، ومعاييرك عالية جداً. مهمتك حماية سلامة المرضى، لذلك لا تمنح درجات مجانية أبداً.

البيانات الطبية المرجعية الصحيحة (Gold Standard) لحالة هذا المريض:
{gold_json}

السجل الكامل للمحادثة بين الطبيب والمريض:
{transcript}

==================================================
قواعد التقييم الصارمة (اقرأها جيداً والتزم بها حرفياً):
==================================================

1. قيّم فقط ما قاله الطبيب فعلياً في السجل. ممنوع منعاً باتاً أن تفترض أو تتخيل أو "تحسن الظن" بأن الطبيب كان يقصد شيئاً لم يقله صراحة. إذا لم يُكتب في السجل، فهو لم يحدث = صفر.

2. مبدأ "غير المذكور = غير موجود": أي سؤال، فحص، تحليل، صورة، دواء، أو نصيحة لم يذكرها الطبيب بوضوح تُحتسب كنقطة ضعف ونقص في الدرجة.

3. العقوبات الإجبارية (Critical Fails): امنح الطبيب "راسب" (Fail) بغض النظر عن باقي الأداء في أي من الحالات التالية:
  - وصل لتشخيص خاطئ أو لم يصل لأي تشخيص.
  - وصف دواءً خطيراً أو غير مناسب للحالة (Patient Safety Risk).
  - أنهى الاستشارة دون أخذ قصة مرضية كافية (أقل من 4 أسئلة استكشافية حقيقية).
  - لم يطلب أي استقصاء تشخيصي أساسي ضروري لتأكيد التشخيص.

4. كن بخيلاً بالدرجات. الطبيب المتوسط يحصل على درجة متوسطة (50-65%) وليس عالية. الدرجة فوق 85% تُمنح فقط لأداء شبه مثالي يغطي كل المحاور تقريباً.

5. ممنوع المجاملة أو العبارات التشجيعية المجانية. كن مباشراً ونقدياً.

==================================================
نظام التقييم (املأ كل محور بدرجة رقمية صريحة):
==================================================

المحور 1: أخذ القصة المرضية (History Taking) — 25 نقطة
- اذكر الأسئلة الجيدة التي طرحها فعلاً (اقتباس مختصر).
- اذكر بالتفصيل كل سؤال أساسي *نسيه*.
- درجة المحور:  / 25

المحور 2: الفحص السريري (Examination) — 10 نقاط
- هل ذكر أنه سيفحص المريض (التسمع، العلامات الحيوية)؟ إن لم يفعل = صفر.
- درجة المحور:  / 10

المحور 3: الاستقصاءات (Investigations) — 20 نقطة
- قارن ما طلبه بالاستقصاءات المرجعية.
- اذكر صراحة كل استقصاء ضروري نسيه.
- درجة المحور:  / 20

المحور 4: صحة التشخيص (Diagnosis) — 20 نقطة
- ما التشخيص الذي وصل إليه؟ هل هو صحيح؟ هل ذكّر بالتشخيص التفريقي؟
- إن كان التشخيص خاطئاً = صفر + Critical Fail عام.
- درجة المحور:  / 20

المحور 5: الخطة العلاجية (Management) — 20 نقطة
- قارن خطته (الدوائية واللادوائية والوقاية) بالبيانات المرجعية.
- اذكر كل دواء أو نصيحة أساسية نسيها.
- إن وصف دواءً خاطئاً/خطيراً، نبّه عليه بوضوح كخطر على سلامة المريض.
- درجة المحور:  / 20

المحور 6: التواصل (Communication) — 5 نقاط
- هل كان واضحاً، شرح للمريض حالته، وطمأنه؟
- درجة المحور:  / 5

==================================================
الخلاصة النهائية (إلزامية):
==================================================
- الدرجة الإجمالية: __ / 100
- النتيجة: (ناجح بامتياز / ناجح / ناجح بصعوبة / راسب) — طبّق قواعد الـ Critical Fail بصرامة.
- أهم 3 أخطاء يجب إصلاحها فوراً (مرتبة حسب الخطورة على المريض).
- حكم المُمتحِن: جملة أو جملتان مباشرتان وصريحتان عن المستوى العام.

اجعل الرد كله بالعربية، منظماً بعناوين واضحة، ودقيقاً وصارماً.
"""

    evaluation = _call_llm(
        [{"role": "user", "content": evaluator_prompt}],
        max_tokens=2500,
        temperature=0.2,
    )
    return EvaluateResponse(evaluation=evaluation)


@app.get("/session/{session_id}", tags=["Sessions"], summary="Get a session's history")
def get_session(session_id: str, include_system: bool = False):
    """Return the stored history for a session.

    By default the system prompt is hidden (it contains the hidden case).
    Pass ?include_system=true to see it too.
    """
    session = _get_session(session_id)
    messages = session["messages"]
    if not include_system:
        messages = [m for m in messages if m["role"] != "system"]
    return {
        "session_id": session_id,
        "scenario_id": session["scenario_id"],
        "messages": messages,
    }


@app.delete("/session/{session_id}", tags=["Sessions"], summary="Delete a session")
def delete_session(session_id: str):
    """Drop a session's stored history (e.g. when the doctor finishes/resets)."""
    with _sessions_lock:
        existed = SESSIONS.pop(session_id, None) is not None
    return {"deleted": existed}


@app.get("/health", tags=["System"], summary="Health check")
def health():
    return {"status": "ok", "api_key_set": bool(API_KEY), "active_sessions": len(SESSIONS)}
