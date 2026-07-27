"""Real-time voice WebSocket — the low-latency replacement for POST /chat-voice.

The client protocol is specified in UNITY_VOICE_CLIENT.md; keep the two in sync.

Why this exists
---------------
REST /chat-voice is strictly sequential: transcribe, fix the transcript, answer,
diacritize, render ALL the audio, then reply. Nothing reaches the headset for
10-18 seconds. Here the same work is pipelined instead:

    mic streams in continuously
      -> Silero VAD decides when the doctor stopped talking
      -> STT
      -> the patient LLM is STREAMED, split into sentences as it generates
      -> each sentence goes to the TTS immediately
      -> its audio is pushed to the client while the NEXT sentence is generating

First audio lands in ~2-3s. The dialect-fix LLM pass is gone entirely — the
patient prompt now tells the model to read past STT errors itself.

Threading
---------
The pipeline is blocking (HTTP calls + GPU inference), so it runs in a worker
thread and its output is pumped back into the event loop through a queue. That
keeps the receive loop free to notice a `cancel` mid-utterance.
"""

import asyncio
import json
import threading
from typing import AsyncIterator, Callable, Iterator, Optional, Tuple

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

import tts  # noqa: F401  (already imported by routes.voice; model is on the GPU)

from ..config import VOICE_WS_TASHKEEL
from ..llm import add_tashkeel
from ..sessions import SESSIONS, stream_chat_turn
from ..stt_client import transcribe_pcm
from ..vad import SAMPLE_RATE as MIC_SAMPLE_RATE
from ..vad import SpeechEndpointer

router = APIRouter(tags=["Voice"])

Event = Tuple[str, object]


# --- The pipeline (blocking; runs in a worker thread) -------------------------

def _run_turn(session: dict, audio: np.ndarray, stop: threading.Event) -> Iterator[Event]:
    """One full voice turn. Yields ('transcript'|'reply'|'audio'|'error', payload).

    `stop` is checked between stages and between audio chunks so a barge-in
    stops the work reasonably promptly — a worker thread can't be killed, only
    asked to stand down.
    """
    text = transcribe_pcm(audio, MIC_SAMPLE_RATE)
    if not text:
        yield "error", "No speech detected."
        return
    if stop.is_set():
        return

    yield "transcript", text

    for sentence in stream_chat_turn(session, text):
        if stop.is_set():
            return
        yield "reply", sentence

        spoken = add_tashkeel(sentence) if VOICE_WS_TASHKEEL else sentence
        for pcm in tts.synthesize_stream(spoken):
            if stop.is_set():
                return
            yield "audio", pcm


async def _pump(make_gen: Callable[[], Iterator[Event]]) -> AsyncIterator[Event]:
    """Run a blocking generator in a thread and re-yield its items asynchronously."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def produce() -> None:
        try:
            for item in make_gen():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as e:                      # noqa: BLE001 — surface to client
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=produce, daemon=True).start()

    while True:
        item = await queue.get()
        if item is done:
            return
        yield item


# --- The endpoint -------------------------------------------------------------

@router.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket, session_id: str) -> None:
    """Streaming voice turn. Connect to /ws/voice?session_id=... after POST /start."""
    await websocket.accept()

    session = SESSIONS.get(session_id)
    if session is None:
        await websocket.send_json(
            {"type": "error", "detail": f"Unknown session_id: {session_id}. Call POST /start first."}
        )
        await websocket.close()
        return

    endpointer = SpeechEndpointer()
    turn: Optional[asyncio.Task] = None
    turn_stop: Optional[threading.Event] = None

    async def send(payload: dict) -> None:
        """Send JSON, tolerating a client that has already gone away."""
        if websocket.client_state is WebSocketState.CONNECTED:
            await websocket.send_json(payload)

    async def play_turn(audio: np.ndarray, stop: threading.Event) -> None:
        """Drive one turn and stream its events + audio to the client."""
        try:
            async for kind, payload in _pump(lambda: _run_turn(session, audio, stop)):
                if kind == "audio":
                    if websocket.client_state is WebSocketState.CONNECTED:
                        await websocket.send_bytes(payload)
                elif kind == "reply":
                    await send({"type": "reply", "text": payload, "final": False})
                elif kind == "transcript":
                    await send({"type": "transcript", "text": payload})
                elif kind == "error":
                    await send({"type": "error", "detail": payload})
                    return
        except asyncio.CancelledError:
            raise
        finally:
            # The mic is live again only once the client knows the turn is over.
            await send({"type": "turn_end"})
            endpointer.reset()

    def start_turn(audio: np.ndarray) -> asyncio.Task:
        """Kick off a turn with its own stop flag.

        A fresh Event per turn matters: a cancelled worker thread cannot be
        killed, only asked to stand down, and it may still be unwinding when the
        next turn begins. Sharing one flag would un-cancel that zombie.
        """
        nonlocal turn_stop
        turn_stop = threading.Event()
        return asyncio.create_task(play_turn(audio, turn_stop))

    async def cancel_turn() -> None:
        nonlocal turn
        if turn and not turn.done():
            if turn_stop:
                turn_stop.set()
            turn.cancel()
            try:
                await turn
            except asyncio.CancelledError:
                pass
        turn = None

    await send(
        {
            "type": "ready",
            "in_sample_rate": MIC_SAMPLE_RATE,
            "out_sample_rate": tts.SAMPLE_RATE,
        }
    )

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # --- Control frames ---
            if (raw := message.get("text")) is not None:
                try:
                    cmd = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if cmd.get("type") == "cancel":
                    await cancel_turn()
                    await send({"type": "turn_end"})
                    endpointer.reset()
                elif cmd.get("type") == "ptt":
                    if cmd.get("state") == "down":
                        await cancel_turn()
                        endpointer.force_start()
                        await send({"type": "speech_start"})
                    elif cmd.get("state") == "up":
                        utterance = endpointer.force_end()
                        if utterance is not None and utterance.size:
                            await send({"type": "speech_end"})
                            turn = start_turn(utterance)
                continue

            # --- Audio frames ---
            chunk = message.get("bytes")
            if not chunk:
                continue

            # Half-duplex: while the patient is talking the headset mic is hearing
            # the speakers, so anything arriving now is echo, not the doctor.
            if turn and not turn.done():
                continue

            for kind, utterance in endpointer.feed(chunk):
                if kind == "speech_start":
                    await send({"type": "speech_start"})
                elif kind == "speech_end" and utterance is not None and utterance.size:
                    await send({"type": "speech_end"})
                    turn = start_turn(utterance)
                    break   # stop feeding: the rest of this chunk is mid-turn audio

    except WebSocketDisconnect:
        pass
    finally:
        await cancel_turn()
        if websocket.client_state is WebSocketState.CONNECTED:
            await websocket.close()
