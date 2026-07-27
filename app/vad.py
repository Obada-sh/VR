"""Streaming voice-activity detection and utterance endpointing (Silero VAD).

The VR client streams microphone audio continuously and never decides anything
about it. This module watches that stream and answers one question: when did the
doctor start talking, and when did they stop?

Silero VAD is a ~2 MB ONNX model. We run it on the CPU via onnxruntime rather
than through torch, so it never contends with the GPU that Leva-TTS is using.

Frame math: Silero requires EXACTLY 512 samples per call at 16 kHz, which is
32 ms. Every duration below is converted into a count of those frames.
"""

import collections
from typing import Iterator, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512                      # what Silero requires at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2          # PCM16 -> 2 bytes per sample
FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000   # 32.0 ms

# --- Tuning ------------------------------------------------------------------
# Raise THRESHOLD if a noisy room keeps triggering; lower it if quiet speech is
# missed. 0.5 is Silero's default and is a sane starting point for a headset mic.
THRESHOLD = 0.5
MIN_SPEECH_MS = 250      # ignore blips shorter than this (coughs, clicks, taps)
MIN_SILENCE_MS = 700     # how long a pause must last before we call it "done"
PRE_ROLL_MS = 300        # audio kept from BEFORE the trigger so no word is clipped
MAX_UTTERANCE_S = 30     # hard stop, so a stuck-open mic can't buffer forever

_MIN_SPEECH_FRAMES = int(MIN_SPEECH_MS / FRAME_MS)
_MIN_SILENCE_FRAMES = int(MIN_SILENCE_MS / FRAME_MS)
_PRE_ROLL_FRAMES = int(PRE_ROLL_MS / FRAME_MS)
_MAX_UTTERANCE_FRAMES = int(MAX_UTTERANCE_S * 1000 / FRAME_MS)


def _pcm16_to_float32(raw: bytes) -> np.ndarray:
    """Decode little-endian PCM16 bytes into float32 samples in [-1, 1]."""
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class SpeechEndpointer:
    """Feed it PCM16 bytes, it yields ('speech_start', None) / ('speech_end', audio).

    One instance per WebSocket connection: the Silero model carries internal
    recurrent state, so two concurrent sessions must NOT share one instance.

    Push-to-talk: when the client sends explicit ptt down/up, call
    force_start()/force_end() instead and the VAD is bypassed for that utterance.
    """

    def __init__(self) -> None:
        from silero_vad import load_silero_vad

        # onnx=True -> runs on onnxruntime (CPU), keeping the GPU free for TTS.
        self._model = load_silero_vad(onnx=True)
        self._buf = bytearray()                                   # partial frame carry-over
        self._pre_roll: collections.deque = collections.deque(maxlen=max(_PRE_ROLL_FRAMES, 1))
        self._utterance: list = []
        self._speaking = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._forced = False        # True while a push-to-talk utterance is open

    # --- Main entry point ----------------------------------------------------
    def feed(self, pcm16: bytes) -> Iterator[Tuple[str, Optional[np.ndarray]]]:
        """Consume a chunk of mic audio and yield any endpointing events.

        The chunk may be any length; whatever doesn't fill a 512-sample frame is
        carried over to the next call.
        """
        self._buf.extend(pcm16)

        while len(self._buf) >= FRAME_BYTES:
            frame_bytes = bytes(self._buf[:FRAME_BYTES])
            del self._buf[:FRAME_BYTES]
            frame = _pcm16_to_float32(frame_bytes)

            # In push-to-talk mode the client owns the boundaries; just collect.
            if self._forced:
                self._utterance.append(frame)
                continue

            yield from self._step(frame)

    def _step(self, frame: np.ndarray) -> Iterator[Tuple[str, Optional[np.ndarray]]]:
        """Run one 32 ms frame through the VAD state machine."""
        import torch

        with torch.no_grad():
            prob = self._model(torch.from_numpy(frame), SAMPLE_RATE).item()
        is_speech = prob >= THRESHOLD

        if not self._speaking:
            # Always keep the most recent frames so that when speech does start
            # we can prepend the audio from just BEFORE the trigger fired.
            self._pre_roll.append(frame)

            if is_speech:
                self._speech_frames += 1
                if self._speech_frames >= _MIN_SPEECH_FRAMES:
                    self._speaking = True
                    self._silence_frames = 0
                    self._utterance = list(self._pre_roll)   # pre-roll included
                    self._pre_roll.clear()
                    yield "speech_start", None
            else:
                self._speech_frames = 0      # blip, not speech — reset
            return

        # --- Currently speaking ---
        self._utterance.append(frame)

        if is_speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            if self._silence_frames >= _MIN_SILENCE_FRAMES:
                yield "speech_end", self._finish()
                return

        if len(self._utterance) >= _MAX_UTTERANCE_FRAMES:
            yield "speech_end", self._finish()

    # --- Push-to-talk overrides ---------------------------------------------
    def force_start(self) -> None:
        """Client pressed the talk button: open an utterance, bypass the VAD."""
        self._forced = True
        self._speaking = True
        self._utterance = list(self._pre_roll)
        self._pre_roll.clear()
        self._speech_frames = 0
        self._silence_frames = 0

    def force_end(self) -> Optional[np.ndarray]:
        """Client released the talk button: close the utterance and return it."""
        if not self._forced:
            return None
        self._forced = False
        return self._finish()

    # --- Housekeeping --------------------------------------------------------
    def _finish(self) -> np.ndarray:
        """Close the current utterance and reset for the next one."""
        audio = (
            np.concatenate(self._utterance)
            if self._utterance
            else np.zeros(0, dtype=np.float32)
        )
        self.reset()
        return audio

    def reset(self) -> None:
        """Drop any in-progress utterance and clear the model's recurrent state.

        Called after every utterance, and whenever we stop trusting the input —
        e.g. while the patient is speaking and the mic is hearing the speakers.
        """
        self._utterance = []
        self._pre_roll.clear()
        self._speaking = False
        self._forced = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._buf.clear()
        self._model.reset_states()
