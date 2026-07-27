# Unity VR Voice Client — Implementation Brief

Contract for the real-time voice link between the Unity VR headset client and the
Patient Simulator backend. The backend side of this is already implemented; build
the Unity side to match this document exactly.

## What changed and why

The old flow was: push-to-talk button → record whole clip → `POST /chat-voice` →
wait ~10-18s → receive one complete WAV → play it. The wait was the problem.

The new flow is a **persistent WebSocket**. The mic streams continuously, the
server detects when the doctor stops speaking (VAD, server-side), and the
patient's reply comes back **as it is generated, one sentence at a time**. The
first audio arrives in ~2-3s instead of ~15s.

Two things move off the Unity client's plate entirely:

- **You do not implement VAD.** The server runs Silero VAD on the incoming
  stream and tells you when speech started and ended.
- **You do not stop and restart the microphone.** It opens once and stays open.

`POST /chat-voice` still exists and still works. Keep it as a fallback path so
you can migrate incrementally.

---

## Connection

```
ws://<backend-host>:8000/ws/voice?session_id=<session_id>
```

Get `session_id` from `POST /start` exactly as you do today. Open the socket once
when the case begins, keep it open for the whole session, close it at the end.

On connect the server sends:

```json
{"type":"ready","in_sample_rate":16000,"out_sample_rate":24000}
```

Do not send audio before `ready` arrives.

### Reconnecting

If the socket drops, reconnect with the same `session_id`. Conversation history
lives in the session on the server, so nothing is lost. Use backoff (1s, 2s, 4s,
capped at 10s). Discard any half-received audio on a drop.

---

## Audio formats

These are not negotiable — the models require them.

|              | Direction        | Format                                  |
| ------------ | ---------------- | --------------------------------------- |
| Microphone   | Unity → server   | PCM16 LE, **16000 Hz**, mono            |
| Patient voice| server → Unity   | PCM16 LE, **24000 Hz**, mono            |

The rates differ. Do not assume one rate for both.

---

## Messages: Unity → server

**Binary frames** = raw mic audio, PCM16 LE 16 kHz mono. Send continuously in
**512-sample chunks (1024 bytes)**. That is exactly one Silero VAD frame — other
sizes are buffered and still work, but 512 gives the tightest endpointing.

**Text frames** = JSON control:

```json
{"type":"ptt","state":"down"}   // optional: force speech start, bypass VAD
{"type":"ptt","state":"up"}     // optional: force speech end, bypass VAD
{"type":"cancel"}               // abort the in-flight turn (barge-in)
```

`ptt` is optional. If you never send it, server-side VAD decides the boundaries
on its own. Wire it to a controller button anyway as an escape hatch for noisy
rooms — once you send `ptt down`, VAD is bypassed until the matching `ptt up`.

`cancel` stops generation immediately. The server replies with `turn_end`. Use
it if the doctor starts talking over the patient.

---

## Messages: server → Unity

**Binary frames** = patient voice audio, PCM16 LE 24 kHz mono. **Every binary
frame from the server is audio** — there is no other kind, so no header to parse.
Append straight to your playback buffer.

**Text frames** = JSON events, in this order per turn:

```json
{"type":"speech_start"}                          // doctor started talking
{"type":"speech_end"}                            // doctor stopped; pipeline running
{"type":"transcript","text":"..."}               // what the doctor said
{"type":"reply","text":"...","final":false}      // patient text, one sentence at a time
{"type":"reply","text":"...","final":true}       // last sentence of the reply
{"type":"turn_end"}                              // turn complete, mic live again
{"type":"error","detail":"..."}                  // something failed; turn aborted
```

Binary audio frames are interleaved with the `reply` messages — audio for
sentence 1 starts arriving while sentence 2 is still being generated. That
interleaving is the entire point of the design; do not buffer everything and wait
for `turn_end` to start playing.

`error` means the turn is dead. Show a message, re-enable the mic, do not expect
`turn_end`.

---

## Unity implementation notes

These are the parts that are easy to get wrong. Please follow them.

### 1. Open the microphone once and never stop it

`Microphone.Start()` has real device-open latency on Quest (100-500ms). Calling
it per utterance is a large part of what made the old version feel sluggish.

```csharp
// ONCE, at session start:
_micClip = Microphone.Start(null, true, 10, 16000);   // looping 10s ring buffer
```

Never call `Microphone.End()` until the session is over. Each frame, poll
`Microphone.GetPosition(null)`, read the samples written since your last read
with `_micClip.GetData(...)`, and send them.

**Handle ring wraparound.** The buffer loops; when the write head passes the end
it restarts at 0. If you read naively you will get garbled audio at every 10s
boundary. Read in two spans when `newPos < lastPos`.

### 2. Convert float32 → PCM16

Unity gives you `float[]` in `[-1, 1]`. The server wants little-endian `short`:

```csharp
short s = (short)(Mathf.Clamp(sample, -1f, 1f) * 32767f);
```

`System.BitConverter.GetBytes(short)` is already little-endian on ARM/x86.

### 3. Play streaming audio with a PCMReaderCallback

Do **not** create a new `AudioClip` per chunk — the seams between clips are
audible and the scheduling drifts. Create one streaming clip at startup and feed
it from a queue:

```csharp
_playbackClip = AudioClip.Create("patient", 24000, 1, 24000, true, OnAudioRead);
_audioSource.clip = _playbackClip;
_audioSource.loop = true;
_audioSource.Play();          // once, at startup — leave it running forever
```

`OnAudioRead(float[] data)` drains a `ConcurrentQueue<float[]>` that the
WebSocket thread fills. **When the queue is empty, fill `data` with zeros and
return** — never block, never wait. It runs on the audio thread; blocking there
crackles or deadlocks the whole mixer. Do not allocate, log, or lock heavily
inside it.

### 4. WebSocket library

`System.Net.WebSockets.ClientWebSocket` is in the .NET base library and works on
Quest under IL2CPP — no package needed. If you end up targeting WebGL, that class
is unavailable and you will need
[NativeWebSocket](https://github.com/endel/NativeWebSocket) instead.

Receive on a background task. `ReceiveAsync` can return a message in **multiple
partial frames** — accumulate until `result.EndOfMessage` is true before treating
it as complete. This is the single most common bug in Unity WebSocket code.

### 5. Two Android settings that will cost you hours

- **Cleartext traffic.** Android 9+ blocks unencrypted `ws://` and `http://`.
  Connecting to a LAN IP fails with no useful error. Set
  `android:usesCleartextTraffic="true"` in your `AndroidManifest.xml`.
- **Mic permission.** Add `<uses-permission android:name="android.permission.RECORD_AUDIO" />`
  and request it at runtime before `Microphone.Start`:
  `Permission.RequestUserPermission(Permission.Microphone)`. On Quest the
  microphone silently returns all zeros if you skip the runtime request.

### 6. Echo — stop sending while the patient talks

The headset mic picks up the patient's own synthesized voice through the
speakers, which would make the server's VAD trigger on the patient itself. The
server already ignores incoming audio between `speech_end` and `turn_end`, but
stop sending on your side too, for safety and bandwidth.

Resume sending on `turn_end`.

### 7. Suggested UI states

Drive the on-screen indicator from the server events, not from local guesses:

| Event          | State to show      |
| -------------- | ------------------ |
| `ready`        | Idle / listening   |
| `speech_start` | "Listening…"       |
| `speech_end`   | "Thinking…"        |
| `transcript`   | Show doctor's text |
| `reply`        | Append to subtitle |
| `turn_end`     | Back to listening  |

---

## Test order

Build it in this sequence — each step is verifiable on its own:

1. Connect, receive `ready`, log it. Nothing else.
2. Stream mic audio up. Confirm the server logs `speech_start` / `speech_end`
   when you talk and pause. (Do not implement playback yet.)
3. Log `transcript` and `reply` text. Confirm the words are right.
4. Add the streaming playback path last — it is the fiddliest piece, and by then
   everything feeding it is already proven.
