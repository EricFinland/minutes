# Minutes: design and module contract

A tiny, fully local meeting recorder for Windows. It records your mic ("Me") and the
computer's audio ("Them": everyone on the Zoom/Meet/Teams call) as two separate
streams, transcribes them live with Whisper on the CPU, saves everything to disk as it
goes, and can summarize a meeting with a local Ollama model on request. The window is
invisible to screen sharing and recording.

## Goals (in priority order)

1. **Accuracy.** Capture every word from every speaker. Never clip the start or end of
   an utterance. Never invent words during silence.
2. **Never lose data.** Transcript is persisted line by line (append-only) so a crash or
   power loss loses at most the utterance in flight.
3. **Out of the way.** Small, quiet UI. Hidden from screen capture, taskbar and Alt-Tab.
4. **Light.** Near-idle during silence. One Whisper model in memory. No polling loops
   faster than needed.
5. **Simple to get text out.** One click to copy the transcript, one click to open it.

## Non-goals

- Splitting "Them" into individual remote participants (no diarization). Me vs Them only.
- Any network call other than `http://localhost:11434` (Ollama). No cloud APIs, no telemetry.
- Calendar integration, accounts, sync, cloud anything.

## Anti-goals (failure even if it "works")

- Hallucinated text in the transcript ("Thank you for watching", repeated phrases).
- Duplicate lines when the user is on speakers and the mic hears the call (echo).
- UI that steals focus, flashes, or shows up in a screen share.
- A transcript file that is corrupt or empty after a crash.

## Environment

- Windows 11, Python **3.14** at `C:\Python314\python.exe` (use this interpreter).
- CPU: AMD Ryzen AI 9 HX 370 (12 cores / 24 threads), 31 GB RAM, **no CUDA**.
- Installed: faster-whisper 1.2.1, ctranslate2 4.7.1, numpy 2.3.4, onnxruntime 1.24.2,
  sounddevice 0.5.5, SoundCard 0.4.6, PyQt6 6.10, requests, pywin32, psutil, pytest.
  NOT installed: soundfile, keyboard. Do not add heavy deps; stdlib `wave` is fine for WAV.
- Ollama running on `localhost:11434` with `llama3:latest` (8B) pulled.
- Silero VAD v6 ONNX ships with faster-whisper: `faster_whisper/assets/silero_vad_v6.onnx`
  (`faster_whisper.vad` has `SileroVADModel`, `get_speech_timestamps`, `VadOptions`).
- Reference code from the user's older app (read for patterns, do not import from it):
  `C:\Users\ericc\OneDrive\Desktop\RandomProjects\InterviewBot\meeting_copilot\`
  - `audio/audio_capture.py`: WASAPI loopback via `soundcard`. On the capture thread you
    MUST call `pythoncom.CoInitializeEx(0)` (or ctypes `ole32.CoInitializeEx(None, 0)`)
    or you get error 0x800401f0.
  - `ui/stealth_window.py`: `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE=0x11)`
    plus `WS_EX_TOOLWINDOW`, applied in `showEvent`.
  - `ui/overlay_window.py`: `_GlobalHotkey` (Win32 `RegisterHotKey` on a daemon thread).
  - `run.pyw`: hidden-console fd fix for ctranslate2 under `pythonw`.
- SoundCard 0.4.6 + numpy 2: `soundcard` may call `numpy.fromstring` (removed for binary
  data in numpy 2). If it breaks, monkeypatch as the reference app does, or check how
  `meeting_copilot` handles it.

## Repository layout

```
run.pyw                  # windowless launcher (B)
app.py                   # entry point: builds Recorder + UI, wires them (B)
requirements.txt
README.md                # (C)
bench.py                 # model speed/accuracy benchmark (A)
docs/DESIGN.md           # this file
minutes/
  __init__.py
  config.py              # settings + paths (shared, already written)
  models.py              # Line, Event, file formats (shared, already written)
  engine/                # (A) capture, VAD, transcription, session files
    __init__.py          # exports Recorder, get_model
    audio.py
    vad.py
    transcriber.py
    session.py
    recorder.py
    model.py             # get_model(cfg) -> shared WhisperModel singleton
  ui/                    # (B) overlay, stealth, hotkey, startup toggle
  post/                  # (C) summary, polish pass, export, cleanup
    __init__.py
    summarizer.py
    polish.py
    export.py
tests/
```

Each agent owns only its own directory (plus the top-level files listed for it). Do not
edit `minutes/config.py` or `minutes/models.py`; if you truly need a change there, say so
in your final report instead.

## On-disk format (contract between A, B and C)

Data root: `config.data_dir()` = `C:\Users\ericc\Minutes` (NOT OneDrive). Each meeting is
a folder `meetings/YYYY-MM-DD_HHMMSS/` containing:

| File | Writer | Format |
|---|---|---|
| `lines.jsonl` | engine | One JSON object per finalized line, append-only, flushed + fsync'd per line: `{"speaker","start","end","text"}`. Source of truth. |
| `transcript.txt` | engine | Human-readable, regenerated atomically (write tmp, `os.replace`) after each line via `models.format_transcript`. Lines sorted by `start`. |
| `meta.json` | engine | `{"started": iso8601, "ended": iso8601 or null, "model": str, "audio_spans": [...]}` |
| `audio_me.wav`, `audio_them.wav` | engine | 16 kHz mono int16 WAV, written continuously while recording (silence included). Header must be valid even after a crash (rewrite header periodically, e.g. every 5 s). |
| `summary.md` | post | Markdown summary. |
| `transcript_polished.txt`, `lines_polished.jsonl` | post | Same formats as above, from the polish pass. |

`audio_spans`: list of `{"meeting_t": float, "file_t": float, "dur": float}`. Each time
recording resumes inside the same meeting, a new span is appended: `meeting_t` = seconds
since meeting start (wall clock), `file_t` = seconds into the WAV files where that span
begins. Both WAVs always share the same `file_t` timeline. `dur` is updated on stop/flush.
Line `start`/`end` are always in **meeting time** (seconds since `meta.started`).

Use helpers in `minutes/models.py` for reading/writing lines and transcripts.

## Engine API (A implements, B and C consume)

```python
from minutes.engine import Recorder, get_model

rec = Recorder(cfg: dict, listener: Callable[[Event], None])
rec.start()          # begin or resume recording. Non-blocking. Loads model in background
                     # on first call (emit status events while loading). Opens a new
                     # meeting if none is open.
rec.stop()           # pause capture. Flushes in-flight speech (transcribes it). Non-blocking.
rec.new_meeting()    # finalize current meeting (if any) and open a new one. Keeps the
                     # recording state as it was.
rec.shutdown(timeout=8.0)  # stop capture, finish pending transcription, finalize meta,
                     # join threads. Blocking. Safe to call twice.
rec.recording -> bool
rec.meeting_dir -> Path | None
rec.lines() -> list[Line]  # copy of current meeting's lines, sorted by start

get_model(cfg) -> faster_whisper.WhisperModel  # process-wide singleton, thread-safe init
```

`listener` is called from engine threads (never assume GUI thread). Events are
`minutes.models.Event` with `kind` one of:

- `"line"`: `line=Line`, `meeting_dir=Path`. A finalized transcript line.
- `"state"`: `recording=bool`.
- `"meeting"`: `meeting_dir=Path or None`. A meeting was opened (Path) or closed (None).
- `"status"`: `text=str`, `level="info"|"warn"|"error"`. E.g. "Loading speech model...",
  "Listening", "Mic unavailable", backlog warnings.
- `"activity"`: `speaker="Me"|"Them"`, `active=bool`. Speech started/stopped on a stream
  (lets the UI show who is talking). Emit only on transitions, never per audio frame.

Meeting auto-split: if neither stream has had speech for `cfg["meeting_gap_minutes"]`
minutes, close the current meeting (emit `meeting` None). The next detected speech opens
a new one. Recording itself stays on (this is the "always recording" mode).

### Engine design requirements

- Two capture sources at 16 kHz mono float32: mic via `sounddevice` (default input, or
  `cfg["mic_device"]`), system audio via `soundcard` WASAPI loopback of the default output
  (or `cfg["speaker_device"]`). If the default output device changes mid-session
  (e.g. AirPods connect), detect it (check every few seconds) and re-open loopback.
- Streaming VAD per stream with Silero (`silero_vad_v6.onnx` via onnxruntime, 1 thread).
  Sensitive threshold (`cfg["vad_threshold"]`, default 0.35), **pre-roll padding** of at
  least 400 ms and post-roll of at least 300 ms so first and last words are never clipped.
  An utterance ends after `cfg["end_silence_ms"]` (default 700) of silence. Utterances
  longer than `cfg["max_utterance_s"]` (default 25) are cut at the lowest-energy point
  in the last few seconds, not mid-word.
- One shared `WhisperModel` (`cfg["whisper_model"]`, `compute_type="int8"`, `device="cpu"`,
  `cpu_threads` from cfg, `num_workers=2`) so Me and Them can transcribe concurrently.
  Use `beam_size=5`, `language=cfg["language"]`, `vad_filter=False` (we already VAD'd),
  `condition_on_previous_text=False`, and pass the last ~200 chars of that stream's text as
  `initial_prompt` for continuity. `word_timestamps` not needed.
- Hallucination guard: drop segments with `no_speech_prob > 0.6 and avg_logprob < -1.0`,
  drop `compression_ratio > 2.4`, drop known filler hallucinations ("Thank you.",
  "Thanks for watching!", "Subtitles by...", etc.) when the utterance RMS is low.
- Echo dedupe: if a "Me" line overlaps in time with a "Them" line and the normalized text
  similarity (difflib ratio) is >= 0.6, drop the "Me" line (mic heard the speakers).
  Hold "Me" lines briefly (up to ~3 s) to allow the matching "Them" line to arrive.
- Back-pressure: if the transcription queue backlog exceeds 60 s of audio, emit a
  `status` warn. Never drop audio.
- Idle cost: during silence only VAD runs (cheap). No busy loops; block on queues.
- Robustness: if mic or loopback fails to open, keep the other one running and emit a
  `status` error. Retry the failed source every 10 s.

## UI (B)

Frameless PyQt6 window, always on top, never steals focus (`WA_ShowWithoutActivating`,
`Qt.WindowDoesNotAcceptFocus` only when collapsed), excluded from capture, not in taskbar
or Alt-Tab.

- **Collapsed pill** (about 220 x 36): recording dot (red when recording, grey when
  stopped; a subtle pulse is fine but cheap), elapsed meeting time, a tiny "Me"/"Them"
  talking indicator, record/stop button, expand button. Draggable. Position remembered.
- **Expanded panel** (about 380 x 460, resizable): live transcript (Me lines visually
  distinct from Them lines; auto-scroll only if already at bottom), and a bottom bar:
  **Copy** (whole transcript to clipboard), **Open** (opens transcript.txt), **Summarize**
  (runs `post.summarize` in a background thread, shows progress, then opens summary.md),
  **New meeting**. A small "..." menu: Open meetings folder, Polish transcript,
  Previous meetings (opens folder), Start with Windows (checkable), Quit.
- Global hotkey `cfg["hotkey_toggle"]` (default `ctrl+alt+m`) shows/hides the window.
- Dark, quiet styling, readable 12-13 px text, rounded corners, ~92% opacity.
- All engine events arrive via a Qt signal (marshal from engine threads).
- Closing the window hides it; Quit calls `rec.shutdown()` then exits.
- Recording starts automatically on launch if `cfg["record_on_launch"]`.
- Single instance: a second launch should just show the existing window (named mutex or
  local socket) and exit.

## Post (C)

```python
from minutes.post import summarize, polish, transcript_text, list_meetings, cleanup_old_audio

summarize(meeting_dir, cfg, on_progress=None) -> Path      # writes summary.md
polish(meeting_dir, cfg, on_progress=None) -> Path         # writes transcript_polished.txt
transcript_text(meeting_dir) -> str                        # best available transcript text
list_meetings(cfg) -> list[Path]                           # newest first
cleanup_old_audio(cfg) -> int                              # deletes WAVs older than
                                                           # cfg["keep_audio_days"]; returns count
```

- Summarizer: Ollama `/api/chat` at `cfg["ollama_url"]` with `cfg["summary_model"]`.
  Long transcripts: chunk to fit an 8k context (map: summarize chunks, reduce: merge).
  Output Markdown with sections: Summary (3-5 sentences), Key points, Decisions,
  Action items (with owner Me/Them if clear), Open questions. Stick to what was said; no
  invented facts. `on_progress(str)` for status. Clear error if Ollama is not running
  (try to start `ollama serve` once, hidden, then retry).
- Polish: re-transcribe `audio_me.wav` and `audio_them.wav` with `get_model(cfg)`, full
  context (`condition_on_previous_text=True`, `vad_filter=True` with padded speech
  params, `beam_size=5`), map file time back to meeting time via `audio_spans`, apply the
  same hallucination guard and echo dedupe, write `lines_polished.jsonl` and
  `transcript_polished.txt`. Lower its own thread priority so live recording is not
  starved.
- `transcript_text`: polished if present, else live transcript.

## Style rules (all agents)

- No em-dash characters anywhere in files (code, comments, strings, docs). Use periods,
  commas, colons or parentheses.
- No mention of AI assistants or code generators in code, comments or docs.
- Plain, readable code; comments only where they explain a non-obvious why.
- Do not run git commit. Do not create files outside the project folder (except the
  data dir at runtime and the HF model cache).
