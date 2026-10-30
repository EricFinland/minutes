"""Stand-ins for the engine and post modules, for UI development and tests.

FakeRecorder follows the Recorder contract in docs/DESIGN.md: it emits events from a
background thread and writes real meeting files (lines.jsonl, transcript.txt,
meta.json) so Copy, Open and the post stubs behave like the real thing.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from minutes import config
from minutes.models import (
    ME, META_FILE, POLISHED_LINES_FILE, POLISHED_TRANSCRIPT_FILE, SUMMARY_FILE, THEM,
    TRANSCRIPT_FILE, LINES_FILE, Event, Line, append_line, event, format_transcript,
    meeting_title, read_lines, read_meta, write_meta, write_text_atomic,
)

SCRIPT: list[tuple[str, str]] = [
    (THEM, "Okay, looks like everyone's here. Should we get started?"),
    (ME, "Yeah, let's do it. I put the agenda in the chat."),
    (THEM, "Great. First thing is the launch date for the mobile release."),
    (ME, "Right now we're tracking for the fourteenth, but QA still has two open blockers."),
    (THEM, "What are the blockers exactly?"),
    (ME, "One is the login crash on older Android devices, the other is a payments timeout."),
    (THEM, "Priya's team can take the payments one. I'll ask her after this call."),
    (ME, "Perfect. I'll keep working on the Android crash and post an update tomorrow."),
    (THEM, "Do we need to move the date, or is it too early to say?"),
    (ME, "Let's decide on Thursday once we know if the fixes landed."),
    (THEM, "Works for me. Next item is the onboarding redesign."),
    (ME, "Design shared the new flows yesterday. I think the second option tested better."),
    (THEM, "Agreed, the drop-off was way lower in the user sessions."),
    (ME, "Then let's go with option two and get engineering estimates by Friday."),
    (THEM, "I'll set up a meeting with design and engineering for Wednesday."),
    (ME, "Thanks. Anything else before we wrap?"),
    (THEM, "Just a reminder that the quarterly review deck is due next Monday."),
    (ME, "Got it. I'll send my slides over by Friday afternoon."),
]


def _new_meeting_dir() -> Path:
    base = config.meetings_dir()
    name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d = base / name
    n = 2
    while d.exists():
        d = base / f"{name}_{n}"
        n += 1
    d.mkdir(parents=True)
    return d


class FakeRecorder:
    def __init__(self, cfg: dict, listener: Callable[[Event], None], *,
                 line_interval: float = 3.0, load_delay: float = 1.5,
                 script: Optional[list[tuple[str, str]]] = None):
        self.cfg = cfg
        self._listener = listener
        self._interval = line_interval
        self._load_delay = load_delay
        self._script = script or SCRIPT
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._quit = threading.Event()
        self._recording = False
        self._loaded = False
        self._meeting_dir: Optional[Path] = None
        self._started = 0.0
        self._lines: list[Line] = []
        self._i = 0
        self._thread: Optional[threading.Thread] = None
        self.shutdown_calls = 0

    # Recorder API ---------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def meeting_dir(self) -> Optional[Path]:
        return self._meeting_dir

    def lines(self) -> list[Line]:
        with self._lock:
            return sorted(self._lines, key=lambda l: l.start)

    def start(self) -> None:
        if self._quit.is_set():
            return
        with self._lock:
            if self._recording:
                return
            self._recording = True
            if self._meeting_dir is None:
                self._open_meeting()
        self._emit(event("state", recording=True))
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="fake-recorder", daemon=True)
            self._thread.start()
        self._wake.set()

    def stop(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
        self._emit(event("activity", speaker=ME, active=False))
        self._emit(event("activity", speaker=THEM, active=False))
        self._emit(event("state", recording=False))
        self._emit(event("status", text="Paused", level="info"))
        self._wake.set()

    def new_meeting(self) -> None:
        with self._lock:
            self._close_meeting()
            self._open_meeting()

    def shutdown(self, timeout: float = 8.0) -> None:
        self.shutdown_calls += 1
        self._quit.set()
        self._wake.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        with self._lock:
            self._recording = False
            self._close_meeting()

    # internals ------------------------------------------------------------
    def _emit(self, ev: Event) -> None:
        try:
            self._listener(ev)
        except Exception:
            pass

    def _open_meeting(self) -> None:
        d = _new_meeting_dir()
        self._meeting_dir = d
        self._started = time.time()
        self._lines = []
        write_meta(d, {"started": datetime.now().isoformat(timespec="seconds"),
                       "ended": None, "model": "fake", "audio_spans": []})
        write_text_atomic(d / TRANSCRIPT_FILE, format_transcript([], meeting_title(d)))
        self._emit(event("meeting", meeting_dir=d))

    def _close_meeting(self) -> None:
        d = self._meeting_dir
        if d is None:
            return
        meta = read_meta(d)
        meta["ended"] = datetime.now().isoformat(timespec="seconds")
        write_meta(d, meta)
        self._meeting_dir = None
        self._emit(event("meeting", meeting_dir=None))

    def _sleep(self, seconds: float) -> bool:
        """Wait, returning False if shutting down."""
        return not self._quit.wait(seconds)

    def _run(self) -> None:
        if not self._loaded:
            self._emit(event("status", text="Loading speech model...", level="info"))
            if not self._sleep(self._load_delay):
                return
            self._loaded = True
        self._emit(event("status", text="Listening", level="info"))
        while not self._quit.is_set():
            if not self._recording:
                self._wake.wait()
                self._wake.clear()
                if self._recording and not self._quit.is_set():
                    self._emit(event("status", text="Listening", level="info"))
                continue
            speaker, text = self._script[self._i % len(self._script)]
            self._i += 1
            t0 = time.time()
            self._emit(event("activity", speaker=speaker, active=True))
            if not self._sleep(self._interval * 0.6):
                return
            self._emit(event("activity", speaker=speaker, active=False))
            self._add_line(speaker, text, t0, time.time())
            if not self._sleep(self._interval * 0.4):
                return

    def _add_line(self, speaker: str, text: str, t0: float, t1: float) -> None:
        with self._lock:
            d = self._meeting_dir
            if d is None:
                self._open_meeting()
                d = self._meeting_dir
            line = Line(speaker, round(t0 - self._started, 2), round(t1 - self._started, 2), text)
            self._lines.append(line)
            append_line(d / LINES_FILE, line)
            write_text_atomic(d / TRANSCRIPT_FILE, format_transcript(self._lines, meeting_title(d)))
        self._emit(event("line", line=line, meeting_dir=d))


class FakePost:
    """Mimics minutes.post with quick, local, deterministic results."""

    def __init__(self, delay: float = 0.6, fail: Optional[str] = None):
        self.delay = delay
        self.fail = fail

    def _steps(self, on_progress, steps: list[str]) -> None:
        for s in steps:
            if on_progress:
                on_progress(s)
            time.sleep(self.delay / max(1, len(steps)))
        if self.fail:
            raise RuntimeError(self.fail)

    def summarize(self, meeting_dir: Path, cfg: dict, on_progress=None) -> Path:
        meeting_dir = Path(meeting_dir)
        self._steps(on_progress, ["Reading transcript", "Summarizing part 1 of 1", "Writing summary"])
        lines = read_lines(meeting_dir / LINES_FILE)
        points = "\n".join(f"- {ln.speaker}: {ln.text}" for ln in lines[:6]) or "- (nothing said yet)"
        body = (f"# {meeting_title(meeting_dir)}\n\n## Summary\n\nPlaceholder summary from the "
                f"development stub ({len(lines)} lines).\n\n## Key points\n\n{points}\n\n"
                "## Decisions\n\n- None recorded\n\n## Action items\n\n- None recorded\n\n"
                "## Open questions\n\n- None recorded\n")
        path = meeting_dir / SUMMARY_FILE
        write_text_atomic(path, body)
        return path

    def polish(self, meeting_dir: Path, cfg: dict, on_progress=None) -> Path:
        meeting_dir = Path(meeting_dir)
        self._steps(on_progress, ["Re-transcribing Me", "Re-transcribing Them", "Merging"])
        lines = read_lines(meeting_dir / LINES_FILE)
        with open(meeting_dir / POLISHED_LINES_FILE, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(json.dumps(ln.__dict__, ensure_ascii=False) + "\n")
        path = meeting_dir / POLISHED_TRANSCRIPT_FILE
        write_text_atomic(path, format_transcript(lines, meeting_title(meeting_dir)))
        return path

    def transcript_text(self, meeting_dir: Path) -> str:
        meeting_dir = Path(meeting_dir)
        for name in (POLISHED_TRANSCRIPT_FILE, TRANSCRIPT_FILE):
            p = meeting_dir / name
            if p.exists():
                return p.read_text(encoding="utf-8")
        return ""

    def list_meetings(self, cfg: dict) -> list[Path]:
        root = config.meetings_dir()
        dirs = [p for p in root.iterdir() if p.is_dir() and (p / META_FILE).exists()]
        return sorted(dirs, key=lambda p: p.name, reverse=True)

    def cleanup_old_audio(self, cfg: dict) -> int:
        return 0


class UnavailablePost(FakePost):
    """Used when minutes.post cannot be imported: reading still works, tools explain why not."""

    def __init__(self, reason: str):
        super().__init__(delay=0)
        self.reason = reason

    def summarize(self, meeting_dir, cfg, on_progress=None):
        raise RuntimeError(self.reason)

    def polish(self, meeting_dir, cfg, on_progress=None):
        raise RuntimeError(self.reason)


class UnavailableRecorder:
    """Used when minutes.engine cannot be imported, so the window still opens and explains."""

    def __init__(self, cfg: dict, listener: Callable[[Event], None], reason: str = ""):
        self._listener = listener
        self.reason = reason or "Speech engine is not available"
        self.recording = False
        self.meeting_dir = None

    def _say(self) -> None:
        self._listener(event("status", text=self.reason, level="error"))

    def start(self) -> None:
        self._say()

    def stop(self) -> None:
        pass

    def new_meeting(self) -> None:
        self._say()

    def shutdown(self, timeout: float = 8.0) -> None:
        pass

    def lines(self) -> list[Line]:
        return []

