"""Recorder: capture, VAD, transcription and meeting files, wired together.

Threads (all daemons):
  control           start/stop/new meeting/shutdown, periodic WAV header + meta sync
  events            calls the listener, so engine locks are never held during callbacks
  model loader      loads Whisper in the background on first start
  per stream (Me, Them):
    source runner   opens the device, retries every 10 s, reopens if it dies
    worker          writes the WAV, runs the VAD, queues utterances
    transcriber     turns utterances into lines

Time: every stream sample has an absolute index on one shared 16 kHz timeline that starts
when the Recorder is created. Real-time sources are aligned to the wall clock (a gap in
delivery becomes silence), so equal indices on Me and Them mean the same moment. Meeting
time is (index - meeting origin) / 16000.
"""
from __future__ import annotations

import difflib
import queue
import re
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from minutes import config
from minutes.models import ME, THEM, Event, Line, event

from .audio import AudioSource, default_sources
from .model import get_model
from .session import Session
from .transcriber import Item, Transcriber, pack
from .vad import StreamingVAD, Utterance

SR = 16000
GAP_RESYNC_S = 0.5        # real-time delivery gaps longer than this become silence
ROLLING_S = 5.0           # audio kept while no meeting is open (pre-roll for the next one)
SOURCE_RETRY_S = 10.0
SOURCE_CHECK_S = 1.0
BACKLOG_WARN_S = 60.0
BACKLOG_OK_S = 15.0
ECHO_SIMILARITY = 0.6
ECHO_TOLERANCE_S = 1.0    # VAD edges differ between a voice and its quieter echo
ECHO_MAX_HOLD_S = 8.0
RECENT_THEM_S = 120.0
MODEL_RETRY_S = 30.0
LABELS = {ME: "Mic", THEM: "System audio"}


@dataclass(eq=False)
class _Job:
    speaker: str
    session: Session
    origin: int
    utt: Utterance


@dataclass(eq=False)
class _Held:
    line: Line
    session: Session
    a0: int
    a1: int
    deadline: float


@dataclass
class _Stream:
    speaker: str
    factory: Callable[[], AudioSource]
    vad: StreamingVAD
    q: queue.Queue = field(default_factory=queue.Queue)
    tq: queue.Queue = field(default_factory=queue.Queue)
    pos: int = 0                      # next sample index (worker thread only)
    # Snapshots below are written under the recorder lock.
    done_pos: int = 0                 # everything before this has been through the VAD
    open_utt: int | None = None       # start of the utterance the VAD is inside, if any
    pending: list = field(default_factory=list)   # (a0, a1, samples) queued or in progress
    connected: bool = False
    active: bool = False
    rolling: deque = field(default_factory=deque)
    rolling_n: int = 0
    stop_source: threading.Event | None = None
    runner: threading.Thread | None = None
    transcriber: Transcriber | None = None
    last_session: Session | None = None


def _plain_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def echo_similarity(me_text: str, them_texts: list[str]) -> float:
    """How strongly a Me line looks like the mic picking up the Them lines."""
    me = _plain_words(me_text)
    if not me or not them_texts:
        return 0.0
    me_s = " ".join(me)
    best = 0.0
    joined: list[str] = []
    for t in them_texts:
        words = _plain_words(t)
        joined += words
        best = max(best, difflib.SequenceMatcher(None, me_s, " ".join(words)).ratio())
    best = max(best, difflib.SequenceMatcher(None, me_s, " ".join(joined)).ratio())
    if len(me) >= 3 and joined:
        # The mic may catch only part of a long remote sentence: count Me words found in
        # order inside the remote text.
        sm = difflib.SequenceMatcher(None, me, joined, autojunk=False)
        matched = sum(b.size for b in sm.get_matching_blocks())
        best = max(best, matched / len(me))
    return best


class Recorder:
    def __init__(self, cfg: dict, listener: Callable[[Event], None],
                 sources: dict[str, Callable[[], AudioSource]] | None = None) -> None:
        """`sources` maps "Me"/"Them" to factories returning AudioSource objects; the
        default is the system microphone and WASAPI loopback of the default output."""
        self.cfg = dict(cfg)
        self._listener = listener
        factories = sources or default_sources(self.cfg)
        self._lock = threading.RLock()
        self._t0 = time.monotonic()
        self._want_recording = False
        self._capturing = False
        self._closed = False
        self._session: Session | None = None
        self._origin = 0                  # timeline index of meeting time 0
        self._span_abs: int | None = None # timeline index where the current span starts
        self._span_file = 0               # WAV sample index where the current span starts
        self._recent: list[tuple[Session, int]] = []
        self._last_speech = 0
        self._held: list[_Held] = []
        self._recent_them: deque = deque()   # (a0, a1, text)
        self._backlog_warned = False
        self._model_ready = threading.Event()
        self._loader: threading.Thread | None = None
        self._gap = int(float(self.cfg.get("meeting_gap_minutes", 15)) * 60 * SR)
        self._sync_s = float(self.cfg.get("wav_sync_s", 5.0))

        self._streams = {spk: _Stream(spk, factories[spk], StreamingVAD(self.cfg))
                         for spk in (ME, THEM)}
        self._events: queue.Queue = queue.Queue()
        self._cmds: queue.Queue = queue.Queue()
        self._threads = [
            threading.Thread(target=self._event_loop, name="minutes-events", daemon=True),
            threading.Thread(target=self._control_loop, name="minutes-control", daemon=True),
        ]
        for s in self._streams.values():
            self._threads.append(threading.Thread(target=self._worker_loop, args=(s,),
                                                  name=f"minutes-vad-{s.speaker}", daemon=True))
            self._threads.append(threading.Thread(target=self._transcribe_loop, args=(s,),
                                                  name=f"minutes-stt-{s.speaker}", daemon=True))
        for t in self._threads:
            t.start()

    # ================================================================ public API

    def start(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._want_recording = True
        self._cmds.put(("start", None))

    def stop(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._want_recording = False
        self._cmds.put(("stop", None))

    def new_meeting(self) -> None:
        if not self._closed:
            self._cmds.put(("new_meeting", None))

    def shutdown(self, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        with self._lock:
            first = not self._closed
            self._closed = True
            self._want_recording = False
        if first:
            done = threading.Event()
            self._cmds.put(("shutdown", (done, deadline)))
            done.wait(max(0.0, deadline - time.monotonic()))
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(max(0.05, deadline - time.monotonic()))

    @property
    def recording(self) -> bool:
        return self._want_recording

    @property
    def meeting_dir(self) -> Path | None:
        with self._lock:
            return self._session.dir if self._session else None

    def lines(self) -> list[Line]:
        with self._lock:
            session = self._session
        return session.lines() if session else []

    # ================================================================ events

    def _emit(self, kind: str, **data) -> None:
        self._events.put(event(kind, **data))

    def _status(self, text: str, level: str = "info") -> None:
        self._emit("status", text=text, level=level)

    def _event_loop(self) -> None:
        while True:
            ev = self._events.get()
            if ev is None:
                return
            try:
                self._listener(ev)
            except Exception:
                pass

    # ================================================================ control thread

    def _control_loop(self) -> None:
        next_sync = None
        while True:
            timeout = None
            now = time.monotonic()
            if next_sync is not None:
                timeout = max(0.0, next_sync - now)
            with self._lock:
                if self._held:
                    soonest = min(h.deadline for h in self._held) - now
                    timeout = max(0.05, min(timeout if timeout is not None else soonest, soonest))
            try:
                cmd, arg = self._cmds.get(timeout=timeout)
            except queue.Empty:
                cmd, arg = "tick", None
            if cmd == "start":
                self._do_start()
            elif cmd == "stop":
                self._do_stop()
            elif cmd == "new_meeting":
                self._do_new_meeting()
            elif cmd == "shutdown":
                self._do_shutdown(*arg)
                return
            with self._lock:
                self._release_held()
                if self._session is not None and self._capturing:
                    now = time.monotonic()
                    if next_sync is None or now >= next_sync:
                        self._sync_session()
                        next_sync = now + self._sync_s
                else:
                    next_sync = None

    def _now_abs(self) -> int:
        wall = int((time.monotonic() - self._t0) * SR)
        return max(wall, max(s.pos for s in self._streams.values()))

    def _do_start(self) -> None:
        with self._lock:
            if self._capturing or not self._want_recording:
                return
            now = self._now_abs()
            for s in self._streams.values():
                s.q.put(("reset", now))
            if self._session is None:
                self._open_session(now, datetime.now().astimezone())
            else:
                self._begin_span(now)
            self._last_speech = now
            self._capturing = True
        self._ensure_model()
        for s in self._streams.values():
            s.stop_source = threading.Event()
            s.runner = threading.Thread(target=self._source_loop, args=(s, s.stop_source),
                                        name=f"minutes-src-{s.speaker}", daemon=True)
            s.runner.start()
        self._emit("state", recording=True)
        if self._model_ready.is_set():
            self._status("Listening")

    def _do_stop(self, emit_state: bool = True) -> None:
        if not self._capturing:
            return
        for s in self._streams.values():
            s.stop_source.set()
        for s in self._streams.values():
            s.runner.join(timeout=5.0)
        flushed = []
        for s in self._streams.values():
            ev = threading.Event()
            s.q.put(("flush", ev))
            flushed.append(ev)
        for ev in flushed:
            ev.wait(10.0)
        with self._lock:
            self._capturing = False
            self._end_span()
        if emit_state:
            self._emit("state", recording=False)
            self._status("Paused")

    def _do_new_meeting(self) -> None:
        with self._lock:
            self._close_session()
            if self._capturing:
                now = self._now_abs()
                self._open_session(now, datetime.now().astimezone())
                self._last_speech = now

    def _do_shutdown(self, done: threading.Event, deadline: float) -> None:
        try:
            self._do_stop(emit_state=False)
            # Let queued utterances finish while there is time; audio is safe on disk anyway.
            while time.monotonic() < deadline - 0.5:
                with self._lock:
                    busy = any(s.pending for s in self._streams.values())
                if not busy:
                    break
                time.sleep(0.05)
            with self._lock:
                self._release_held(force=True)
                self._close_session()
            self._emit("state", recording=False)
        finally:
            for s in self._streams.values():
                s.q.put(("quit", None))
                s.tq.put(None)
            self._events.put(None)
            done.set()

    # ================================================================ sessions (lock held)

    def _open_session(self, origin: int, started: datetime) -> None:
        self._session = Session(config.meetings_dir(), started, str(self.cfg.get("whisper_model")))
        self._origin = origin
        self._recent.append((self._session, origin))
        del self._recent[:-4]
        self._span_abs = None
        self._begin_span(origin)
        self._emit("meeting", meeting_dir=self._session.dir)

    def _begin_span(self, at: int) -> None:
        self._span_file = self._session.begin_span((at - self._origin) / SR)
        self._span_abs = at
        for s in self._streams.values():
            self._replay_rolling(s)

    def _end_span(self) -> None:
        if self._session is None or self._span_abs is None:
            return
        end = max(s.pos for s in self._streams.values())
        self._session.set_span_duration((end - self._span_abs) / SR)
        self._session.align_audio()
        self._session.sync()
        self._span_abs = None

    def _close_session(self) -> None:
        if self._session is None:
            return
        self._end_span()
        self._session.close()
        if not self._session.had_speech:
            # Nobody spoke: the folder would only hold silence. Do not leave clutter.
            shutil.rmtree(self._session.dir, ignore_errors=True)
        self._session = None
        for s in self._streams.values():
            s.rolling.clear()
            s.rolling_n = 0
        self._emit("meeting", meeting_dir=None)

    def _sync_session(self) -> None:
        if self._span_abs is not None:
            self._session.set_span_duration((self._now_abs() - self._span_abs) / SR)
        self._session.sync()

    def _replay_rolling(self, s: _Stream) -> None:
        for a, samples in s.rolling:
            self._session.write_audio(s.speaker, samples, self._span_file + a - self._span_abs)
        s.rolling.clear()
        s.rolling_n = 0

    def _session_for(self, a: int) -> tuple[Session | None, int]:
        best = None
        for session, origin in self._recent:
            if origin <= a + SR // 10:
                best = (session, origin)
        if best is None and self._recent:
            best = self._recent[0]
        return best if best else (None, 0)

    # ================================================================ sources

    def _source_loop(self, s: _Stream, stop: threading.Event) -> None:
        label = LABELS[s.speaker]
        reported = False
        while not stop.is_set():
            # Marked connected before open: a fast source may deliver everything at once.
            with self._lock:
                s.connected = True
            try:
                src = s.factory()
                src.open(lambda x, t: s.q.put(("audio", x, t)), self._status)
            except Exception as exc:
                with self._lock:
                    s.connected = False
                if not reported:
                    self._status(f"{label} unavailable ({exc}). Retrying every "
                                 f"{int(SOURCE_RETRY_S)} s.", "error")
                    reported = True
                stop.wait(SOURCE_RETRY_S)
                continue
            if reported:
                self._status(f"{label} connected: {src.name}")
                reported = False
            while not stop.wait(SOURCE_CHECK_S):
                if not src.alive():
                    break
            src.close()
            with self._lock:
                s.connected = False
            if not stop.is_set():
                self._status(f"{label} lost, reconnecting", "warn")
                reported = True

    # ================================================================ VAD worker

    def _worker_loop(self, s: _Stream) -> None:
        while True:
            item = s.q.get()
            kind = item[0]
            if kind == "audio":
                self._process_audio(s, item[1], item[2])
            elif kind == "reset":
                self._handle_vad(s, s.vad.flush())
                s.pos = item[1]
                s.vad.reset(s.pos)
            elif kind == "flush":
                self._handle_vad(s, s.vad.flush())
                item[1].set()
            elif kind == "quit":
                return

    def _process_audio(self, s: _Stream, samples: np.ndarray, t: float | None) -> None:
        if t is not None:
            expected = int((t - self._t0) * SR) - samples.size
            if expected - s.pos > GAP_RESYNC_S * SR:
                self._handle_vad(s, s.vad.flush())
                s.pos = expected
                s.vad.reset(expected)
        self._write_audio(s, s.pos, samples)
        s.pos += samples.size
        self._handle_vad(s, s.vad.feed(samples))

    def _write_audio(self, s: _Stream, a: int, samples: np.ndarray) -> None:
        with self._lock:
            if self._session is not None and self._span_abs is not None:
                self._session.write_audio(s.speaker, samples, self._span_file + a - self._span_abs)
            elif self._capturing:
                s.rolling.append((a, samples))
                s.rolling_n += samples.size
                while s.rolling and s.rolling_n - s.rolling[0][1].size >= ROLLING_S * SR:
                    s.rolling_n -= s.rolling.popleft()[1].size

    def _handle_vad(self, s: _Stream, events: list) -> None:
        with self._lock:
            for kind, val in events:
                if kind == "start":
                    if self._session is None and self._capturing:
                        started = datetime.now().astimezone() - timedelta(seconds=(s.pos - val) / SR)
                        self._open_session(val, started)
                    self._last_speech = max(self._last_speech, val)
                    s.active = True
                    self._emit("activity", speaker=s.speaker, active=True)
                elif kind == "end":
                    self._last_speech = max(self._last_speech, val)
                    if s.active:
                        s.active = False
                        self._emit("activity", speaker=s.speaker, active=False)
                elif kind == "utterance":
                    self._queue_utterance(s, val)
            s.done_pos = s.vad.pos
            s.open_utt = s.vad.speech_start
            if s.open_utt is not None:
                self._last_speech = max(self._last_speech, s.vad.pos)
            if self._held:
                self._release_held()
            self._check_meeting_gap()

    def _check_meeting_gap(self) -> None:
        if self._session is None or not self._capturing or self._gap <= 0:
            return
        # A stream still holding queued audio counts even if its device was just closed.
        live = [s for s in self._streams.values() if s.connected or s.q.qsize()]
        if not live or any(s.open_utt is not None for s in self._streams.values()):
            return
        if min(s.done_pos for s in live) - self._last_speech >= self._gap:
            self._close_session()

    def _queue_utterance(self, s: _Stream, u: Utterance) -> None:
        session, origin = self._session_for(u.start)
        if session is None:
            return
        session.had_speech = True
        s.pending.append((u.start, u.end, u.audio.size))
        s.tq.put(_Job(s.speaker, session, origin, u))
        backlog = sum(n for st in self._streams.values() for _, _, n in st.pending) / SR
        if backlog > BACKLOG_WARN_S and not self._backlog_warned:
            self._backlog_warned = True
            self._status(f"Transcription is {int(backlog)} s behind; audio is saved and "
                         "will be transcribed", "warn")

    # ================================================================ transcription

    def _ensure_model(self) -> None:
        if self._model_ready.is_set() or (self._loader and self._loader.is_alive()):
            return
        self._loader = threading.Thread(target=self._load_model, name="minutes-model", daemon=True)
        self._loader.start()

    def _load_model(self) -> None:
        name = self.cfg.get("whisper_model")
        while not self._closed:
            self._status(f"Loading speech model ({name})...")
            try:
                get_model(self.cfg)
            except Exception as exc:
                self._status(f"Could not load speech model {name}: {exc}", "error")
                for _ in range(int(MODEL_RETRY_S * 10)):
                    if self._closed:
                        return
                    time.sleep(0.1)
                continue
            self._model_ready.set()
            self._status("Listening" if self._capturing else "Ready")
            return

    def _transcribe_loop(self, s: _Stream) -> None:
        backlog: list[_Job] = []
        quitting = False
        while backlog or not quitting:
            if not backlog:
                job = s.tq.get()
                if job is None:
                    return
                backlog.append(job)
            while True:  # take everything already queued so it can be packed
                try:
                    job = s.tq.get_nowait()
                except queue.Empty:
                    break
                if job is None:
                    quitting = True
                else:
                    backlog.append(job)
            same = [j for j in backlog if j.session is backlog[0].session]
            batch = same[:pack([j.utt.audio.size / SR for j in same])]
            taken = {id(j) for j in batch}
            backlog = [j for j in backlog if id(j) not in taken]
            texts = self._transcribe_batch(s, batch)
            with self._lock:
                for job, text in zip(batch, texts):
                    key = (job.utt.start, job.utt.end, job.utt.audio.size)
                    if key in s.pending:
                        s.pending.remove(key)
                    if text:
                        self._add_text(job, text)
                self._release_held()
                left = sum(n for st in self._streams.values() for _, _, n in st.pending) / SR
                if self._backlog_warned and left < BACKLOG_OK_S:
                    self._backlog_warned = False
                    self._status("Transcription caught up")

    def _transcribe_batch(self, s: _Stream, batch: list[_Job]) -> list[str]:
        while not self._model_ready.wait(1.0):
            if self._closed and not (self._loader and self._loader.is_alive()):
                return [""] * len(batch)
        try:
            if s.transcriber is None:
                s.transcriber = Transcriber(self.cfg, get_model(self.cfg))
            if batch[0].session is not s.last_session:
                s.transcriber.reset_context()
                s.last_session = batch[0].session
            return s.transcriber.transcribe_many(
                [Item(j.utt.audio, j.utt.rms, j.utt.speech_prob) for j in batch])
        except Exception as exc:
            self._status(f"Transcription failed: {exc}", "error")
            return [""] * len(batch)

    # ================================================================ lines + echo dedupe (lock held)

    def _add_text(self, job: _Job, text: str) -> None:
        u = job.utt
        line = Line(job.speaker, round(max(0.0, (u.start - job.origin) / SR), 2),
                    round(max(0.0, (u.end - job.origin) / SR), 2), text)
        if job.speaker == THEM:
            self._commit(job.session, line)
            self._recent_them.append((u.start, u.end, text))
            horizon = u.end - int(RECENT_THEM_S * SR)
            while self._recent_them and self._recent_them[0][1] < horizon:
                self._recent_them.popleft()
        else:
            self._held.append(_Held(line, job.session, u.start, u.end,
                                    time.monotonic() + ECHO_MAX_HOLD_S))
            self._cmds.put(("tick", None))

    def _commit(self, session: Session, line: Line) -> None:
        try:
            session.add_line(line)
        except OSError as exc:
            self._status(f"Could not save transcript line: {exc}", "error")
        self._emit("line", line=line, meeting_dir=session.dir)

    def _them_settled(self, a0: int, a1: int) -> bool:
        """True once no Them speech overlapping [a0, a1] can still turn into a line."""
        them = self._streams[THEM]
        tol = int(ECHO_TOLERANCE_S * SR)
        if any(p0 - tol <= a1 and p1 + tol >= a0 for p0, p1, _ in them.pending):
            return False
        if not (self._capturing and them.connected):
            return True
        if them.open_utt is not None and them.open_utt - tol <= a1:
            return False
        return them.done_pos >= a1 + tol

    def _release_held(self, force: bool = False) -> None:
        if not self._held:
            return
        now = time.monotonic()
        tol = int(ECHO_TOLERANCE_S * SR)
        keep = []
        for h in self._held:
            overlapping = [t for a0, a1, t in self._recent_them
                           if a0 - tol <= h.a1 and a1 + tol >= h.a0]
            if overlapping and echo_similarity(h.line.text, overlapping) >= ECHO_SIMILARITY:
                continue  # the mic heard the speakers; the Them line already has it
            if force or now >= h.deadline or self._them_settled(h.a0, h.a1):
                self._commit(h.session, h.line)
            else:
                keep.append(h)
        self._held = keep
