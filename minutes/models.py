"""Shared data types and on-disk formats used by the engine, UI and post tools."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ME = "Me"
THEM = "Them"

LINES_FILE = "lines.jsonl"
TRANSCRIPT_FILE = "transcript.txt"
META_FILE = "meta.json"
SUMMARY_FILE = "summary.md"
POLISHED_LINES_FILE = "lines_polished.jsonl"
POLISHED_TRANSCRIPT_FILE = "transcript_polished.txt"
AUDIO_FILES = {ME: "audio_me.wav", THEM: "audio_them.wav"}


@dataclass
class Line:
    speaker: str   # ME or THEM
    start: float   # seconds since meeting start
    end: float
    text: str


@dataclass
class Event:
    kind: str                      # "line" | "state" | "meeting" | "status" | "activity"
    data: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__["data"][name]
        except KeyError:
            raise AttributeError(name) from None


def event(kind: str, **data: Any) -> Event:
    return Event(kind, data)


def fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def meeting_title(meeting_dir: Path) -> str:
    try:
        started = datetime.strptime(meeting_dir.name, "%Y-%m-%d_%H%M%S")
        return "Meeting " + started.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return "Meeting " + meeting_dir.name


def format_transcript(lines: Iterable[Line], title: str) -> str:
    out = [title, ""]
    for ln in sorted(lines, key=lambda l: l.start):
        out.append(f"[{fmt_ts(ln.start)}] {ln.speaker}: {ln.text}")
    return "\n".join(out) + "\n"


_LINE_RE = re.compile(r"^\[(\d+):(\d\d):(\d\d)\] ([^:]+): (.*)$")


def parse_transcript(text: str) -> list[Line]:
    lines = []
    for raw in text.splitlines():
        m = _LINE_RE.match(raw)
        if m:
            h, mi, s, speaker, body = m.groups()
            t = int(h) * 3600 + int(mi) * 60 + int(s)
            lines.append(Line(speaker, float(t), float(t), body))
    return lines


def append_line(path: Path, line: Line) -> None:
    """Append one line to a jsonl file and force it to disk."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(line), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_lines(path: Path) -> list[Line]:
    """Read a jsonl file, skipping a torn last line left by a crash."""
    lines: list[Line] = []
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                try:
                    d = json.loads(raw)
                    lines.append(Line(d["speaker"], float(d["start"]), float(d["end"]), d["text"]))
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        pass
    lines.sort(key=lambda l: l.start)
    return lines


_write_lock = threading.Lock()


def write_text_atomic(path: Path, text: str) -> None:
    # Serialized because several threads rewrite meta.json. The retry covers Windows
    # briefly locking the target (antivirus, search indexer, a reader with it open).
    with _write_lock:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))


def read_meta(meeting_dir: Path) -> dict:
    try:
        with open(meeting_dir / META_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_meta(meeting_dir: Path, meta: dict) -> None:
    write_text_atomic(meeting_dir / META_FILE, json.dumps(meta, indent=2))
