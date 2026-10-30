"""Reading finished meetings back out, listing them, and pruning old audio."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from minutes import config
from minutes.models import (
    AUDIO_FILES,
    LINES_FILE,
    POLISHED_LINES_FILE,
    POLISHED_TRANSCRIPT_FILE,
    TRANSCRIPT_FILE,
    Line,
    format_transcript,
    meeting_title,
    parse_transcript,
    read_lines,
    read_meta,
)

# A meeting whose meta says it has not ended and that was touched this recently
# may still be recording, so its audio is never deleted.
LIVE_GRACE_S = 3600


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def transcript_text(meeting_dir) -> str:
    """Best available transcript: polished, else live transcript, else rebuilt from lines."""
    meeting_dir = Path(meeting_dir)
    for name in (POLISHED_TRANSCRIPT_FILE, TRANSCRIPT_FILE):
        text = _read_text(meeting_dir / name)
        if text and text.strip():
            return text
    lines = read_lines(meeting_dir / LINES_FILE)
    if lines:
        return format_transcript(lines, meeting_title(meeting_dir))
    return ""


def load_lines(meeting_dir) -> list[Line]:
    """Best available lines with real timestamps, sorted by start."""
    meeting_dir = Path(meeting_dir)
    for name in (POLISHED_LINES_FILE, LINES_FILE):
        lines = read_lines(meeting_dir / name)
        if lines:
            return lines
    for name in (POLISHED_TRANSCRIPT_FILE, TRANSCRIPT_FILE):
        text = _read_text(meeting_dir / name)
        if text:
            lines = parse_transcript(text)
            if lines:
                return lines
    return []


def _folder_time(path: Path) -> float:
    try:
        return datetime.strptime(path.name, "%Y-%m-%d_%H%M%S").timestamp()
    except ValueError:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0


def list_meetings(cfg: dict | None = None) -> list[Path]:
    """Meeting folders that contain a transcript, newest first."""
    root = config.meetings_dir()
    found = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    for d in entries:
        if d.is_dir() and ((d / LINES_FILE).exists() or (d / TRANSCRIPT_FILE).exists()):
            found.append(d)
    found.sort(key=lambda p: (_folder_time(p), p.name), reverse=True)
    return found


def _parse_iso(value) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.timestamp()  # naive means local time
    return dt.astimezone(timezone.utc).timestamp()


def _newest_mtime(meeting_dir: Path) -> float:
    newest = 0.0
    try:
        for f in meeting_dir.iterdir():
            try:
                newest = max(newest, f.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return newest


def cleanup_old_audio(cfg: dict) -> int:
    """Delete audio WAVs of meetings older than cfg["keep_audio_days"]. Returns files deleted.

    Text files (transcripts, lines, summary, meta) are never touched.
    """
    days = cfg.get("keep_audio_days", 7)
    if days is None:
        return 0
    try:
        days = float(days)
    except (TypeError, ValueError):
        return 0
    if days < 0:
        return 0

    now = time.time()
    cutoff = now - days * 86400
    deleted = 0
    try:
        folders = [d for d in config.meetings_dir().iterdir() if d.is_dir()]
    except OSError:
        return 0

    for d in folders:
        wavs = [d / name for name in AUDIO_FILES.values() if (d / name).exists()]
        if not wavs:
            continue
        meta = read_meta(d)
        newest = _newest_mtime(d)
        if meta.get("ended") is None and now - newest < LIVE_GRACE_S:
            continue  # may be recording right now
        ended = _parse_iso(meta.get("ended"))
        reference = ended if ended is not None else newest
        if reference > cutoff:
            continue
        for wav in wavs:
            try:
                wav.unlink()
                deleted += 1
            except OSError:
                continue  # locked or already gone; try again next time
    return deleted
