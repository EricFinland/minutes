"""transcript_text, list_meetings and cleanup_old_audio on fake meeting folders."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import pytest

from minutes import config
from minutes.models import (
    LINES_FILE,
    META_FILE,
    POLISHED_TRANSCRIPT_FILE,
    SUMMARY_FILE,
    TRANSCRIPT_FILE,
    Line,
    append_line,
)
from minutes.post import cleanup_old_audio, list_meetings, transcript_text

WAVS = ("audio_me.wav", "audio_them.wav")
TEXTS = (LINES_FILE, TRANSCRIPT_FILE, META_FILE, SUMMARY_FILE)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINUTES_DATA_DIR", str(tmp_path))
    return tmp_path


def _cfg(**over):
    cfg = dict(config.DEFAULTS)
    cfg.update(over)
    return cfg


def _folder(data_dir, name):
    d = data_dir / "meetings" / name
    d.mkdir(parents=True)
    return d


def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def _full_meeting(data_dir, name, days_old, ended=True):
    d = _folder(data_dir, name)
    append_line(d / LINES_FILE, Line("Me", 0.0, 1.0, "hello"))
    (d / TRANSCRIPT_FILE).write_text("t\n", encoding="utf-8")
    (d / SUMMARY_FILE).write_text("# s\n", encoding="utf-8")
    end = (datetime.now() - timedelta(days=days_old)).isoformat(timespec="seconds")
    (d / META_FILE).write_text(json.dumps({"started": end, "ended": end if ended else None,
                                           "model": "x", "audio_spans": []}), encoding="utf-8")
    for w in WAVS:
        (d / w).write_bytes(b"RIFF0000WAVE")
    for f in d.iterdir():
        _age(f, days_old)
    return d


def test_transcript_text_prefers_polished_then_live_then_lines(data_dir):
    d = _folder(data_dir, "2026-10-29_090000")
    assert transcript_text(d) == ""
    append_line(d / LINES_FILE, Line("Them", 61.0, 62.0, "From lines."))
    assert transcript_text(d) == "Meeting 2026-10-29 09:00\n\n[00:01:01] Them: From lines.\n"
    (d / TRANSCRIPT_FILE).write_text("live\n", encoding="utf-8")
    assert transcript_text(d) == "live\n"
    (d / POLISHED_TRANSCRIPT_FILE).write_text("polished\n", encoding="utf-8")
    assert transcript_text(d) == "polished\n"


def test_list_meetings_newest_first_and_only_with_transcripts(data_dir):
    a = _folder(data_dir, "2026-10-01_090000")
    append_line(a / LINES_FILE, Line("Me", 0, 1, "x"))
    b = _folder(data_dir, "2026-10-28_170000")
    (b / TRANSCRIPT_FILE).write_text("x\n", encoding="utf-8")
    c = _folder(data_dir, "2026-10-15_120000")
    append_line(c / LINES_FILE, Line("Me", 0, 1, "x"))
    _folder(data_dir, "2026-10-29_080000")                      # empty: not a meeting yet
    (data_dir / "meetings" / "notes.txt").write_text("x", encoding="utf-8")
    assert list_meetings(_cfg()) == [b, c, a]


def test_list_meetings_when_no_meetings(data_dir):
    assert list_meetings(_cfg()) == []


def test_cleanup_deletes_only_old_wavs(data_dir):
    old = _full_meeting(data_dir, "2026-10-01_090000", days_old=10)
    new = _full_meeting(data_dir, "2026-10-27_090000", days_old=2)
    assert cleanup_old_audio(_cfg(keep_audio_days=7)) == 2
    assert not any((old / w).exists() for w in WAVS)
    assert all((old / f).exists() for f in TEXTS)
    assert all((new / w).exists() for w in WAVS)
    assert cleanup_old_audio(_cfg(keep_audio_days=7)) == 0


def test_cleanup_skips_meeting_that_may_be_live(data_dir):
    live = _full_meeting(data_dir, "2026-10-01_090000", days_old=10, ended=False)
    os.utime(live / "audio_me.wav")             # still being written
    assert cleanup_old_audio(_cfg(keep_audio_days=7)) == 0
    assert all((live / w).exists() for w in WAVS)


def test_cleanup_crashed_meeting_without_end_uses_file_age(data_dir):
    crashed = _full_meeting(data_dir, "2026-10-01_090000", days_old=9, ended=False)
    assert cleanup_old_audio(_cfg(keep_audio_days=7)) == 2
    assert all((crashed / f).exists() for f in TEXTS)


def test_cleanup_disabled(data_dir):
    _full_meeting(data_dir, "2026-10-01_090000", days_old=30)
    assert cleanup_old_audio(_cfg(keep_audio_days=None)) == 0
    assert cleanup_old_audio(_cfg(keep_audio_days=-1)) == 0
