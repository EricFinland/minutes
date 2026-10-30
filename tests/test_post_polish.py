"""Polish pass tests. Speech clips are made with Windows text to speech (once, then kept
under tests/fixtures/post/) and placed into fake meeting WAVs with two recording spans."""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from minutes.models import (
    LINES_FILE,
    META_FILE,
    POLISHED_LINES_FILE,
    POLISHED_TRANSCRIPT_FILE,
    Line,
    append_line,
    read_lines,
)
from minutes.post import PolishError, polish
from minutes.post.polish import (
    Span,
    drop_echo,
    file_to_meeting,
    is_hallucination,
    merge_lines,
    normalize_spans,
    wav_info,
)

FIXTURES = Path(__file__).parent / "fixtures" / "post"
SR = 16000

CLIPS = {
    "me_1": ("male", "Let's move the product launch to next Thursday."),
    "them_1": ("female", "The budget review is finished and the numbers look good."),
    "me_2": ("male", "I will send the updated design document tomorrow morning."),
    "them_2": ("female", "Please schedule a follow up meeting with the marketing team."),
}

PS_TTS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $s.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::{gender}) } catch {}
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile('{path}', $fmt)
$s.Speak('{text}')
$s.Dispose()
"""


def _make_clip(name: str) -> Path:
    path = FIXTURES / f"{name}.wav"
    if path.exists():
        return path
    if sys.platform != "win32":
        pytest.skip("speech fixtures need Windows text to speech")
    gender, text = CLIPS[name]
    FIXTURES.mkdir(parents=True, exist_ok=True)
    script = PS_TTS.replace("{gender}", gender.capitalize()).replace(
        "{path}", str(path)).replace("{text}", text.replace("'", "''"))
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                   check=True, capture_output=True, timeout=60)
    return path


def _load(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)


def _save(path: Path, samples: np.ndarray) -> None:
    data = np.clip(samples, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def _place(track: np.ndarray, clip: np.ndarray, at_s: float, gain: float = 1.0) -> None:
    i = int(at_s * SR)
    track[i:i + len(clip)] += clip * gain


@pytest.fixture(scope="module")
def clips():
    return {name: _load(_make_clip(name)) for name in CLIPS}


@pytest.fixture(scope="module")
def whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel("base", device="cpu", compute_type="int8")


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINUTES_DATA_DIR", str(tmp_path))
    return tmp_path


def _cfg(**over):
    from minutes import config
    cfg = dict(config.DEFAULTS)
    cfg.update(over)
    return cfg


# Pure helpers -----------------------------------------------------------------------

def test_normalize_spans_missing_means_one_span_from_zero():
    assert normalize_spans(None, 42.0) == [Span(0.0, 0.0, 42.0)]
    assert normalize_spans([{"bad": 1}], 5.0) == [Span(0.0, 0.0, 5.0)]


def test_normalize_spans_contiguous_and_last_runs_to_file_end():
    raw = [{"meeting_t": 300.0, "file_t": 10.0, "dur": 2.0},   # stale dur after a crash
           {"meeting_t": 0.0, "file_t": 0.0, "dur": 10.0}]
    spans = normalize_spans(raw, 25.0)
    assert spans == [Span(0.0, 0.0, 10.0), Span(300.0, 10.0, 15.0)]
    assert file_to_meeting(spans, 4.0) == 4.0
    assert file_to_meeting(spans, 12.5) == 302.5


def test_hallucination_guard():
    assert is_hallucination("Thank you.", 0.1, -0.3, 1.2, rms=0.001)
    assert not is_hallucination("Thank you.", 0.1, -0.3, 1.2, rms=0.1)
    assert is_hallucination("Subtitles by the Amara.org community", 0.1, -0.3, 1.2, rms=0.0)
    assert is_hallucination("words words", 0.8, -1.5, 1.2, rms=0.1)
    assert is_hallucination("la la la la la la", 0.0, -0.2, 3.0, rms=0.1)
    assert is_hallucination("  ...  ", 0.0, -0.2, 1.0, rms=0.1)
    assert not is_hallucination("We agreed on Friday.", 0.0, -0.2, 1.2, rms=0.1)


def test_echo_dedupe_and_merge():
    them = [Line("Them", 10.0, 13.0, "Please schedule a follow up meeting.")]
    me = [Line("Me", 10.2, 13.1, "please schedule a follow-up meeting"),
          Line("Me", 10.5, 11.0, "Sure."),
          Line("Me", 30.0, 32.0, "Please schedule a follow up meeting.")]
    kept = drop_echo(me, them)
    assert [l.text for l in kept] == ["Sure.", "Please schedule a follow up meeting."]
    # My words and the speaker echo landed in one segment: keep mine, drop the echo.
    them = [Line("Them", 107.3, 112.3, "Please schedule a follow-up meeting with the marketing team.")]
    me = [Line("Me", 100.7, 107.6, "I will send the design document. Please schedule a "
                                   "follow-up meeting with the marketing team.")]
    assert [l.text for l in drop_echo(me, them)] == ["I will send the design document."]
    merged = merge_lines([Line("Me", 0.0, 1.0, "Hello"), Line("Me", 1.3, 2.0, "there."),
                          Line("Them", 2.1, 3.0, "Hi."), Line("Me", 5.0, 6.0, "Again.")])
    assert [(l.speaker, l.text) for l in merged] == [
        ("Me", "Hello there."), ("Them", "Hi."), ("Me", "Again.")]


def test_wav_info_uses_file_size_when_header_is_stale(tmp_path):
    p = tmp_path / "a.wav"
    _save(p, np.zeros(SR, dtype=np.float32))
    with open(p, "ab") as f:            # data appended after the header was last written
        f.write(np.zeros(SR, dtype="<i2").tobytes())
    assert abs(wav_info(p).duration - 2.0) < 1e-6


def test_polish_without_audio_raises_clear_error(data_dir):
    d = data_dir / "meetings" / "2026-10-01_090000"
    d.mkdir(parents=True)
    append_line(d / LINES_FILE, Line("Me", 0.0, 1.0, "hello"))
    with pytest.raises(PolishError, match="no longer available"):
        polish(d, _cfg(), model=object())


# Real transcription -------------------------------------------------------------------

def test_polish_two_spans_maps_meeting_time_and_drops_echo(data_dir, clips, whisper_model):
    # File timeline: span A is file 0..12 s (meeting 0..12), span B is file 12..26 s
    # (meeting 100..114) after a pause in recording.
    total = 26.0
    me = np.zeros(int(total * SR), dtype=np.float32)
    them = np.zeros(int(total * SR), dtype=np.float32)
    _place(me, clips["me_1"], 1.0)
    _place(them, clips["them_1"], 6.0)
    _place(me, clips["me_2"], 13.0)
    _place(them, clips["them_2"], 19.5)
    _place(me, clips["them_2"], 19.5, gain=0.35)   # mic hears the speakers

    d = data_dir / "meetings" / "2026-10-29_140000"
    d.mkdir(parents=True)
    _save(d / "audio_me.wav", me)
    _save(d / "audio_them.wav", them)
    (d / META_FILE).write_text(json.dumps({
        "started": "2026-10-29T14:00:00", "ended": "2026-10-29T14:01:54", "model": "base",
        "audio_spans": [{"meeting_t": 0.0, "file_t": 0.0, "dur": 12.0},
                        {"meeting_t": 100.0, "file_t": 12.0, "dur": 14.0}],
    }), encoding="utf-8")

    progress = []
    out = polish(d, _cfg(language="en"), on_progress=progress.append, model=whisper_model)

    assert out == d / POLISHED_TRANSCRIPT_FILE and out.exists()
    lines = read_lines(d / POLISHED_LINES_FILE)
    text = out.read_text(encoding="utf-8")
    print("\n" + text)

    def find(speaker, word):
        hits = [l for l in lines if l.speaker == speaker and word in l.text.lower()]
        assert hits, f"no {speaker} line containing {word!r}: {lines}"
        return hits[0]

    assert abs(find("Me", "launch").start - 1.0) < 1.0
    assert abs(find("Them", "budget").start - 6.0) < 1.0
    assert abs(find("Me", "design").start - 101.0) < 1.0
    assert abs(find("Them", "marketing").start - 107.5) < 1.0
    assert not [l for l in lines if l.speaker == "Me" and "marketing" in l.text.lower()]
    assert "[00:01:41] Me:" in text or "[00:01:40] Me:" in text
    assert progress[-1].endswith("100%") and any("%" in p for p in progress[1:-1])


def test_polish_without_spans_and_one_missing_stream(data_dir, clips, whisper_model):
    them = np.zeros(8 * SR, dtype=np.float32)
    _place(them, clips["them_1"], 2.0)
    d = data_dir / "meetings" / "2026-10-29_150000"
    d.mkdir(parents=True)
    _save(d / "audio_them.wav", them)
    append_line(d / LINES_FILE, Line("Me", 0.5, 1.5, "Live line from me."))
    out = polish(d, _cfg(), model=whisper_model)
    lines = read_lines(d / POLISHED_LINES_FILE)
    assert [l.text for l in lines if l.speaker == "Me"] == ["Live line from me."]
    budget = [l for l in lines if l.speaker == "Them" and "budget" in l.text.lower()]
    assert budget and abs(budget[0].start - 2.0) < 1.0
    assert out.exists()
