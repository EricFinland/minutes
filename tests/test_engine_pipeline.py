"""Full Recorder pipeline with injected file sources (no audio hardware needed).

Most tests use the small cached `base` Whisper model; one accuracy test uses the
configured default model.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from minutes import config, models
from minutes.engine import FileSource, Recorder
from minutes.engine.audio import read_wav

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench  # noqa: E402  (word error rate helper)

FIX = Path(__file__).parent / "fixtures"
SR = 16000
MANIFEST = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
REFS = MANIFEST["conversation"]["utterances"]


class Collector:
    def __init__(self) -> None:
        self.events: list[models.Event] = []
        self._lock = threading.Lock()

    def __call__(self, ev: models.Event) -> None:
        with self._lock:
            self.events.append(ev)

    def of(self, kind: str) -> list[models.Event]:
        with self._lock:
            return [e for e in self.events if e.kind == kind]


def wait_for(cond, timeout: float, step: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINUTES_DATA_DIR", str(tmp_path))
    return tmp_path


def make_cfg(**over) -> dict:
    cfg = dict(config.DEFAULTS)
    cfg.update(whisper_model="base", cpu_threads=8, wav_sync_s=0.5)
    cfg.update(over)
    return cfg


def run_files(cfg, me_audio, them_audio, speed: float = 0.0, timeout: float = 180.0):
    """Run a Recorder on two arrays until both are consumed, then shut it down."""
    sources = {"Me": FileSource(me_audio, speed=speed), "Them": FileSource(them_audio, speed=speed)}
    col = Collector()
    rec = Recorder(cfg, col, sources={k: (lambda s=s: s) for k, s in sources.items()})
    rec.start()
    assert wait_for(lambda: all(s.finished.is_set() for s in sources.values()), timeout)
    rec.shutdown(timeout=timeout)
    return rec, col


def meeting_dirs(root: Path) -> list[Path]:
    return sorted(p for p in (root / "meetings").iterdir() if p.is_dir())


def wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2)
        return w.getnframes()


def noise(seconds: float, rms: float = 0.0015, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, rms, int(seconds * SR)).astype(np.float32)


def speaker_wer(lines, speaker: str, refs=REFS) -> float:
    ref = " ".join(r["text"] for r in refs if r["speaker"] == speaker)
    hyp = " ".join(l.text for l in sorted(lines, key=lambda l: l.start) if l.speaker == speaker)
    return bench.wer(ref, hyp)


# ---------------------------------------------------------------- tests

def test_two_speakers_end_to_end(data_dir):
    me, them = read_wav(FIX / "conv_me.wav"), read_wav(FIX / "conv_them.wav")
    rec, col = run_files(make_cfg(), me, them)

    dirs = meeting_dirs(data_dir)
    assert len(dirs) == 1
    d = dirs[0]
    lines = models.read_lines(d / models.LINES_FILE)
    assert {l.speaker for l in lines} == {"Me", "Them"}
    assert speaker_wer(lines, "Me") < 0.1
    assert speaker_wer(lines, "Them") < 0.1

    # every line sits on a reference utterance of the same speaker
    for line in lines:
        mid = (line.start + line.end) / 2
        assert any(r["speaker"] == line.speaker and r["start"] - 1 <= mid <= r["end"] + 1
                   for r in REFS), line

    # first and last words of every reference utterance survive (no clipping)
    text = {s: " ".join(bench.normalize(" ".join(l.text for l in lines if l.speaker == s)))
            for s in ("Me", "Them")}
    for r in REFS:
        words = bench.normalize(r["text"])
        assert words[0] in text[r["speaker"]].split(), (r["text"], text[r["speaker"]])
        assert words[-1] in text[r["speaker"]].split(), (r["text"], text[r["speaker"]])

    transcript = (d / models.TRANSCRIPT_FILE).read_text(encoding="utf-8")
    assert transcript.startswith(models.meeting_title(d))
    assert len(models.parse_transcript(transcript)) == len(lines)

    meta = models.read_meta(d)
    assert meta["ended"] and meta["model"] == "base"
    assert len(meta["audio_spans"]) == 1 and meta["audio_spans"][0]["file_t"] == 0.0
    n_me, n_them = wav_frames(d / "audio_me.wav"), wav_frames(d / "audio_them.wav")
    assert n_me == n_them
    assert abs(n_me - me.size) < SR // 2

    assert rec.lines() == [] and rec.meeting_dir is None   # closed by shutdown
    line_events = col.of("line")
    assert len(line_events) == len(lines)
    assert all(e.meeting_dir == d for e in line_events)
    assert [e.meeting_dir for e in col.of("meeting")] == [d, None]
    assert [e.recording for e in col.of("state")][0] is True
    assert col.of("state")[-1].recording is False
    for spk in ("Me", "Them"):
        acts = [e.active for e in col.of("activity") if e.speaker == spk]
        assert acts and acts[0] is True and acts[-1] is False
        assert all(a != b for a, b in zip(acts, acts[1:]))


def test_echo_is_deduplicated(data_dir):
    them = read_wav(FIX / "conv_them.wav")
    delay = int(0.03 * SR)
    echo = np.concatenate([np.zeros(delay, np.float32), them[:-delay] * 0.35])
    echo += noise(echo.size / SR, seed=3)
    run_files(make_cfg(), echo, them)
    lines = models.read_lines(meeting_dirs(data_dir)[0] / models.LINES_FILE)
    assert lines, "expected Them lines"
    assert all(l.speaker == "Them" for l in lines), [l for l in lines if l.speaker == "Me"]
    assert speaker_wer(lines, "Them") < 0.1


def test_no_text_from_silence_or_noise(data_dir):
    quiet = np.concatenate([np.zeros(5 * SR, np.float32), noise(15, rms=0.003)])
    hum = noise(20, rms=0.001, seed=1) + (0.005 * np.sin(2 * np.pi * 50 * np.arange(20 * SR) / SR)).astype(np.float32)
    rec, col = run_files(make_cfg(), quiet, hum)
    assert col.of("line") == []
    assert not any(e.speaker for e in col.of("activity"))
    # a meeting in which nobody spoke is not kept
    assert len(col.of("meeting")) == 2 and meeting_dirs(data_dir) == []


def test_guard_on_forced_noise_utterance():
    """Bypass the VAD: feed noise straight to Whisper and make sure nothing survives."""
    from minutes.engine import get_model
    from minutes.engine.transcriber import Transcriber

    tr = Transcriber(make_cfg(), get_model(make_cfg()))
    for seed in range(3):
        audio = noise(2.0, rms=0.004, seed=seed)
        assert tr.transcribe(audio, rms=0.004, speech_prob=0.4) == ""
    assert tr.transcribe(np.zeros(SR, np.float32), rms=0.0, speech_prob=0.36) == ""


def test_files_are_valid_while_recording(data_dir):
    """Crash safety: lines.jsonl, WAV headers and meta are readable without shutdown."""
    me, them = read_wav(FIX / "conv_me.wav"), read_wav(FIX / "conv_them.wav")
    sources = {"Me": FileSource(me, speed=4.0), "Them": FileSource(them, speed=4.0)}
    rec = Recorder(make_cfg(), Collector(), sources={k: (lambda s=s: s) for k, s in sources.items()})
    try:
        rec.start()
        assert wait_for(lambda: rec.meeting_dir is not None, 10)
        d = rec.meeting_dir
        assert wait_for(lambda: len(models.read_lines(d / models.LINES_FILE)) >= 3, 60)
        time.sleep(1.2)   # past at least one header sync (wav_sync_s=0.5)
        lines = models.read_lines(d / models.LINES_FILE)
        assert len(lines) >= 3 and all(l.text for l in lines)
        for name in ("audio_me.wav", "audio_them.wav"):
            n = wav_frames(d / name)
            assert n > 2 * SR
            assert (d / name).stat().st_size >= 44 + 2 * n
        meta = json.loads((d / models.META_FILE).read_text(encoding="utf-8"))
        assert meta["ended"] is None and meta["audio_spans"][0]["dur"] > 0
        assert "Me:" in (d / models.TRANSCRIPT_FILE).read_text(encoding="utf-8") or \
               "Them:" in (d / models.TRANSCRIPT_FILE).read_text(encoding="utf-8")
    finally:
        rec.shutdown(timeout=30)


def test_meeting_auto_split(data_dir):
    me_track = read_wav(FIX / "conv_me.wav")
    first = me_track[int(0.5 * SR):int(7.0 * SR)]       # "Hi everyone ... roadmap review."
    second = me_track[int(18.3 * SR):int(22.4 * SR)]     # "I think we should move ..."
    me = np.concatenate([first, noise(6, seed=5), second, noise(1.5, seed=6)])
    them = noise(me.size / SR, seed=7)
    rec, col = run_files(make_cfg(meeting_gap_minutes=3 / 60), me, them)

    # opening order (folder names come from wall-clock start times, which a faster than
    # real-time replay does not respect)
    dirs = [e.meeting_dir for e in col.of("meeting") if e.meeting_dir is not None]
    assert len(dirs) == 2 and sorted(dirs) == meeting_dirs(data_dir)
    l1 = models.read_lines(dirs[0] / models.LINES_FILE)
    l2 = models.read_lines(dirs[1] / models.LINES_FILE)
    assert "roadmap" in " ".join(l.text for l in l1).lower()
    assert "launch date" in " ".join(l.text for l in l2).lower()
    assert all(l.start < 2.0 for l in l2[:1])          # new meeting starts at the speech
    assert [e.meeting_dir is None for e in col.of("meeting")] == [False, True, False, True]
    m2 = models.read_meta(dirs[1])
    assert m2["ended"] and m2["audio_spans"][0]["meeting_t"] == 0.0
    # the second meeting's audio begins with the pre-roll, not with silence
    n = wav_frames(dirs[1] / "audio_me.wav")
    assert 3 * SR < n < 9 * SR


def test_stop_and_resume_appends_a_span(data_dir):
    me, them = read_wav(FIX / "conv_me.wav")[:9 * SR], read_wav(FIX / "conv_them.wav")[:13 * SR]
    made: list[FileSource] = []

    def factory(audio):
        def make():
            made.append(FileSource(audio))
            return made[-1]
        return make

    col = Collector()
    rec = Recorder(make_cfg(), col, sources={"Me": factory(me), "Them": factory(them)})
    rec.start()
    assert wait_for(lambda: len(made) == 2 and all(s.finished.is_set() for s in made), 30)
    rec.stop()
    assert wait_for(lambda: any(e.recording is False for e in col.of("state")), 30)
    rec.start()
    assert wait_for(lambda: len(made) == 4 and all(s.finished.is_set() for s in made), 30)
    rec.shutdown(timeout=120)

    d = meeting_dirs(data_dir)[0]
    spans = models.read_meta(d)["audio_spans"]
    assert len(spans) == 2
    assert spans[1]["file_t"] >= spans[0]["file_t"] + spans[0]["dur"] - 0.01
    assert spans[1]["meeting_t"] >= spans[0]["meeting_t"]
    assert wav_frames(d / "audio_me.wav") == wav_frames(d / "audio_them.wav")
    lines = models.read_lines(d / models.LINES_FILE)
    assert sum("roadmap review" in l.text.lower() for l in lines) == 2   # once per span


def test_unavailable_source_keeps_the_other_running(data_dir):
    def broken():
        raise RuntimeError("no microphone")

    them = read_wav(FIX / "conv_them.wav")[:13 * SR]
    src = FileSource(them)
    col = Collector()
    rec = Recorder(make_cfg(), col, sources={"Me": broken, "Them": lambda: src})
    rec.start()
    assert wait_for(src.finished.is_set, 30)
    rec.shutdown(timeout=60)
    errors = [e.text for e in col.of("status") if e.level == "error"]
    assert any("Mic unavailable" in t for t in errors), errors
    lines = models.read_lines(meeting_dirs(data_dir)[0] / models.LINES_FILE)
    assert lines and all(l.speaker == "Them" for l in lines)


def test_shutdown_is_idempotent(data_dir):
    rec = Recorder(make_cfg(), Collector(),
                   sources={"Me": lambda: FileSource(noise(1)), "Them": lambda: FileSource(noise(1))})
    rec.shutdown(timeout=5)
    rec.shutdown(timeout=5)
    rec.start()   # ignored after shutdown
    assert rec.recording is False


def test_default_model_accuracy(data_dir):
    """The configured default model on both tracks plus a monologue that must be cut."""
    cfg = make_cfg(whisper_model=config.DEFAULTS["whisper_model"],
                   cpu_threads=config.DEFAULTS["cpu_threads"])
    me, them = read_wav(FIX / "conv_me.wav"), read_wav(FIX / "conv_them.wav")
    mono = read_wav(FIX / MANIFEST["monologue"]["file"])
    them = np.concatenate([them, mono])
    me = np.concatenate([me, noise(mono.size / SR, seed=9)])
    run_files(cfg, me, them, timeout=600)

    lines = models.read_lines(meeting_dirs(data_dir)[0] / models.LINES_FILE)
    assert speaker_wer(lines, "Me") <= 0.05
    refs = REFS + [{"speaker": "Them", "text": MANIFEST["monologue"]["text"]}]
    assert speaker_wer(lines, "Them", refs) <= 0.05
