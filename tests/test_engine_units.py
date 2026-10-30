"""Engine unit tests that need no Whisper model: VAD, guard, echo scoring, packing, files."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from minutes import models
from minutes.engine.audio import read_wav
from minutes.engine.recorder import echo_similarity
from minutes.engine.session import Session, WavWriter
from minutes.engine.transcriber import Seg, clean_segments, pack, window_frames
from minutes.engine.vad import StreamingVAD

FIX = Path(__file__).parent / "fixtures"
SR = 16000
MANIFEST = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))


def run_vad(audio: np.ndarray, cfg: dict | None = None, chunk: int = 1600):
    vad = StreamingVAD(cfg or {})
    events = []
    for i in range(0, audio.size, chunk):
        events += vad.feed(audio[i:i + chunk])
    events += vad.flush()
    return events


def utterances(events):
    return [v for k, v in events if k == "utterance"]


@pytest.mark.parametrize("track,speaker", [("conv_me.wav", "Me"), ("conv_them.wav", "Them")])
def test_vad_covers_every_reference_utterance(track, speaker):
    audio = read_wav(FIX / track)
    utts = utterances(run_vad(audio))
    refs = [u for u in MANIFEST["conversation"]["utterances"] if u["speaker"] == speaker]
    for ref in refs:
        rs, re_ = int(ref["start"] * SR), int(ref["end"] * SR)
        covering = [u for u in utts if u.start <= rs and u.end >= rs]
        assert covering, f"start of {ref['text']!r} not covered"
        # pre-roll: at least 300 ms of audio before the first sound
        assert rs - covering[0].start >= int(0.3 * SR)
        ending = [u for u in utts if u.start <= re_ <= u.end]
        assert ending, f"end of {ref['text']!r} clipped"
        assert ending[0].end - re_ >= int(0.2 * SR)
    # nothing detected outside the reference speech
    for u in utts:
        mid = (u.start + u.end) / 2 / SR
        assert any(r["start"] - 1 <= mid <= r["end"] + 1 for r in refs)


def test_vad_activity_events_pair_up():
    events = run_vad(read_wav(FIX / "conv_me.wav"))
    kinds = [k for k, _ in events if k in ("start", "end")]
    assert kinds and kinds[0] == "start"
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    assert kinds[-1] == "end"


def test_vad_cuts_long_utterance_at_a_quiet_point():
    audio = read_wav(FIX / "monologue.wav")
    utts = utterances(run_vad(audio, {"max_utterance_s": 25}))
    assert len(utts) >= 2
    assert utts[0].cut and not utts[-1].cut
    assert all((u.end - u.start) <= 25 * SR for u in utts)
    # contiguous: no audio lost or duplicated at the cut
    assert utts[1].start == utts[0].end
    cut = utts[0].end
    around = audio[cut - int(0.015 * SR): cut + int(0.015 * SR)]
    assert np.sqrt(np.mean(around ** 2)) < 0.01, "cut landed inside a word"


def test_vad_ignores_silence_and_low_noise():
    rng = np.random.default_rng(0)
    silence = np.zeros(10 * SR, np.float32)
    noise = rng.normal(0, 0.003, 20 * SR).astype(np.float32)   # about -50 dBFS
    hum = (0.01 * np.sin(2 * np.pi * 60 * np.arange(10 * SR) / SR)).astype(np.float32)
    assert utterances(run_vad(np.concatenate([silence, noise, hum]))) == []


def test_guard_drops_known_hallucinations_only_when_audio_is_weak():
    s = lambda t, ns=0.1, lp=-0.3, cr=1.2: Seg(t, ns, lp, cr)
    assert clean_segments([s(" Thank you.")], 1.0, rms=0.002, speech_prob=0.9) == ""
    assert clean_segments([s(" Thank you.")], 1.0, rms=0.05, speech_prob=0.5) == ""
    assert clean_segments([s(" Thank you.")], 1.0, rms=0.05, speech_prob=0.95) == "Thank you."
    sub = s(" Subtitles by the Amara.org community")
    assert clean_segments([sub], 2.0, rms=0.005, speech_prob=0.9) == ""
    assert clean_segments([s(" Thanks for watching!")], 2.0, rms=0.05, speech_prob=0.7) == ""


def test_guard_statistics_rules():
    ok = Seg(" We ship on Friday.", 0.1, -0.2, 1.1)
    assert clean_segments([ok], 2.0, 0.05, 0.95) == "We ship on Friday."
    assert clean_segments([Seg(" We ship on Friday.", 0.7, -1.2, 1.1)], 2.0, 0.05, 0.95) == ""
    assert clean_segments([Seg(" no no no no no no no no", 0.1, -0.2, 3.0)], 2.0, 0.05, 0.95) == ""
    # far more words than could be spoken in 0.5 s
    many = Seg(" " + " ".join(["word"] * 20), 0.1, -0.2, 1.5)
    assert clean_segments([many], 0.5, 0.05, 0.95) == ""
    # prompt regurgitation
    prompt = "Let's review the budget numbers for the second quarter today."
    echo = Seg(" review the budget numbers for the second quarter", 0.1, -0.2, 1.2)
    assert clean_segments([echo], 3.0, 0.05, 0.95, prompt=prompt) == ""
    assert clean_segments([echo], 3.0, 0.05, 0.95, prompt="") != ""


def test_echo_similarity():
    them = ["The design team needs about three more weeks to finish the onboarding flow."]
    assert echo_similarity("the design team needs about three more weeks", them) >= 0.6
    assert echo_similarity("design team needs three weeks to finish onboarding", them) >= 0.6
    assert echo_similarity("Can you send me the budget by Friday?", them) < 0.6
    assert echo_similarity("Yeah.", ["Yeah."]) >= 0.6
    assert echo_similarity("Yeah.", them) < 0.6


def test_pack_and_window():
    assert pack([3.0]) == 1
    assert pack([3.0, 4.0, 5.0]) == 3
    assert pack([20.0, 6.0]) == 1
    assert pack([30.0, 1.0]) == 1
    assert window_frames(1.0) == 1500
    assert window_frames(14.2) == 2000
    assert window_frames(27.0) == 3000


def _wav_info(path: Path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()


def test_wav_header_valid_without_close(tmp_path):
    w = WavWriter(tmp_path / "a.wav")
    w.write(np.full(SR, 0.25, np.float32))
    w.write_silence(SR // 2)
    w.sync()
    w.write(np.full(SR, 0.25, np.float32))   # after the last sync, simulated crash
    assert _wav_info(tmp_path / "a.wav") == (SR + SR // 2, SR, 1, 2)
    w.close()
    assert _wav_info(tmp_path / "a.wav")[0] == 2 * SR + SR // 2


def test_session_files(tmp_path):
    from datetime import datetime
    s = Session(tmp_path, datetime(2026, 1, 2, 3, 4, 5).astimezone(), "base")
    assert s.dir.name == "2026-01-02_030405"
    s.begin_span(0.0)
    s.write_audio("Me", np.full(SR, 0.1, np.float32), 0)
    s.write_audio("Them", np.full(SR // 2, 0.1, np.float32), SR)   # gap becomes silence
    s.write_audio("Me", np.full(SR, 0.1, np.float32), SR // 2)     # overlap is skipped
    s.add_line(models.Line("Them", 1.5, 2.0, "second"))
    s.add_line(models.Line("Me", 0.2, 1.0, "first"))
    text = (s.dir / models.TRANSCRIPT_FILE).read_text(encoding="utf-8")
    assert text.index("first") < text.index("second")
    assert [l.text for l in models.read_lines(s.dir / models.LINES_FILE)] == ["first", "second"]
    s.set_span_duration(2.0)
    s.close()
    assert _wav_info(s.dir / "audio_me.wav")[0] == _wav_info(s.dir / "audio_them.wav")[0] == SR * 3 // 2
    meta = models.read_meta(s.dir)
    assert meta["ended"] and meta["model"] == "base"
    assert meta["audio_spans"] == [{"meeting_t": 0.0, "file_t": 0.0, "dur": 2.0}]
    s2 = Session(tmp_path, datetime(2026, 1, 2, 3, 4, 5).astimezone(), "base")
    assert s2.dir != s.dir
    s2.close()
