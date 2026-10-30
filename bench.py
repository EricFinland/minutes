"""Speech model benchmark: real-time factor and word error rate on CPU.

    python bench.py --make-fixtures      # (re)generate tests/fixtures with Windows TTS
    python bench.py                      # benchmark the default model grid
    python bench.py --models large-v3-turbo --threads 8 --dual

Utterances are cut from the fixture tracks with the same padding the live engine uses
and run through the engine's own Transcriber, so the numbers reflect what the recorder
actually does. "single" transcribes one utterance at a time (the live path while keeping
up; s/utt is the mean time per utterance). "dual" queues everything on both streams at
once and transcribes them concurrently on one shared model (num_workers=2), which is the
worst case during a meeting and exercises packing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests" / "fixtures"
SR = 16000

DAVID = "Microsoft David Desktop"
ZIRA = "Microsoft Zira Desktop"

# (speaker, voice, rate, gap_before_s, text). A negative gap overlaps the previous line.
CONVERSATION = [
    ("Me", DAVID, 0, 1.0, "Hi everyone, thanks for joining. Let's start with the quarterly roadmap review."),
    ("Them", ZIRA, 0, 0.8, "Thanks for setting this up. I reviewed the draft roadmap last night."),
    ("Me", DAVID, 1, 0.6, "Yeah."),
    ("Them", ZIRA, -1, 0.9, "The design team needs about three more weeks to finish the onboarding flow."),
    ("Me", DAVID, 0, 1.0, "I think we should move the launch date to the second week of March."),
    ("Them", ZIRA, 1, 0.7, "Sure, I can do that."),
    ("Me", DAVID, 2, 0.8, "Can you send me the updated budget numbers by Friday afternoon?"),
    ("Them", ZIRA, 0, 1.1, "One concern is that the analytics dashboard still has a few open bugs."),
    ("Me", DAVID, 0, 0.6, "Sounds good."),
    ("Them", ZIRA, -1, 0.9, "We also need approval from legal before we publish the pricing page."),
    ("Me", DAVID, 1, 1.2, "Okay, let's circle back on the hiring plan next week."),
    ("Them", ZIRA, 0, 0.8, "Great, talk to you all on Thursday."),
]

# One long turn with short pauses so it exceeds max_utterance_s and must be cut.
MONOLOGUE = [
    "Let me walk through the numbers from last quarter in a bit more detail.",
    "Revenue grew by about twelve percent, mostly from the enterprise accounts.",
    "Churn went down slightly, although the smaller customers are still leaving faster than we would like.",
    "Support tickets were flat, which is good given how many new users we added.",
    "For next quarter, the plan is to focus on onboarding and the mobile app.",
    "I will share the full deck after this call so everyone can review it.",
]
MONOLOGUE_GAP_S = 0.3

NOISE_RMS = 0.0015  # about -56 dBFS, a quiet room or a clean call


# ---------------------------------------------------------------- audio helpers

def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _trim(audio: np.ndarray, thresh: float = 0.004, margin_s: float = 0.02) -> np.ndarray:
    idx = np.flatnonzero(np.abs(audio) > thresh)
    if idx.size == 0:
        return audio
    m = int(margin_s * SR)
    return audio[max(0, idx[0] - m): idx[-1] + m]


def _tts_batch(items: list[tuple[str, int, str]], out_dir: Path) -> list[Path]:
    """Synthesize (voice, rate, text) items to 16 kHz mono WAVs with one PowerShell call."""
    spec = [{"voice": v, "rate": r, "text": t, "path": str(out_dir / f"u{i:02d}.wav")}
            for i, (v, r, t) in enumerate(items)]
    spec_path = out_dir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        f"$items = Get-Content -Raw -Encoding UTF8 '{spec_path}' | ConvertFrom-Json;"
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000,"
        "([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen),"
        "([System.Speech.AudioFormat.AudioChannel]::Mono);"
        "foreach ($it in $items) {"
        "  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "  $s.SelectVoice($it.voice); $s.Rate = [int]$it.rate;"
        "  $s.SetOutputToWaveFile($it.path, $fmt); $s.Speak($it.text); $s.Dispose() }"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                   check=True, capture_output=True)
    return [Path(s["path"]) for s in spec]


def make_fixtures() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        items = [(v, r, t) for _, v, r, _, t in CONVERSATION]
        items += [(ZIRA, 0, t) for t in MONOLOGUE]
        clips = [_trim(read_wav(p)) for p in _tts_batch(items, tmpd)]

    conv_clips, mono_clips = clips[:len(CONVERSATION)], clips[len(CONVERSATION):]

    utterances = []
    t = 0.0
    for (speaker, voice, rate, gap, text), clip in zip(CONVERSATION, conv_clips):
        start = t + gap
        utterances.append({"speaker": speaker, "start": round(start, 3),
                           "end": round(start + len(clip) / SR, 3), "text": text,
                           "voice": voice, "rate": rate})
        t = start + len(clip) / SR
    total = int((t + 1.5) * SR)
    tracks = {"Me": np.zeros(total, np.float32), "Them": np.zeros(total, np.float32)}
    for u, clip in zip(utterances, conv_clips):
        s = int(u["start"] * SR)
        tracks[u["speaker"]][s:s + len(clip)] += clip
    for name, audio in tracks.items():
        audio += rng.normal(0, NOISE_RMS, len(audio)).astype(np.float32)
        write_wav(FIXTURES / f"conv_{name.lower()}.wav", audio)

    gap = np.zeros(int(MONOLOGUE_GAP_S * SR), np.float32)
    lead = np.zeros(int(0.8 * SR), np.float32)
    parts = [lead]
    for clip in mono_clips:
        parts += [clip, gap]
    parts.append(lead)
    mono = np.concatenate(parts)
    mono += rng.normal(0, NOISE_RMS, len(mono)).astype(np.float32)
    write_wav(FIXTURES / "monologue.wav", mono)

    manifest = {
        "sample_rate": SR,
        "conversation": {"me": "conv_me.wav", "them": "conv_them.wav", "utterances": utterances},
        "monologue": {"file": "monologue.wav", "text": " ".join(MONOLOGUE),
                      "duration": round(len(mono) / SR, 3)},
    }
    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    size = sum(p.stat().st_size for p in FIXTURES.glob("*.wav"))
    print(f"fixtures written to {FIXTURES} ({size / 1e6:.1f} MB)")


# ---------------------------------------------------------------- scoring

_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "first": "1st", "second": "2nd", "third": "3rd",
}
_SPELLING = {"ok": "okay", "percent": "%", "alright": "all right"}


def normalize(text: str) -> list[str]:
    text = text.lower().replace("%", " percent ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = []
    for w in text.split():
        w = w.strip("'")
        w = _SPELLING.get(w, w)
        w = _NUMBERS.get(w, w)
        if w:
            words.append(w)
    return words


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


# ---------------------------------------------------------------- benchmark

def load_utterances(pre_s: float = 0.4, post_s: float = 0.3) -> dict[str, list[tuple[np.ndarray, str]]]:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    conv = manifest["conversation"]
    tracks = {"Me": read_wav(FIXTURES / conv["me"]), "Them": read_wav(FIXTURES / conv["them"])}
    out: dict[str, list[tuple[np.ndarray, str]]] = {"Me": [], "Them": []}
    for u in conv["utterances"]:
        a = tracks[u["speaker"]]
        s, e = int((u["start"] - pre_s) * SR), int((u["end"] + post_s) * SR)
        out[u["speaker"]].append((a[max(0, s):e], u["text"]))
    mono = read_wav(FIXTURES / manifest["monologue"]["file"])
    # Live engine cuts at 25 s, so the monologue arrives as two utterances. Split at the
    # pause before the last sentence to mimic that.
    cut = _quietest_point(mono, len(mono) - 9 * SR, len(mono) - 4 * SR)
    sentences = manifest["monologue"]["text"]
    out["Them"].append((mono[:cut], sentences))
    out["Them"].append((mono[cut:], ""))  # scored together with the previous piece
    return out


def _quietest_point(a: np.ndarray, lo: int, hi: int) -> int:
    win = int(0.2 * SR)
    best, best_e = lo, float("inf")
    for s in range(lo, hi - win, SR // 50):
        e = float(np.mean(a[s:s + win] ** 2))
        if e < best_e:
            best, best_e = s + win // 2, e
    return best


def _transcribe_stream(model, items, packed, results, key):
    """Run one stream through the engine's Transcriber.

    packed=False: one utterance at a time (the live path when keeping up).
    packed=True: everything is already queued, so utterances are packed into shared
    windows exactly as the recorder does when it falls behind.
    """
    from minutes.engine.transcriber import Item, Transcriber, pack

    tr = Transcriber({"language": "en", "beam_size": 5}, model)
    queue = [Item(a, 0.1, 0.95) for a, _ in items]
    hyp_parts = []
    t0 = time.perf_counter()
    while queue:
        n = pack([it.audio.size / SR for it in queue]) if packed else 1
        hyp_parts += tr.transcribe_many(queue[:n])
        queue = queue[n:]
    elapsed = time.perf_counter() - t0
    results[key] = {"elapsed": elapsed, "audio": sum(a.size for a, _ in items) / SR,
                    "ref": " ".join(r for _, r in items if r), "hyp": " ".join(hyp_parts)}


def bench_one(name: str, threads: int, dual: bool, data) -> dict:
    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads,
                         num_workers=2)
    load_s = time.perf_counter() - t0
    list(model.transcribe(data["Me"][0][0], language="en", beam_size=5)[0])

    results: dict = {}
    if dual:
        ts = [threading.Thread(target=_transcribe_stream,
                               args=(model, data[k], True, results, k)) for k in ("Me", "Them")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    else:
        for k in ("Me", "Them"):
            _transcribe_stream(model, data[k], False, results, k)

    rtf = max(r["elapsed"] / r["audio"] for r in results.values())
    n_utts = sum(len(v) for v in data.values())
    latency = sum(r["elapsed"] for r in results.values()) / n_utts
    ref = " ".join(results[k]["ref"] for k in ("Me", "Them"))
    hyp = " ".join(results[k]["hyp"] for k in ("Me", "Them"))
    del model
    return {"model": name, "threads": threads, "mode": "dual" if dual else "single",
            "load_s": round(load_s, 1), "rtf": round(rtf, 3),
            "latency_s": round(latency, 2) if not dual else None,
            "wer": round(wer(ref, hyp), 4), "hyp": hyp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-fixtures", action="store_true")
    ap.add_argument("--models", nargs="+", default=["large-v3-turbo", "distil-large-v3"])
    ap.add_argument("--threads", nargs="+", type=int, default=[6, 8, 12])
    ap.add_argument("--dual", action="store_true", help="only run the two-stream test")
    ap.add_argument("--single", action="store_true", help="only run the one-stream test")
    ap.add_argument("--json", type=Path, help="also write results to this file")
    args = ap.parse_args()

    if args.make_fixtures:
        make_fixtures()
        return
    if not (FIXTURES / "manifest.json").exists():
        make_fixtures()

    data = load_utterances()
    modes = [False, True]
    if args.dual:
        modes = [True]
    elif args.single:
        modes = [False]

    rows = []
    print(f"{'model':<18}{'threads':>8}{'mode':>8}{'load s':>8}{'RTF':>8}{'WER':>8}{'s/utt':>8}")
    for name in args.models:
        for threads in args.threads:
            for dual in modes:
                r = bench_one(name, threads, dual, data)
                rows.append(r)
                print(f"{r['model']:<18}{r['threads']:>8}{r['mode']:>8}{r['load_s']:>8}"
                      f"{r['rtf']:>8.3f}{r['wer'] * 100:>7.1f}%"
                      f"{r['latency_s'] if r['latency_s'] is not None else '':>8}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    worst = {r["model"]: r["hyp"] for r in rows}
    for name, hyp in worst.items():
        print(f"\n{name} transcript sample: {hyp[:300]}")


if __name__ == "__main__":
    sys.exit(main())
