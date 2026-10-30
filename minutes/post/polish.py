"""Polish pass: re-transcribe a finished meeting's audio with full context.

The live transcript is made utterance by utterance. Here each stream is transcribed in
long runs (one per recording span) so Whisper sees surrounding context, then the same
hallucination guard and echo dedupe as the live engine are applied.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from minutes.models import (
    AUDIO_FILES,
    LINES_FILE,
    ME,
    POLISHED_LINES_FILE,
    POLISHED_TRANSCRIPT_FILE,
    THEM,
    Line,
    format_transcript,
    meeting_title,
    read_lines,
    read_meta,
    write_text_atomic,
)

SAMPLE_RATE = 16000
MIN_SPAN_S = 0.3
ECHO_SIMILARITY = 0.6
ECHO_SLACK_S = 1.0         # timestamps of the two streams are not perfectly aligned
ECHO_MIN_RUN = 5           # words in a row shared with a Them line to count as echo
MERGE_GAP_S = 0.7
MERGE_MAX_S = 25.0
LOW_RMS = 0.01
PROMPT_CHARS = 200

Progress = Callable[[str], None] | None


class PolishError(Exception):
    """Raised with a message that can be shown to the user as is."""


# Audio ----------------------------------------------------------------------------

@dataclass
class WavInfo:
    path: Path
    data_offset: int
    rate: int
    channels: int
    width: int          # bytes per sample
    is_float: bool
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.rate if self.rate else 0.0


def wav_info(path: Path) -> WavInfo:
    """Parse a WAV header. The frame count comes from the file size, so a file whose
    header was not updated before a crash is still read in full."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise PolishError(f"{path.name} is not a valid WAV file.")
        fmt = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csize = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid == b"fmt ":
                body = f.read(csize)
                tag, channels, rate = struct.unpack("<HHI", body[:8])
                bits = struct.unpack("<H", body[14:16])[0]
                if tag == 0xFFFE and len(body) >= 26:
                    tag = struct.unpack("<H", body[24:26])[0]
                fmt = (tag, channels, rate, bits)
                if csize % 2:
                    f.read(1)
            elif cid == b"data":
                if fmt is None:
                    break
                tag, channels, rate, bits = fmt
                width = bits // 8
                if (tag, width) not in ((1, 2), (3, 4)) or channels < 1:
                    raise PolishError(f"{path.name} has an unsupported audio format.")
                offset = f.tell()
                frames = (size - offset) // (width * channels)
                return WavInfo(path, offset, rate, channels, width, tag == 3, frames)
            else:
                f.seek(csize + (csize % 2), os.SEEK_CUR)
    raise PolishError(f"{path.name} is not a valid WAV file.")


def read_audio(info: WavInfo, start_s: float, dur_s: float) -> np.ndarray:
    """Mono float32 samples at 16 kHz for [start_s, start_s + dur_s)."""
    first = max(0, int(round(start_s * info.rate)))
    count = max(0, min(info.frames - first, int(round(dur_s * info.rate))))
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    frame_bytes = info.width * info.channels
    with open(info.path, "rb") as f:
        f.seek(info.data_offset + first * frame_bytes)
        raw = f.read(count * frame_bytes)
    n = len(raw) // frame_bytes
    if info.is_float:
        data = np.frombuffer(raw[: n * frame_bytes], dtype="<f4").astype(np.float32)
    else:
        data = np.frombuffer(raw[: n * frame_bytes], dtype="<i2").astype(np.float32) / 32768.0
    if info.channels > 1:
        data = data.reshape(-1, info.channels).mean(axis=1)
    if info.rate != SAMPLE_RATE and len(data):
        new_n = int(len(data) * SAMPLE_RATE / info.rate)
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


# Time mapping ---------------------------------------------------------------------

@dataclass
class Span:
    meeting_t: float
    file_t: float
    dur: float


def normalize_spans(raw, file_dur: float) -> list[Span]:
    """Turn meta audio_spans into contiguous spans that cover the whole file.

    The WAV timeline is contiguous, so each span runs until the next one starts and the
    last runs to the end of the file (its stored dur may be stale after a crash).
    Missing or unusable spans mean one span from 0.
    """
    spans: list[Span] = []
    for s in raw or []:
        try:
            spans.append(Span(float(s["meeting_t"]), float(s["file_t"]), float(s.get("dur") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    spans.sort(key=lambda s: s.file_t)
    if not spans:
        return [Span(0.0, 0.0, max(0.0, file_dur))]
    if spans[0].file_t > 0:
        first = spans[0]
        spans[0] = Span(max(0.0, first.meeting_t - first.file_t), 0.0, first.dur + first.file_t)
    out = []
    for i, s in enumerate(spans):
        end = spans[i + 1].file_t if i + 1 < len(spans) else max(file_dur, s.file_t)
        out.append(Span(s.meeting_t, s.file_t, max(0.0, end - s.file_t)))
    return out


def file_to_meeting(spans: list[Span], t: float) -> float:
    span = spans[0]
    for s in spans:
        if s.file_t <= t:
            span = s
        else:
            break
    return span.meeting_t + (t - span.file_t)


# Filters --------------------------------------------------------------------------

_FILLERS = {
    "you", "thank you", "thank you very much", "thanks", "thank you so much",
    "thanks for watching", "thank you for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "bye", "music", "applause", "silence",
}
_FILLER_PREFIXES = ("subtitles by", "captions by", "transcribed by", "transcription by",
                    "translated by", "subtitled by")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def is_hallucination(text: str, no_speech_prob: float, avg_logprob: float,
                     compression_ratio: float, rms: float) -> bool:
    norm = normalize_text(text)
    if not norm:
        return True
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return True
    if compression_ratio > 2.4:
        return True
    if rms < LOW_RMS and (norm in _FILLERS or norm.startswith(_FILLER_PREFIXES)):
        return True
    return False


def similarity(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def strip_echo_runs(text: str, other: str, min_run: int = ECHO_MIN_RUN) -> str:
    """Remove runs of min_run or more words that also appear, in order, in other.

    With long context Whisper can put my own words and the speaker echo in one segment;
    this keeps my words and removes only the echoed part.
    """
    tokens = text.split()
    words: list[str] = []
    owner: list[int] = []
    for i, tok in enumerate(tokens):
        for w in normalize_text(tok).split():
            words.append(w)
            owner.append(i)
    other_words = normalize_text(other).split()
    if len(words) < min_run or len(other_words) < min_run:
        return text
    sm = difflib.SequenceMatcher(None, words, other_words, autojunk=False)
    remove = set()
    for block in sm.get_matching_blocks():
        if block.size >= min_run:
            remove.update(owner[block.a:block.a + block.size])
    if not remove:
        return text
    kept = [tok for i, tok in enumerate(tokens) if i not in remove]
    return " ".join(kept).strip(" ,;:-")


def drop_echo(me: list[Line], them: list[Line]) -> list[Line]:
    """Drop Me lines that the mic picked up from the speakers (they repeat a Them line)."""
    kept = []
    for m in me:
        near = [t for t in them
                if t.start < m.end + ECHO_SLACK_S and t.end > m.start - ECHO_SLACK_S]
        if near:
            joined = " ".join(t.text for t in near)
            text = strip_echo_runs(m.text, joined)
            if not normalize_text(text):
                continue
            best = max(similarity(text, t.text) for t in near)
            if len(near) > 1:
                best = max(best, similarity(text, joined))
            if best >= ECHO_SIMILARITY:
                continue
            if text != m.text:
                m = Line(m.speaker, m.start, m.end, text)
        kept.append(m)
    return kept


def merge_lines(lines: list[Line]) -> list[Line]:
    """Join consecutive segments of one speaker into utterance-sized lines."""
    out: list[Line] = []
    for ln in sorted(lines, key=lambda l: l.start):
        prev = out[-1] if out else None
        if (prev and prev.speaker == ln.speaker and ln.start - prev.end <= MERGE_GAP_S
                and ln.end - prev.start <= MERGE_MAX_S):
            out[-1] = Line(prev.speaker, prev.start, max(prev.end, ln.end),
                           f"{prev.text} {ln.text}".strip())
        else:
            out.append(ln)
    return out


# Thread priority ------------------------------------------------------------------

def _set_low_priority() -> Callable[[], None]:
    """Lower this thread's priority on Windows. Returns a function that restores it."""
    if sys.platform != "win32":
        return lambda: None
    try:
        import win32api
        import win32con

        handle = win32api.GetCurrentThread()
        old = win32api.GetThreadPriority(handle)
        win32api.SetThreadPriority(handle, win32con.THREAD_PRIORITY_BELOW_NORMAL)
        return lambda: win32api.SetThreadPriority(win32api.GetCurrentThread(), old)
    except Exception:
        return lambda: None


# Transcription --------------------------------------------------------------------

def _transcribe_stream(model, info: WavInfo, spans: list[Span], speaker: str, cfg: dict,
                       report: Callable[[float], None]) -> list[Line]:
    pad_ms = max(int(cfg.get("pre_roll_ms", 400)), 400)
    vad_params = {
        "threshold": float(cfg.get("vad_threshold", 0.35)),
        "speech_pad_ms": pad_ms,
        "min_silence_duration_ms": max(int(cfg.get("end_silence_ms", 700)), 500),
        "min_speech_duration_ms": 0,
    }
    lines: list[Line] = []
    prompt = None
    for span in spans:
        if span.dur < MIN_SPAN_S:
            report(span.dur)
            continue
        audio = read_audio(info, span.file_t, span.dur)
        if len(audio) < MIN_SPAN_S * SAMPLE_RATE:
            report(span.dur)
            continue
        segments, _ = model.transcribe(
            audio,
            language=cfg.get("language") or None,
            beam_size=int(cfg.get("beam_size", 5)),
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=vad_params,
            initial_prompt=prompt,
            word_timestamps=False,
        )
        done = 0.0
        for seg in segments:
            text = seg.text.strip()
            a = max(0, int(seg.start * SAMPLE_RATE))
            b = max(a + 1, int(seg.end * SAMPLE_RATE))
            chunk = audio[a:b]
            rms = float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0
            progressed = min(seg.end, span.dur) - done
            if progressed > 0:
                report(progressed)
                done += progressed
            if is_hallucination(text, seg.no_speech_prob, seg.avg_logprob,
                                seg.compression_ratio, rms):
                continue
            if lines and lines[-1].speaker == speaker and len(text.split()) >= 4 \
                    and normalize_text(lines[-1].text) == normalize_text(text):
                continue  # repetition loop
            start = span.meeting_t + seg.start
            end = span.meeting_t + min(seg.end, span.dur)
            lines.append(Line(speaker, round(start, 2), round(max(end, start), 2), text))
        if span.dur > done:
            report(span.dur - done)
        tail = " ".join(l.text for l in lines[-6:])
        prompt = tail[-PROMPT_CHARS:] or None
    return lines


def polish(meeting_dir, cfg: dict, on_progress: Progress = None, model=None) -> Path:
    """Re-transcribe the meeting audio and write lines_polished.jsonl and
    transcript_polished.txt. Returns the polished transcript path. Raises PolishError.

    on_progress receives short status strings such as "Polishing transcript... 42%".
    """
    meeting_dir = Path(meeting_dir)
    restore = _set_low_priority()
    try:
        return _polish(meeting_dir, cfg, on_progress, model)
    finally:
        restore()


def _polish(meeting_dir: Path, cfg: dict, on_progress: Progress, model) -> Path:
    keep_days = cfg.get("keep_audio_days", 7)
    infos: dict[str, WavInfo] = {}
    for speaker, name in AUDIO_FILES.items():
        path = meeting_dir / name
        if path.exists():
            infos[speaker] = wav_info(path)
    if not infos:
        raise PolishError(
            "The audio for this meeting is no longer available (recordings are deleted "
            f"after {keep_days} days), so it cannot be polished. The original transcript "
            "is still there."
        )

    meta = read_meta(meeting_dir)
    spans_by = {sp: normalize_spans(meta.get("audio_spans"), info.duration)
                for sp, info in infos.items()}
    total = sum(s.dur for spans in spans_by.values() for s in spans) or 1.0
    state = {"done": 0.0, "last": -1}

    def report(seconds: float) -> None:
        state["done"] += seconds
        pct = min(99, int(state["done"] * 100 / total))
        if on_progress and pct != state["last"]:
            state["last"] = pct
            on_progress(f"Polishing transcript... {pct}%")

    if on_progress:
        on_progress("Polishing transcript... 0%")
        state["last"] = 0
    if model is None:
        if on_progress:
            on_progress("Loading speech model...")
        from minutes.engine import get_model
        model = get_model(cfg)

    by_speaker: dict[str, list[Line]] = {}
    for speaker in (THEM, ME):
        if speaker in infos:
            by_speaker[speaker] = _transcribe_stream(
                model, infos[speaker], spans_by[speaker], speaker, cfg, report)
        else:
            # Audio for one side is gone: keep that side's live lines.
            by_speaker[speaker] = [l for l in read_lines(meeting_dir / LINES_FILE)
                                   if l.speaker == speaker]

    me = drop_echo(by_speaker[ME], by_speaker[THEM])
    lines = merge_lines(me + by_speaker[THEM])

    jsonl = "".join(json.dumps(asdict(ln), ensure_ascii=False) + "\n" for ln in lines)
    write_text_atomic(meeting_dir / POLISHED_LINES_FILE, jsonl)

    out = meeting_dir / POLISHED_TRANSCRIPT_FILE
    write_text_atomic(out, format_transcript(lines, meeting_title(meeting_dir) + " (polished)"))
    if on_progress:
        on_progress("Polishing transcript... 100%")
    return out
