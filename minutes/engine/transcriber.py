"""Whisper transcription of utterances, with a guard against hallucinated text.

Speed on CPU is dominated by the Whisper encoder, which faster-whisper always runs on a
full 30 s window however short the audio is. Two things keep live transcription ahead:

* Adaptive window: the mel input is padded only to max(15 s, audio + 5 s) instead of
  30 s. Shorter windows (under about 10 s) make large models loop ("Yeah. Yeah. Yeah."),
  so 15 s keeps a wide margin, and a repetition rejected by the guard is retried once
  with the full 30 s window.
* Packing: when utterances queue up, several are joined (with 0.5 s of silence between
  them) into one window and the words are split back by their timestamps. Throughput
  then rises automatically exactly when the transcriber falls behind. Whisper sometimes
  stops early inside a packed window; any utterance that got no words is transcribed
  again at the front of its own window, so nothing is lost silently.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass

import numpy as np

SR = 16000
PROMPT_CHARS = 200
FULL_WINDOW_S = 30.0
MIN_WINDOW_S = 15.0
WINDOW_PAD_S = 5.0
PACK_SEP_S = 0.5
GAP_MARGIN_S = 0.1    # words entirely inside inserted silence were never spoken
PACK_MAX_S = FULL_WINDOW_S - WINDOW_PAD_S   # audio (with separators) per packed window

# Phrases Whisper is known to invent on noise or silence (from its subtitle training data).
# "Strong" ones essentially never occur in a meeting; "weak" ones do, so they are only
# dropped when the audio itself looks like it had no real speech.
_STRONG = re.compile(
    r"(thanks?( you)?( so much| very much)? for watching|please (like|subscribe)|"
    r"like and subscribe|subtitles? by|captions? by|transcri(bed|ption) by|translated by|"
    r"amara\.org|www\.|\.com\b|see you in the next video)", re.I)
_WEAK = {
    "you", "thank you", "thank you very much", "thank you so much", "thanks", "bye",
    "bye bye", "oh", "so", "hmm", "the end", "i'm sorry", "music",
}
# Real speech, even quiet speech, gets a high mean Silero probability. Noise that slips
# past the VAD threshold sits much lower, or is barely above the noise floor.
WEAK_PROB, WEAK_RMS = 0.6, 0.003       # -50 dBFS
STRONG_PROB, STRONG_RMS = 0.8, 0.01    # -40 dBFS
MAX_WORDS_PER_S = 5.0    # faster than any real speaker; more words means invented text
MAX_COMPRESSION = 2.4


@dataclass
class Seg:
    text: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float


@dataclass
class Item:
    audio: np.ndarray
    rms: float
    speech_prob: float


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split())


def clean_segments(segs: list[Seg], duration_s: float, rms: float, speech_prob: float,
                   prompt: str = "") -> str:
    """Apply the hallucination guard and return the surviving text."""
    quiet = rms < WEAK_RMS or speech_prob < WEAK_PROB
    doubtful = rms < STRONG_RMS or speech_prob < STRONG_PROB
    prompt_plain = _plain(prompt)
    kept: list[str] = []
    for s in segs:
        text = s.text.strip()
        plain = _plain(text)
        if not plain:
            continue
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue
        if s.compression_ratio > MAX_COMPRESSION:
            continue
        if doubtful and _STRONG.search(text):
            continue
        if quiet and plain in _WEAK:
            continue
        # Whisper sometimes echoes the prompt back on unclear audio.
        if len(plain.split()) >= 6 and prompt_plain and plain in prompt_plain:
            continue
        kept.append(text)
    text = " ".join(kept).strip()
    if text and len(_plain(text).split()) > MAX_WORDS_PER_S * duration_s + 4:
        return ""
    return text


# ---------------------------------------------------------------- adaptive window

_local = threading.local()
_patch_lock = threading.Lock()
_patched = False


def _install_window_patch() -> None:
    """Let this module choose the encoder window length for its own calls only."""
    global _patched
    with _patch_lock:
        if _patched:
            return
        import faster_whisper.transcribe as fwt
        original = fwt.pad_or_trim

        def pad_or_trim(array, length: int = 3000, *, axis: int = -1):
            frames = getattr(_local, "frames", None)
            return original(array, frames or length, axis=axis)

        fwt.pad_or_trim = pad_or_trim
        _patched = True


def window_frames(audio_s: float) -> int:
    s = min(FULL_WINDOW_S, max(MIN_WINDOW_S, audio_s + WINDOW_PAD_S))
    return int(math.ceil(s)) * 100


def pack(durations_s: list[float], max_s: float = PACK_MAX_S) -> int:
    """How many leading items (at least one) fit into one packed window."""
    total, n = 0.0, 0
    for d in durations_s:
        add = d + (PACK_SEP_S if n else 0.0)
        if n and total + add > max_s:
            break
        total += add
        n += 1
    return max(1, n)


class Transcriber:
    """Transcribes utterances for one stream, keeping that stream's text as the prompt."""

    def __init__(self, cfg: dict, model) -> None:
        _install_window_patch()
        self.model = model
        self.language = cfg.get("language") or None
        if self.language in ("auto", ""):
            self.language = None
        self.beam_size = int(cfg.get("beam_size", 5))
        self.history = ""

    def reset_context(self) -> None:
        self.history = ""

    def transcribe(self, audio: np.ndarray, rms: float, speech_prob: float) -> str:
        return self.transcribe_many([Item(audio, rms, speech_prob)])[0]

    def transcribe_many(self, items: list[Item]) -> list[str]:
        """Transcribe utterances in one pass; returns one text per item ("" if dropped)."""
        texts = self._transcribe(items)
        joined = " ".join(t for t in texts if t)
        if joined:
            self.history = (self.history + " " + joined)[-PROMPT_CHARS * 2:]
        return texts

    def _transcribe(self, items: list[Item]) -> list[str]:
        sep = np.zeros(int(PACK_SEP_S * SR), np.float32)
        parts: list[np.ndarray] = []
        spans: list[tuple[float, float]] = []
        pos = 0
        for i, it in enumerate(items):
            if i:
                parts.append(sep)
                pos += sep.size
            spans.append((pos / SR, (pos + it.audio.size) / SR))
            parts.append(it.audio.astype(np.float32, copy=False))
            pos += it.audio.size
        audio = np.concatenate(parts)

        frames = window_frames(audio.size / SR)
        texts, heard, repeated = self._run(audio, items, spans, frames)
        if repeated and frames < FULL_WINDOW_S * 100:
            texts, heard, _ = self._run(audio, items, spans, int(FULL_WINDOW_S * 100))
        retry = [i for i in range(1, len(items)) if not heard[i]]
        if retry:
            for i, text in zip(retry, self._transcribe([items[i] for i in retry])):
                texts[i] = text
        return texts

    def _run(self, audio: np.ndarray, items: list[Item], spans: list[tuple[float, float]],
             frames: int) -> tuple[list[str], list[bool], bool]:
        prompt = self.history[-PROMPT_CHARS:].strip()
        split = len(items) > 1
        _local.frames = frames
        try:
            segments, _info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=False,
                condition_on_previous_text=False,
                initial_prompt=prompt or None,
                word_timestamps=split,
            )
            segments = list(segments)
        finally:
            _local.frames = None

        per_item: list[list[Seg]] = [[] for _ in items]
        heard = [False] * len(items)
        repeated = False
        end = spans[-1][1]
        for s in segments:
            if s.compression_ratio > MAX_COMPRESSION:
                repeated = True
            if not split:
                if s.start < end - GAP_MARGIN_S:   # text placed in the padding is invented
                    per_item[0].append(Seg(s.text, s.no_speech_prob, s.avg_logprob,
                                           s.compression_ratio))
                    heard[0] = True
                continue
            groups: dict[int, list[str]] = {}
            for w in s.words or []:
                idx = _owner(spans, w.start, w.end)
                if idx is not None:
                    groups.setdefault(idx, []).append(w.word)
                    heard[idx] = True
            for idx, words in groups.items():
                per_item[idx].append(Seg("".join(words), s.no_speech_prob, s.avg_logprob,
                                         s.compression_ratio))
        texts = [clean_segments(per_item[i], it.audio.size / SR, it.rms, it.speech_prob, prompt)
                 for i, it in enumerate(items)]
        texts = [t.lstrip(".,;:!? ") for t in texts]
        return texts, heard, repeated


def _owner(spans: list[tuple[float, float]], start: float, end: float) -> int | None:
    """Item a word belongs to, or None if it lies wholly in inserted silence."""
    best, best_d = None, float("inf")
    mid = (start + end) / 2
    for i, (a, b) in enumerate(spans):
        if end <= a - GAP_MARGIN_S or start >= b + GAP_MARGIN_S:
            continue
        d = 0.0 if a <= mid <= b else min(abs(mid - a), abs(mid - b))
        if d < best_d:
            best, best_d = i, d
    return best
