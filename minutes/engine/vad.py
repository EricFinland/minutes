"""Streaming voice activity detection with Silero v6 (ONNX), cutting audio into utterances.

Positions are absolute sample indices on the recorder's shared 16 kHz timeline, so an
utterance from the Me stream and one from the Them stream can be compared directly.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np

SR = 16000
FRAME = 512          # Silero v6 window at 16 kHz (32 ms)
CONTEXT = 64         # samples of the previous window the model expects in front
NEG_OFFSET = 0.15    # hysteresis: speech ends only when prob drops below threshold - this
MIN_SPEECH_MS = 160  # fewer voiced ms than this is a click or a breath, not a word
CUT_SEARCH_S = 4.0   # where to look for a quiet spot when an utterance is too long
# Silero confirms soft onsets ("s", "f", "h") a frame or a few late, so the pre-roll is
# counted from where speech probably began, not from the triggering frame.
ONSET_SLACK_MS = 200


def silero_path() -> str:
    import faster_whisper
    return os.path.join(os.path.dirname(faster_whisper.__file__), "assets", "silero_vad_v6.onnx")


_session = None


def _get_session():
    """One ONNX session shared by all streams. Recurrent state is passed per call."""
    global _session
    if _session is None:
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4
        _session = onnxruntime.InferenceSession(
            silero_path(), providers=["CPUExecutionProvider"], sess_options=opts)
    return _session


class SileroStream:
    """Frame-by-frame Silero with carried LSTM state and context."""

    def __init__(self) -> None:
        self._session = _get_session()
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._context = np.zeros(CONTEXT, np.float32)

    def probs(self, frames: np.ndarray) -> np.ndarray:
        """frames: (n, 512) float32. Returns n speech probabilities."""
        n = frames.shape[0]
        batch = np.empty((n, CONTEXT + FRAME), np.float32)
        batch[0, :CONTEXT] = self._context
        if n > 1:
            batch[1:, :CONTEXT] = frames[:-1, -CONTEXT:]
        batch[:, CONTEXT:] = frames
        out, self._h, self._c = self._session.run(
            None, {"input": batch, "h": self._h, "c": self._c})
        self._context = frames[-1, -CONTEXT:].copy()
        return np.asarray(out, np.float32).reshape(-1)


@dataclass
class Utterance:
    start: int            # absolute sample index (inclusive)
    end: int              # absolute sample index (exclusive)
    audio: np.ndarray     # float32 samples [start, end)
    rms: float            # loudness of the voiced part
    speech_prob: float    # mean Silero probability over voiced frames
    cut: bool = False     # ended by the max-length rule rather than by silence


class StreamingVAD:
    """Feed audio in any chunk size; get back activity transitions and finished utterances.

    `feed` returns a list of events:
      ("start", abs_sample)    speech confirmed (enough voiced frames)
      ("end", abs_sample)      speech over (only after a confirmed start)
      ("utterance", Utterance)
    """

    def __init__(self, cfg: dict, model: SileroStream | None = None) -> None:
        self.threshold = float(cfg.get("vad_threshold", 0.35))
        self.neg_threshold = max(0.05, self.threshold - NEG_OFFSET)
        self.pre_roll = int((max(400, cfg.get("pre_roll_ms", 400)) + ONSET_SLACK_MS) * SR / 1000)
        self.post_roll = int(max(300, cfg.get("post_roll_ms", 300)) * SR / 1000)
        self.end_silence = int(cfg.get("end_silence_ms", 700) * SR / 1000)
        self.max_utt = int(float(cfg.get("max_utterance_s", 25)) * SR)
        self.min_voiced = max(1, round(MIN_SPEECH_MS * SR / 1000 / FRAME))
        self.model = model or SileroStream()
        self.reset(0)

    # ------------------------------------------------------------ state

    def reset(self, pos: int) -> None:
        """Start fresh at absolute sample `pos` (after a gap or a pause)."""
        self.model.reset()
        self.pos = pos                      # absolute index of the next sample fed
        self._pending = np.zeros(0, np.float32)
        self._history: deque[np.ndarray] = deque(maxlen=self.pre_roll // FRAME + 2)
        self._last_end = pos                # pre-roll never reaches back past this
        self._in_speech = False
        self._clear_utterance()

    def _clear_utterance(self) -> None:
        self._utt_start = 0
        self._utt_frames: list[np.ndarray] = []
        self._voiced = 0
        self._voiced_sq = 0.0
        self._voiced_prob = 0.0
        self._last_voiced_end = 0
        self._silence_since: int | None = None
        self._confirmed = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech and self._confirmed

    @property
    def speech_start(self) -> int | None:
        return self._utt_start if self._in_speech else None

    # ------------------------------------------------------------ feeding

    def feed(self, samples: np.ndarray) -> list[tuple]:
        events: list[tuple] = []
        if self._pending.size:
            samples = np.concatenate([self._pending, samples])
        n = samples.size // FRAME
        if n:
            frames = samples[:n * FRAME].reshape(n, FRAME)
            probs = self.model.probs(frames)
            for frame, p in zip(frames, probs):
                self._step(frame, float(p), events)
        self._pending = samples[n * FRAME:].copy()
        return events

    def flush(self) -> list[tuple]:
        """End any utterance in progress (recording stopped). Keeps the trailing samples."""
        events: list[tuple] = []
        if self._pending.size:
            tail = self._pending
            self._pending = np.zeros(0, np.float32)
            frame = np.zeros(FRAME, np.float32)
            frame[:tail.size] = tail
            p = float(self.model.probs(frame[None, :])[0])
            self._step(frame, p, events, real=tail.size)
        if self._in_speech:
            end = self._last_voiced_end + self.post_roll
            self._finish(min(end, self.pos), events)
        return events

    def _step(self, frame: np.ndarray, p: float, events: list, real: int = FRAME) -> None:
        start = self.pos
        self.pos += real
        frame = frame[:real]
        if not self._in_speech:
            self._history.append(frame)
            if p >= self.threshold:
                self._begin(start, events)
                self._voice(frame, p, self.pos, events)
            return

        self._utt_frames.append(frame)
        if p >= self.threshold:
            self._voice(frame, p, self.pos, events)
            self._silence_since = None
        elif p < self.neg_threshold and self._silence_since is None:
            self._silence_since = start

        if self._silence_since is not None and self.pos - self._silence_since >= self.end_silence:
            self._finish(min(self._last_voiced_end + self.post_roll, self.pos), events)
        elif self.pos - self._utt_start >= self.max_utt:
            self._cut_long(events)

    def _begin(self, frame_start: int, events: list) -> None:
        self._in_speech = True
        self._clear_utterance()
        start = max(frame_start - self.pre_roll, self._last_end)
        # history already holds the triggering frame; keep the pre-roll in front of it
        hist = list(self._history)
        self._history.clear()
        audio = np.concatenate(hist) if hist else np.zeros(0, np.float32)
        hist_start = self.pos - audio.size
        if start < hist_start:
            start = hist_start
        self._utt_frames = [audio[start - hist_start:]]
        self._utt_start = start

    def _voice(self, frame: np.ndarray, p: float, end: int, events: list) -> None:
        self._voiced += 1
        self._voiced_sq += float(np.dot(frame, frame))
        self._voiced_prob += p
        self._last_voiced_end = end
        if not self._confirmed and self._voiced >= self.min_voiced:
            self._confirmed = True
            events.append(("start", self._utt_start))

    def _finish(self, end: int, events: list, cut: bool = False) -> None:
        audio = np.concatenate(self._utt_frames) if self._utt_frames else np.zeros(0, np.float32)
        keep = max(0, end - self._utt_start)
        tail = audio[keep:]
        audio = audio[:keep]
        if self._confirmed and audio.size:
            n = max(1, self._voiced * FRAME)
            events.append(("utterance", Utterance(
                start=self._utt_start, end=self._utt_start + audio.size, audio=audio,
                rms=float(np.sqrt(self._voiced_sq / n)),
                speech_prob=self._voiced_prob / max(1, self._voiced), cut=cut)))
        if not cut:
            self._history.clear()
            if self._confirmed:
                events.append(("end", end))
                self._last_end = end
                # trailing silence after the post-roll can serve as the next pre-roll
                if tail.size:
                    self._history.append(tail)
            else:
                # a discarded blip (click, breath) may still be the start of real speech
                self._history.append(np.concatenate([audio, tail]))
            self._in_speech = False
            self._clear_utterance()

    def _cut_long(self, events: list) -> None:
        """Split an over-long utterance at the quietest moment near its end."""
        audio = np.concatenate(self._utt_frames)
        cut_rel = quietest_point(audio, max(0, audio.size - int(CUT_SEARCH_S * SR)),
                                 audio.size - FRAME)
        cut = self._utt_start + cut_rel
        confirmed, voiced = self._confirmed, self._voiced
        self._finish(cut, events, cut=True)
        # the remainder continues as a new utterance with no pre-roll (it is contiguous)
        self._clear_utterance()
        self._utt_start = cut
        self._utt_frames = [audio[cut_rel:]]
        self._confirmed = confirmed
        self._voiced = min(voiced, self.min_voiced)
        rest = audio[cut_rel:]
        self._voiced_sq = float(np.dot(rest, rest)) * self._voiced * FRAME / max(1, rest.size)
        self._voiced_prob = 0.9 * self._voiced
        self._last_voiced_end = self.pos


def quietest_point(audio: np.ndarray, lo: int, hi: int) -> int:
    """Index of the centre of the lowest-energy 30 ms window in [lo, hi), 10 ms hop."""
    win, hop = int(0.03 * SR), int(0.01 * SR)
    lo = max(0, lo)
    if hi - lo <= win:
        return max(lo, min(hi, audio.size))
    seg = audio[lo:hi].astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(seg * seg)])
    starts = np.arange(0, seg.size - win, hop)
    energy = csum[starts + win] - csum[starts]
    best = int(starts[int(np.argmin(energy))])
    return lo + best + win // 2
