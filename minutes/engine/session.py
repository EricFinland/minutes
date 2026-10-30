"""One meeting on disk: folder, append-only lines, transcript, meta and the two WAVs."""
from __future__ import annotations

import os
import struct
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from minutes import models
from minutes.models import Line

SR = 16000
_SILENCE_BLOCK = np.zeros(SR * 10, dtype="<i2").tobytes()


class WavWriter:
    """16 kHz mono int16 WAV that stays readable if the process dies mid-write.

    The header is written up front with zero sizes and patched by `sync()`; a reader
    after a crash sees everything up to the last sync, and the samples after it are
    still on disk for tools that trust the file length.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames = 0
        self._f = open(path, "wb")
        self._f.write(self._header(0))
        self._f.flush()

    @staticmethod
    def _header(data_bytes: int) -> bytes:
        return (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVE"
                + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16)
                + b"data" + struct.pack("<I", data_bytes))

    def write(self, samples: np.ndarray) -> None:
        if samples.size:
            pcm = np.clip(samples, -1.0, 1.0) * 32767.0
            self._f.write(pcm.astype("<i2").tobytes())
            self.frames += samples.size

    def write_silence(self, n: int) -> None:
        self.frames += max(0, n)
        while n > 0:
            k = min(n, SR * 10)
            self._f.write(_SILENCE_BLOCK[:k * 2])
            n -= k

    def sync(self) -> None:
        if self._f.closed:
            return
        data_bytes = self.frames * 2
        self._f.flush()
        self._f.seek(4)
        self._f.write(struct.pack("<I", 36 + data_bytes))
        self._f.seek(40)
        self._f.write(struct.pack("<I", data_bytes))
        self._f.seek(0, os.SEEK_END)
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        if not self._f.closed:
            self.sync()
            self._f.close()


def _unique_dir(root: Path, started: datetime) -> Path:
    base = started.strftime("%Y-%m-%d_%H%M%S")
    d = root / base
    n = 2
    while d.exists():
        d = root / f"{base}_{n}"
        n += 1
    d.mkdir(parents=True)
    return d


class Session:
    """Files for one meeting. All methods are thread-safe."""

    def __init__(self, root: Path, started: datetime, model: str) -> None:
        self._lock = threading.Lock()
        self.dir = _unique_dir(root, started)
        self.started = started
        self.closed = False
        self.had_speech = False
        self._lines: list[Line] = []
        self.meta = {"started": started.isoformat(timespec="seconds"), "ended": None,
                     "model": model, "audio_spans": []}
        models.write_meta(self.dir, self.meta)
        (self.dir / models.LINES_FILE).touch()
        self._write_transcript()
        self.wavs = {spk: WavWriter(self.dir / name) for spk, name in models.AUDIO_FILES.items()}

    # ------------------------------------------------------------ transcript

    def add_line(self, line: Line) -> None:
        with self._lock:
            models.append_line(self.dir / models.LINES_FILE, line)
            self._lines.append(line)
            self._write_transcript()

    def lines(self) -> list[Line]:
        with self._lock:
            return sorted(self._lines, key=lambda l: l.start)

    def _write_transcript(self) -> None:
        text = models.format_transcript(self._lines, models.meeting_title(self.dir))
        models.write_text_atomic(self.dir / models.TRANSCRIPT_FILE, text)

    # ------------------------------------------------------------ audio

    def audio_frames(self, speaker: str) -> int:
        with self._lock:
            return self.wavs[speaker].frames

    def write_audio(self, speaker: str, samples: np.ndarray, at: int) -> None:
        """Write samples so the first lands at WAV sample index `at`.

        A gap before `at` is filled with silence; samples before the current end of the
        file (already written) are skipped, so replays and overlaps are harmless.
        """
        with self._lock:
            if self.closed:
                return
            w = self.wavs[speaker]
            if at > w.frames:
                w.write_silence(at - w.frames)
            skip = w.frames - at
            if skip < samples.size:
                w.write(samples[max(0, skip):])

    def align_audio(self) -> int:
        """Pad the shorter WAV with silence so both share one timeline. Returns length."""
        with self._lock:
            n = max(w.frames for w in self.wavs.values())
            if self.closed:
                return n
            for w in self.wavs.values():
                w.write_silence(n - w.frames)
            return n

    def begin_span(self, meeting_t: float) -> int:
        """Start a recording span. Returns its first sample index in the WAV files."""
        file_pos = self.align_audio()
        with self._lock:
            self.meta["audio_spans"].append(
                {"meeting_t": round(meeting_t, 3), "file_t": round(file_pos / SR, 3), "dur": 0.0})
            models.write_meta(self.dir, self.meta)
        return file_pos

    def set_span_duration(self, dur: float) -> None:
        with self._lock:
            if self.meta["audio_spans"]:
                self.meta["audio_spans"][-1]["dur"] = round(max(0.0, dur), 3)

    def sync(self) -> None:
        """Make WAV headers and meta match what has been written so far."""
        with self._lock:
            if self.closed:
                return
            for w in self.wavs.values():
                w.sync()
            models.write_meta(self.dir, self.meta)

    def close(self) -> None:
        self.align_audio()
        with self._lock:
            if self.closed:
                return
            self.closed = True
            for w in self.wavs.values():
                w.close()
            self.meta["ended"] = datetime.now().astimezone().isoformat(timespec="seconds")
            models.write_meta(self.dir, self.meta)
