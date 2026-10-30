"""Audio sources: microphone, system audio (WASAPI loopback) and WAV replay for tests.

Every source delivers 16 kHz mono float32 chunks through `on_audio(samples, t)` where `t`
is `time.monotonic()` when the chunk finished (None for sources that are not real time).
`open()` raises if the device cannot be opened; `alive()` turning False means the
source died and should be reopened by the caller.
"""
from __future__ import annotations

import ctypes
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

SR = 16000
BLOCK = SR // 10  # 100 ms

OnAudio = Callable[[np.ndarray, "float | None"], None]
OnStatus = Callable[[str, str], None]   # (text, level)


class AudioSource(Protocol):
    name: str
    realtime: bool

    def open(self, on_audio: OnAudio, on_status: OnStatus) -> None: ...
    def alive(self) -> bool: ...
    def close(self) -> None: ...


def _decimate(x: np.ndarray, factor: int) -> np.ndarray:
    """Integer-ratio downsampling with a box filter; only used if 16 kHz is refused."""
    n = x.size // factor * factor
    return x[:n].reshape(-1, factor).mean(axis=1).astype(np.float32)


class MicSource:
    """Default (or configured) input device through PortAudio."""

    realtime = True

    def __init__(self, cfg: dict) -> None:
        self.device = cfg.get("mic_device")
        self.name = "microphone"
        self._stream = None
        self._failed = False

    def open(self, on_audio: OnAudio, on_status: OnStatus) -> None:
        import sounddevice as sd

        info = sd.query_devices(self.device, kind="input")
        self.name = str(info["name"])
        factor = 1

        def callback(indata, frames, time_info, status):
            x = indata[:, 0] if indata.ndim == 2 else indata
            x = np.array(x, dtype=np.float32)
            on_audio(_decimate(x, factor) if factor > 1 else x, time.monotonic())

        def finished():
            self._failed = True

        try:
            self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                          blocksize=BLOCK, device=self.device,
                                          callback=callback, finished_callback=finished)
        except Exception:
            native = int(info["default_samplerate"])
            if native % SR:
                raise
            factor = native // SR
            self._stream = sd.InputStream(samplerate=native, channels=1, dtype="float32",
                                          blocksize=BLOCK * factor, device=self.device,
                                          callback=callback, finished_callback=finished)
        self._failed = False
        self._stream.start()

    def alive(self) -> bool:
        return self._stream is not None and self._stream.active and not self._failed

    def close(self) -> None:
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.abort()
                s.close()
            except Exception:
                pass


class LoopbackSource:
    """What the computer is playing, via WASAPI loopback of the output device.

    soundcard's own `record()` spins on a ~1 ms sleep while waiting for data, which costs
    several percent of a core during silence. Instead we give WASAPI a 1 s buffer and
    drain it every 50 ms. When nothing is playing, WASAPI delivers no packets at all, so
    silence is synthesized from the wall clock to keep the timeline continuous.
    """

    realtime = True
    POLL_S = 0.05
    DEVICE_CHECK_S = 3.0
    SILENCE_AFTER_S = 0.2

    def __init__(self, cfg: dict) -> None:
        self.speaker_name = cfg.get("speaker_device")
        self.name = "system audio"
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._dead = False

    def open(self, on_audio: OnAudio, on_status: OnStatus) -> None:
        opened = threading.Event()
        error: list[BaseException] = []
        self._stop.clear()
        self._dead = False
        self._thread = threading.Thread(
            target=self._run, args=(on_audio, on_status, opened, error),
            name="minutes-loopback", daemon=True)
        self._thread.start()
        if not opened.wait(10.0):
            self.close()
            raise RuntimeError("timed out opening system audio")
        if error:
            self.close()
            raise error[0]

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._dead

    def close(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3.0)

    # ------------------------------------------------------------ capture thread

    def _speaker(self, sc):
        if self.speaker_name:
            return sc.get_speaker(self.speaker_name)
        return sc.default_speaker()

    def _run(self, on_audio, on_status, opened, error) -> None:
        hr = None
        first = True
        try:
            # soundcard initializes COM on the thread that first imports it and refuses to
            # import on a thread where COM is already up, so import before our own init.
            import soundcard as sc
            # WASAPI needs COM on the capturing thread, otherwise 0x800401f0.
            hr = ctypes.windll.ole32.CoInitializeEx(None, 0)
            while not self._stop.is_set():
                try:
                    speaker = self._speaker(sc)
                    mic = sc.get_microphone(speaker.id, include_loopback=True)
                    with mic.recorder(samplerate=SR, channels=1, blocksize=SR) as rec:
                        self.name = speaker.name
                        if first:
                            opened.set()
                            first = False
                        else:
                            on_status(f"System audio now from {speaker.name}", "info")
                        changed = self._capture(sc, rec, speaker.id, on_audio)
                    if not changed:
                        return
                except Exception as exc:
                    if first:
                        error.append(exc)
                    return
        except Exception as exc:
            error.append(exc)
        finally:
            opened.set()
            self._dead = True
            if hr in (0, 1):
                ctypes.windll.ole32.CoUninitialize()

    def _capture(self, sc, rec, speaker_id: str, on_audio: OnAudio) -> bool:
        """Read until stopped (returns False) or the default output changes (True)."""
        fast = hasattr(rec, "_capture_available_frames") and hasattr(rec, "_record_chunk")
        last = time.monotonic()
        next_check = last + self.DEVICE_CHECK_S
        while not self._stop.is_set():
            if fast:
                self._stop.wait(self.POLL_S)
                parts = []
                while rec._capture_available_frames() > 0:
                    parts.append(rec._record_chunk())
                now = time.monotonic()
                if parts:
                    on_audio(np.concatenate(parts).astype(np.float32, copy=False), now)
                    last = now
                elif now - last > self.SILENCE_AFTER_S:
                    n = int((now - last) * SR)
                    on_audio(np.zeros(n, np.float32), now)
                    last += n / SR
            else:
                data = rec.record(numframes=BLOCK)
                now = time.monotonic()
                on_audio(np.asarray(data, np.float32).reshape(-1), now)
            if not self.speaker_name and now >= next_check:
                next_check = now + self.DEVICE_CHECK_S
                try:
                    if sc.default_speaker().id != speaker_id:
                        return True
                except Exception:
                    pass
        return False


class FileSource:
    """Replays audio (a WAV path or float32 array) as a source. For tests and tools.

    speed: 1.0 is real time, 4.0 is four times faster, 0 is as fast as possible.
    """

    realtime = False

    def __init__(self, audio: "str | Path | np.ndarray", speed: float = 0.0,
                 chunk_s: float = 0.1, gain: float = 1.0) -> None:
        if isinstance(audio, (str, Path)):
            audio = read_wav(Path(audio))
        self.audio = np.asarray(audio, np.float32) * gain
        self.speed = speed
        self.chunk = max(1, int(chunk_s * SR))
        self.name = "file"
        self.finished = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self, on_audio: OnAudio, on_status: OnStatus) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(on_audio,), daemon=True,
                                        name="minutes-filesource")
        self._thread.start()

    def _run(self, on_audio: OnAudio) -> None:
        t0 = time.monotonic()
        for i in range(0, self.audio.size, self.chunk):
            if self._stop.is_set():
                return
            if self.speed > 0:
                due = t0 + (i / SR) / self.speed
                delay = due - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    return
            on_audio(self.audio[i:i + self.chunk], None)
        self.finished.set()

    def alive(self) -> bool:
        return True  # stays "connected" (silent) after the file ends

    def close(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3.0)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != SR:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def default_sources(cfg: dict) -> dict[str, Callable[[], AudioSource]]:
    from minutes.models import ME, THEM
    return {ME: lambda: MicSource(cfg), THEM: lambda: LoopbackSource(cfg)}
