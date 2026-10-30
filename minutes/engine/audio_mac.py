"""System audio on macOS through ScreenCaptureKit (macOS 13 or newer).

macOS has no loopback device, so the call audio ("Them") comes from ScreenCaptureKit,
which can capture everything the Mac plays without installing a driver. It needs the
Screen Recording permission once (System Settings > Privacy & Security). A tiny 2x2
video stream at 1 fps is part of the deal; its frames are ignored.

Buffers arrive on a ScreenCaptureKit queue. They are handed to a pump thread so the
recorder callback never runs on Apple's queue, and silence is synthesized from the
wall clock when nothing is playing, same as the Windows loopback source.
"""
from __future__ import annotations

import collections
import logging
import threading
import time

import numpy as np

from .audio import SR, OnAudio, OnStatus

log = logging.getLogger(__name__)

PERMISSION_HINT = ("Allow screen and system audio recording for Minutes in System Settings > "
                   "Privacy & Security, then restart Minutes.")

_output_class = None


def _call_async(start, timeout: float = 10.0) -> tuple:
    """Run an Objective-C API that reports through a completion handler; return its args."""
    done = threading.Event()
    box: dict = {}

    def handler(*args):
        box["args"] = args
        done.set()

    start(handler)
    if not done.wait(timeout):
        raise RuntimeError("timed out waiting for ScreenCaptureKit")
    return box["args"]


K_AUDIO_FORMAT_FLAG_NON_INTERLEAVED = 1 << 5


def first_channel(raw: bytes, frames: int, channels: int = 1,
                  interleaved: bool = False) -> np.ndarray:
    """ScreenCaptureKit audio is float32 and normally non-interleaved: keep channel 0."""
    x = np.frombuffer(raw, dtype=np.float32)
    if interleaved and channels > 1:
        x = x[::channels]
    if 0 < frames < x.size:
        x = x[:frames]
    return x.astype(np.float32, copy=True)


def _stream_format(CM, sbuf) -> tuple[float, int, bool]:
    """(sample rate, channels, interleaved) of an audio sample buffer.

    PyObjC returns AudioStreamBasicDescription as a plain tuple unless the CoreAudio
    bindings registered the struct, so read it by position: mSampleRate is field 0,
    mFormatFlags field 2, mChannelsPerFrame field 6.
    """
    try:
        desc = CM.CMSampleBufferGetFormatDescription(sbuf)
        asbd = CM.CMAudioFormatDescriptionGetStreamBasicDescription(desc) if desc else None
        if asbd is None:
            return float(SR), 1, False
        rate = float(asbd[0]) or float(SR)
        flags = int(asbd[2])
        channels = max(1, int(asbd[6]))
        return rate, channels, not flags & K_AUDIO_FORMAT_FLAG_NON_INTERLEAVED
    except Exception:
        log.debug("could not read the audio format", exc_info=True)
        return float(SR), 1, False


def resample(x: np.ndarray, rate: float) -> np.ndarray:
    """Bring audio to 16 kHz if the system ignored the requested sample rate."""
    if not rate or abs(rate - SR) < 1 or x.size == 0:
        return x
    factor = rate / SR
    if abs(factor - round(factor)) < 1e-6:
        f = int(round(factor))
        n = x.size // f * f
        return x[:n].reshape(-1, f).mean(axis=1).astype(np.float32)
    n_out = int(x.size / factor)
    return np.interp(np.arange(n_out) * factor, np.arange(x.size), x).astype(np.float32)


def _make_output_class():
    """Define the stream output/delegate class once per process (ObjC class names are global)."""
    global _output_class
    if _output_class is not None:
        return _output_class
    import objc
    from Foundation import NSObject

    class MinutesStreamOutput(NSObject, protocols=[objc.protocolNamed("SCStreamOutput"),
                                                   objc.protocolNamed("SCStreamDelegate")]):
        def stream_didOutputSampleBuffer_ofType_(self, stream, sbuf, kind):
            if self.on_buffer is not None:
                try:
                    self.on_buffer(sbuf, kind)
                except Exception:
                    log.exception("system audio buffer failed")

        def stream_didStopWithError_(self, stream, error):
            if self.on_stop is not None:
                self.on_stop(str(error))

    _output_class = MinutesStreamOutput
    return _output_class


class ScreenCaptureAudioSource:
    """Everything the Mac plays, excluding Minutes' own sounds."""

    realtime = True
    POLL_S = 0.05
    SILENCE_AFTER_S = 0.2

    def __init__(self, cfg: dict) -> None:
        self.name = "system audio"
        self._stream = None
        self._output = None
        self._pump: threading.Thread | None = None
        self._stop = threading.Event()
        self._chunks: collections.deque = collections.deque()
        self._dead = False

    def open(self, on_audio: OnAudio, on_status: OnStatus) -> None:
        import CoreMedia as CM
        import ScreenCaptureKit as SCK

        content, error = _call_async(
            lambda h: SCK.SCShareableContent.getShareableContentWithCompletionHandler_(h))
        if error is not None or content is None or not content.displays():
            raise PermissionError(f"Cannot capture system audio. {PERMISSION_HINT} ({error})")

        content_filter = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            content.displays()[0], [])
        conf = SCK.SCStreamConfiguration.alloc().init()
        conf.setCapturesAudio_(True)
        conf.setExcludesCurrentProcessAudio_(True)
        conf.setSampleRate_(SR)
        conf.setChannelCount_(1)
        conf.setWidth_(2)
        conf.setHeight_(2)
        conf.setMinimumFrameInterval_(CM.CMTimeMake(1, 1))

        self._dead = False
        self._stop.clear()
        self._chunks.clear()
        output = _make_output_class().alloc().init()
        output.on_buffer = lambda sbuf, kind: self._on_buffer(CM, SCK, sbuf, kind)
        output.on_stop = self._on_stopped
        self._output = output

        stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, conf, output)
        queue = self._queue()
        for kind in (SCK.SCStreamOutputTypeScreen, SCK.SCStreamOutputTypeAudio):
            ok, err = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                output, kind, queue, None)
            if not ok:
                raise RuntimeError(f"Cannot attach to the capture stream: {err}")
        (err,) = _call_async(lambda h: stream.startCaptureWithCompletionHandler_(h))
        if err is not None:
            raise PermissionError(f"Cannot capture system audio. {PERMISSION_HINT} ({err})")
        self._stream = stream
        self._pump = threading.Thread(target=self._run_pump, args=(on_audio,),
                                      name="minutes-sck-pump", daemon=True)
        self._pump.start()

    @staticmethod
    def _queue():
        try:
            from dispatch import dispatch_queue_create
            return dispatch_queue_create(b"minutes.system-audio", None)
        except Exception:
            return None  # ScreenCaptureKit falls back to its own queue

    def _on_buffer(self, CM, SCK, sbuf, kind) -> None:
        if kind != SCK.SCStreamOutputTypeAudio or not CM.CMSampleBufferIsValid(sbuf):
            return
        frames = int(CM.CMSampleBufferGetNumSamples(sbuf))
        block = CM.CMSampleBufferGetDataBuffer(sbuf)
        if block is None or frames <= 0:
            return
        length = int(CM.CMBlockBufferGetDataLength(block))
        status, raw = CM.CMBlockBufferCopyDataBytes(block, 0, length, None)
        if status != 0 or not raw:
            return
        rate, channels, interleaved = _stream_format(CM, sbuf)
        x = first_channel(bytes(raw), frames, channels, interleaved)
        self._chunks.append(resample(x, rate))

    def _on_stopped(self, reason: str) -> None:
        log.warning("system audio stream stopped: %s", reason)
        self._dead = True

    def _run_pump(self, on_audio: OnAudio) -> None:
        last = time.monotonic()
        while not self._stop.wait(self.POLL_S):
            parts = []
            while self._chunks:
                parts.append(self._chunks.popleft())
            now = time.monotonic()
            if parts:
                on_audio(np.concatenate(parts), now)
                last = now
            elif now - last > self.SILENCE_AFTER_S:
                n = int((now - last) * SR)
                on_audio(np.zeros(n, np.float32), now)
                last += n / SR

    def alive(self) -> bool:
        return self._stream is not None and not self._dead

    def close(self) -> None:
        self._stop.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                _call_async(lambda h: stream.stopCaptureWithCompletionHandler_(h), timeout=3.0)
            except Exception:
                log.debug("stopping capture failed", exc_info=True)
        if self._output is not None:
            self._output.on_buffer = None
            self._output.on_stop = None
        pump, self._pump = self._pump, None
        if pump is not None and pump is not threading.current_thread():
            pump.join(timeout=3.0)
