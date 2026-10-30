"""Process-wide Whisper model, shared by live transcription and the polish pass."""
from __future__ import annotations

import threading

NUM_WORKERS = 2  # lets the Me and Them streams decode at the same time

_lock = threading.Lock()
_model = None
_key: tuple | None = None


def _key_for(cfg: dict) -> tuple:
    return (str(cfg.get("whisper_model", "large-v3-turbo")), int(cfg.get("cpu_threads", 8)))


def get_model(cfg: dict):
    """Return the shared faster_whisper.WhisperModel, loading it on first use.

    Callers block while another thread is loading. If the configured model or thread
    count changed, the old model is replaced (it is freed once nobody holds it).
    """
    global _model, _key
    key = _key_for(cfg)
    with _lock:
        if _model is None or _key != key:
            from faster_whisper import WhisperModel
            name, threads = key
            _model = None
            kwargs = dict(device="cpu", compute_type="int8",
                          cpu_threads=threads, num_workers=NUM_WORKERS)
            try:
                # Stay offline once the model is cached; otherwise every launch
                # asks Hugging Face for the latest revision.
                _model = WhisperModel(name, local_files_only=True, **kwargs)
            except Exception:
                _model = WhisperModel(name, **kwargs)  # first run: download once
            _key = key
        return _model


def model_loaded(cfg: dict) -> bool:
    with _lock:
        return _model is not None and _key == _key_for(cfg)
