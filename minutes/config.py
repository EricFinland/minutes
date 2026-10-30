"""Settings and paths. User overrides live in <data_dir>/settings.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULTS: dict = {
    # Speech to text
    "whisper_model": "large-v3-turbo",   # fallback: "distil-large-v3" (faster, English only)
    "language": "en",
    # 8 on a big Windows laptop; leaves a couple of cores for Zoom on smaller Macs.
    "cpu_threads": max(2, min(8, (os.cpu_count() or 4) - 2)),
    "beam_size": 5,
    # Voice activity detection
    "vad_threshold": 0.35,
    "pre_roll_ms": 400,
    "post_roll_ms": 300,
    "end_silence_ms": 700,
    "max_utterance_s": 25,
    # Audio sources (None = system default)
    "mic_device": None,
    "speaker_device": None,
    # Meetings
    "record_on_launch": True,
    "meeting_gap_minutes": 15,
    "keep_audio_days": 7,
    # Summary
    "ollama_url": "http://localhost:11434",
    "summary_model": "llama3:latest",
    # UI
    "hotkey_toggle": "ctrl+alt+m",
    "window_pos": None,
    "expanded": False,
}


def data_dir() -> Path:
    override = os.environ.get("MINUTES_DATA_DIR")
    d = Path(override) if override else Path.home() / "Minutes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def meetings_dir() -> Path:
    d = data_dir() / "meetings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _settings_path() -> Path:
    return data_dir() / "settings.json"


def load() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(_settings_path(), encoding="utf-8") as f:
            user = json.load(f)
        if isinstance(user, dict):
            cfg.update({k: v for k, v in user.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return cfg


def save(cfg: dict) -> None:
    """Persist only values that differ from the defaults."""
    diff = {k: v for k, v in cfg.items() if k in DEFAULTS and DEFAULTS[k] != v}
    path = _settings_path()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2)
    os.replace(tmp, path)
