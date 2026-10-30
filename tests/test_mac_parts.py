"""macOS pieces that can be checked on any OS: parsing, plist output, audio conversion."""
import plistlib

import numpy as np

from minutes.engine import audio
from minutes.engine.audio_mac import ScreenCaptureAudioSource, first_channel, resample
from minutes.ui import hotkey_mac, stealth_mac, startup


def test_mac_hotkey_parse():
    p = hotkey_mac
    assert p.parse_hotkey("ctrl+alt+m") == (p.CONTROL_KEY | p.OPTION_KEY, 0x2E)
    assert p.parse_hotkey("Cmd + Shift + F9") == (p.CMD_KEY | p.SHIFT_KEY, 0x65)
    assert p.parse_hotkey("option+space") == (p.OPTION_KEY, 0x31)
    assert p.parse_hotkey("ctrl+alt") is None
    assert p.parse_hotkey("ctrl+bogus") is None


def test_launch_agent_plist(tmp_path):
    path = startup.write_launch_agent(tmp_path, python="/opt/venv/bin/python",
                                      log_path=tmp_path / "minutes.log")
    assert path.name == startup.LAUNCH_AGENT_NAME
    with open(path, "rb") as f:
        agent = plistlib.load(f)
    assert agent["Label"] == startup.LAUNCH_AGENT_LABEL
    assert agent["RunAtLoad"] is True
    assert agent["ProgramArguments"][0] == "/opt/venv/bin/python"
    assert agent["ProgramArguments"][1].endswith("app.py")
    assert agent["StandardOutPath"] == str(tmp_path / "minutes.log")


def test_first_channel_takes_channel_zero_of_planar_audio():
    left = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    right = np.ones(100, dtype=np.float32)
    raw = np.concatenate([left, right]).tobytes()
    out = first_channel(raw, 100)
    assert out.dtype == np.float32 and out.size == 100
    assert np.allclose(out, left)


def test_resample_to_16k():
    x = np.random.default_rng(0).standard_normal(48000).astype(np.float32)
    assert resample(x, 48000).size == 16000
    assert resample(x[:44100], 44100).size == 16000
    assert resample(x, 16000) is x


def test_mac_sources(monkeypatch):
    monkeypatch.setattr(audio.sys, "platform", "darwin")
    them = audio.default_sources({})["Them"]()
    assert isinstance(them, ScreenCaptureAudioSource)
    them = audio.default_sources({"speaker_device": "BlackHole 2ch"})["Them"]()
    assert isinstance(them, audio.MicSource) and them.device == "BlackHole 2ch"


def test_stealth_mac_is_safe_without_appkit():
    # On Windows/Linux there is no AppKit: helpers must fail soft, never raise.
    if not stealth_mac.available():
        assert stealth_mac.apply(0) == {"capture_excluded": False, "shell_hidden": False}
        stealth_mac.prepare_app()
