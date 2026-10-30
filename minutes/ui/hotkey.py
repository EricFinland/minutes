"""System-wide hotkey via Win32 RegisterHotKey on a daemon thread.

Works while another app (Zoom, a browser) has focus, unlike a Qt shortcut. The
callback runs on the hotkey thread, so pass something thread safe such as a Qt
signal's emit.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
from typing import Callable, Optional

log = logging.getLogger(__name__)

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

_MODS = {
    "alt": MOD_ALT, "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
    "shift": MOD_SHIFT, "win": MOD_WIN, "super": MOD_WIN,
}
_VK_SPECIAL = {
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B,
    "insert": 0x2D, "delete": 0x2E, "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "`": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, "\\": 0xDC,
    ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
}
_VK_SPECIAL.update({f"f{i}": 0x6F + i for i in range(1, 25)})


def parse_hotkey(spec: str) -> Optional[tuple[int, int]]:
    """'ctrl+alt+m' -> (modifier mask, virtual key), or None if it has no usable key."""
    mods = 0
    vk = None
    for part in (spec or "").lower().replace(" ", "").split("+"):
        if not part:
            continue
        if part in _MODS:
            mods |= _MODS[part]
        elif part in _VK_SPECIAL:
            vk = _VK_SPECIAL[part]
        elif len(part) == 1 and part.isalnum() and part.isascii():
            vk = ord(part.upper())
        else:
            return None
    if vk is None:
        return None
    return mods, vk


class GlobalHotkey:
    HOTKEY_ID = 0x4D31

    def __init__(self, spec: str, on_fire: Callable[[], None],
                 on_error: Optional[Callable[[str], None]] = None):
        self.spec = spec
        self.on_fire = on_fire
        self.on_error = on_error
        self._thread: Optional[threading.Thread] = None
        self._thread_id = 0
        self.registered = threading.Event()

    def start(self) -> None:
        if sys.platform != "win32" or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="minutes-hotkey", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        self._thread_id = 0

    def _fail(self, msg: str) -> None:
        log.warning(msg)
        if self.on_error:
            try:
                self.on_error(msg)
            except Exception:
                log.exception("hotkey error callback failed")

    def _run(self) -> None:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        parsed = parse_hotkey(self.spec)
        if parsed is None:
            self._fail(f"Hotkey '{self.spec}' is not valid")
            return
        mods, vk = parsed
        self._thread_id = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, self.HOTKEY_ID, mods | MOD_NOREPEAT, vk):
            self._thread_id = 0
            self._fail(f"Hotkey {self.spec} is in use by another app")
            return
        self.registered.set()
        msg = wintypes.MSG()
        try:
            while True:
                r = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r == 0 or r == -1:
                    break
                if msg.message == WM_HOTKEY and msg.wParam == self.HOTKEY_ID:
                    try:
                        self.on_fire()
                    except Exception:
                        log.exception("hotkey handler failed")
        finally:
            user32.UnregisterHotKey(None, self.HOTKEY_ID)


if sys.platform == "darwin":
    from .hotkey_mac import GlobalHotkey  # noqa: F401,F811
