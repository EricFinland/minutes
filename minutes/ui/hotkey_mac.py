"""System-wide hotkey on macOS via Carbon RegisterEventHotKey.

Unlike an event tap, this needs no Accessibility permission. The handler is called on
the main thread by the app's event loop, so start() must be called on the main thread
after the QApplication exists. Modifier names follow the Windows spec: "ctrl" is
Control, "alt" is Option, "cmd"/"win"/"super" is Command.
"""
from __future__ import annotations

import ctypes
import logging
import threading
from typing import Callable, Optional

log = logging.getLogger(__name__)

CARBON = "/System/Library/Frameworks/Carbon.framework/Carbon"

CMD_KEY = 0x0100
SHIFT_KEY = 0x0200
OPTION_KEY = 0x0800
CONTROL_KEY = 0x1000

_MODS = {
    "cmd": CMD_KEY, "command": CMD_KEY, "win": CMD_KEY, "super": CMD_KEY,
    "shift": SHIFT_KEY, "alt": OPTION_KEY, "option": OPTION_KEY, "opt": OPTION_KEY,
    "ctrl": CONTROL_KEY, "control": CONTROL_KEY,
}

# Carbon virtual key codes (kVK_*), which follow the physical ANSI layout.
_KEYS = {
    "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06, "x": 0x07,
    "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10,
    "t": 0x11, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "=": 0x18,
    "9": 0x19, "7": 0x1A, "-": 0x1B, "8": 0x1C, "0": 0x1D, "]": 0x1E, "o": 0x1F, "u": 0x20,
    "[": 0x21, "i": 0x22, "p": 0x23, "enter": 0x24, "return": 0x24, "l": 0x25, "j": 0x26,
    "'": 0x27, "k": 0x28, ";": 0x29, "\\": 0x2A, ",": 0x2B, "/": 0x2C, "n": 0x2D, "m": 0x2E,
    ".": 0x2F, "tab": 0x30, "space": 0x31, "`": 0x32, "esc": 0x35, "escape": 0x35,
    "delete": 0x75, "home": 0x73, "end": 0x77, "pageup": 0x74, "pagedown": 0x79,
    "left": 0x7B, "right": 0x7C, "down": 0x7D, "up": 0x7E,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61, "f7": 0x62,
    "f8": 0x64, "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F, "f13": 0x69,
    "f14": 0x6B, "f15": 0x71, "f16": 0x6A, "f17": 0x40, "f18": 0x4F, "f19": 0x50,
}

K_EVENT_CLASS_KEYBOARD = int.from_bytes(b"keyb", "big")
K_EVENT_HOTKEY_PRESSED = 5
SIGNATURE = int.from_bytes(b"MNTS", "big")


def parse_hotkey(spec: str) -> Optional[tuple[int, int]]:
    """'ctrl+alt+m' -> (Carbon modifier mask, key code), or None if it has no usable key."""
    mods = 0
    key = None
    for part in (spec or "").lower().replace(" ", "").split("+"):
        if not part:
            continue
        if part in _MODS:
            mods |= _MODS[part]
        elif part in _KEYS:
            key = _KEYS[part]
        else:
            return None
    if key is None:
        return None
    return mods, key


class _EventTypeSpec(ctypes.Structure):
    _fields_ = [("eventClass", ctypes.c_uint32), ("eventKind", ctypes.c_uint32)]


class _EventHotKeyID(ctypes.Structure):
    _fields_ = [("signature", ctypes.c_uint32), ("id", ctypes.c_uint32)]


_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


class GlobalHotkey:
    HOTKEY_ID = 1
    EXISTS_ERR = -9878  # eventHotKeyExistsErr

    def __init__(self, spec: str, on_fire: Callable[[], None],
                 on_error: Optional[Callable[[str], None]] = None):
        self.spec = spec
        self.on_fire = on_fire
        self.on_error = on_error
        self.registered = threading.Event()
        self._carbon = None
        self._callback = None   # keeps the ctypes trampoline alive
        self._handler_ref = ctypes.c_void_p()
        self._hotkey_ref = ctypes.c_void_p()

    def _fail(self, msg: str) -> None:
        log.warning(msg)
        if self.on_error:
            try:
                self.on_error(msg)
            except Exception:
                log.exception("hotkey error callback failed")

    def _load(self):
        c = ctypes.CDLL(CARBON)
        c.GetApplicationEventTarget.restype = ctypes.c_void_p
        c.GetApplicationEventTarget.argtypes = []
        c.InstallEventHandler.restype = ctypes.c_int32
        # inNumTypes is an ItemCount, which is unsigned long (64-bit) on macOS.
        c.InstallEventHandler.argtypes = [ctypes.c_void_p, _HANDLER, ctypes.c_ulong,
                                          ctypes.POINTER(_EventTypeSpec), ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_void_p)]
        c.RemoveEventHandler.restype = ctypes.c_int32
        c.RemoveEventHandler.argtypes = [ctypes.c_void_p]
        c.RegisterEventHotKey.restype = ctypes.c_int32
        c.RegisterEventHotKey.argtypes = [ctypes.c_uint32, ctypes.c_uint32, _EventHotKeyID,
                                          ctypes.c_void_p, ctypes.c_uint32,
                                          ctypes.POINTER(ctypes.c_void_p)]
        c.UnregisterEventHotKey.restype = ctypes.c_int32
        c.UnregisterEventHotKey.argtypes = [ctypes.c_void_p]
        return c

    def start(self) -> None:
        if self._carbon is not None:
            return
        parsed = parse_hotkey(self.spec)
        if parsed is None:
            self._fail(f"Hotkey '{self.spec}' is not valid")
            return
        mods, key = parsed
        try:
            carbon = self._load()
        except OSError as exc:
            self._fail(f"Global hotkey unavailable: {exc}")
            return
        target = carbon.GetApplicationEventTarget()
        self._callback = _HANDLER(self._handle)
        spec = _EventTypeSpec(K_EVENT_CLASS_KEYBOARD, K_EVENT_HOTKEY_PRESSED)
        err = carbon.InstallEventHandler(target, self._callback, 1, ctypes.byref(spec),
                                         None, ctypes.byref(self._handler_ref))
        if err != 0:
            self._fail(f"Could not install the hotkey handler (error {err})")
            return
        err = carbon.RegisterEventHotKey(key, mods, _EventHotKeyID(SIGNATURE, self.HOTKEY_ID),
                                         target, 0, ctypes.byref(self._hotkey_ref))
        if err != 0:
            carbon.RemoveEventHandler(self._handler_ref)
            self._handler_ref = ctypes.c_void_p()
            if err == self.EXISTS_ERR:
                self._fail(f"Hotkey {self.spec} is in use by another app")
            elif not mods & (CMD_KEY | CONTROL_KEY):
                # macOS 15 refuses hotkeys whose only modifiers are Option and/or Shift.
                self._fail(f"Hotkey {self.spec} needs Cmd or Ctrl on this macOS "
                           f"(error {err})")
            else:
                self._fail(f"Could not register hotkey {self.spec} (error {err})")
            return
        self._carbon = carbon
        self.registered.set()

    def _handle(self, _call_ref, _event, _user) -> int:
        try:
            self.on_fire()
        except Exception:
            log.exception("hotkey handler failed")
        return 0

    def stop(self) -> None:
        carbon, self._carbon = self._carbon, None
        if carbon is None:
            return
        if self._hotkey_ref:
            carbon.UnregisterEventHotKey(self._hotkey_ref)
            self._hotkey_ref = ctypes.c_void_p()
        if self._handler_ref:
            carbon.RemoveEventHandler(self._handler_ref)
            self._handler_ref = ctypes.c_void_p()
        self.registered.clear()
