"""Win32 helpers that keep a window out of screen capture, the taskbar and Alt-Tab.

All functions take a raw HWND (int) and are no-ops returning False off Windows, so
the UI can still run under Qt's offscreen platform in tests.
"""
from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000
WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11

_SWP_FLAGS = 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020  # NOSIZE|NOMOVE|NOZORDER|NOACTIVATE|FRAMECHANGED

_user32 = None
if sys.platform == "win32":
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    _user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    _user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    _user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    _user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    _user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
    _user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.SetWindowPos.restype = wintypes.BOOL
    _user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, wintypes.UINT]
    _user32.IsWindow.restype = wintypes.BOOL
    _user32.IsWindow.argtypes = [wintypes.HWND]


def available() -> bool:
    return _user32 is not None


def _valid(hwnd: int) -> bool:
    return bool(_user32 and hwnd and _user32.IsWindow(hwnd))


def get_exstyle(hwnd: int) -> int:
    if not _valid(hwnd):
        return 0
    return int(_user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE))


def _set_exstyle(hwnd: int, style: int) -> None:
    _user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
    _user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, _SWP_FLAGS)


def get_affinity(hwnd: int) -> int:
    if not _valid(hwnd):
        return -1
    value = wintypes.DWORD(0)
    if not _user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(value)):
        return -1
    return int(value.value)


def exclude_from_capture(hwnd: int) -> bool:
    """Hide the window from screen sharing, screenshots and recordings (Win10 2004+)."""
    if not _valid(hwnd):
        return False
    if get_affinity(hwnd) == WDA_EXCLUDEFROMCAPTURE:
        return True
    if _user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
        return True
    log.warning("SetWindowDisplayAffinity failed (error %s)", ctypes.get_last_error())
    return False


def hide_from_shell(hwnd: int) -> bool:
    """Tool window style: no taskbar button, not in Alt-Tab."""
    if not _valid(hwnd):
        return False
    style = get_exstyle(hwnd)
    wanted = (style & ~WS_EX_APPWINDOW) | WS_EX_TOOLWINDOW
    if wanted != style:
        _set_exstyle(hwnd, wanted)
    return True


def set_no_activate(hwnd: int, on: bool) -> bool:
    """Clicks on a WS_EX_NOACTIVATE window do not take focus away from other apps."""
    if not _valid(hwnd):
        return False
    style = get_exstyle(hwnd)
    wanted = (style | WS_EX_NOACTIVATE) if on else (style & ~WS_EX_NOACTIVATE)
    if wanted != style:
        _set_exstyle(hwnd, wanted)
    return True


def apply(hwnd: int, no_activate: bool | None = None) -> dict[str, bool]:
    status = {
        "capture_excluded": exclude_from_capture(hwnd),
        "shell_hidden": hide_from_shell(hwnd),
    }
    if no_activate is not None:
        status["no_activate"] = set_no_activate(hwnd, no_activate)
    return status
