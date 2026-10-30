"""macOS counterparts of the Win32 stealth helpers.

Qt's winId() on macOS is an NSView pointer; its NSWindow gets sharingType "none", which
keeps it out of screenshots and most screen sharing. Note: from macOS 15 some capture
tools built on ScreenCaptureKit no longer honor sharingType, so treat this as best effort.
The app runs as an "accessory" so it has no Dock icon and is not in Cmd-Tab.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

NS_WINDOW_SHARING_NONE = 0
NS_APP_ACTIVATION_POLICY_ACCESSORY = 1
# canJoinAllSpaces | stationary | ignoresCycle | fullScreenAuxiliary: visible on every
# Space and over full-screen Zoom, skipped by Cmd-` window cycling.
_COLLECTION_BEHAVIOR = (1 << 0) | (1 << 4) | (1 << 6) | (1 << 8)
# Bits AppKit treats as mutually exclusive with the ones above (moveToActiveSpace,
# managed, transient, participatesInCycle, fullScreenPrimary, fullScreenNone). Qt sets
# some of them itself, and AppKit raises if both members of a pair are set.
_COLLECTION_CONFLICTS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7) | (1 << 9)


def available() -> bool:
    try:
        import AppKit  # noqa: F401
        import objc  # noqa: F401
        return True
    except ImportError:
        return False


def _window(win_id: int):
    import objc
    if not win_id:
        return None
    view = objc.objc_object(c_void_p=win_id)
    return view.window() if view is not None else None


def exclude_from_capture(win_id: int) -> bool:
    try:
        w = _window(win_id)
        if w is None:
            return False
        w.setSharingType_(NS_WINDOW_SHARING_NONE)
        return int(w.sharingType()) == NS_WINDOW_SHARING_NONE
    except Exception:
        log.warning("could not exclude window from capture", exc_info=True)
        return False


def hide_from_shell(win_id: int) -> bool:
    try:
        w = _window(win_id)
        if w is None:
            return False
        behavior = int(w.collectionBehavior()) & ~_COLLECTION_CONFLICTS
        w.setCollectionBehavior_(behavior | _COLLECTION_BEHAVIOR)
        w.setHidesOnDeactivate_(False)  # stay up while Zoom is the active app
        return True
    except Exception:
        log.warning("could not set window behavior", exc_info=True)
        return False


def apply(win_id: int, no_activate: bool | None = None) -> dict[str, bool]:
    return {
        "capture_excluded": exclude_from_capture(win_id),
        "shell_hidden": hide_from_shell(win_id),
    }


def prepare_app() -> None:
    """No Dock icon, no Cmd-Tab entry. Call once after the QApplication exists."""
    try:
        from AppKit import NSApplication
        NSApplication.sharedApplication().setActivationPolicy_(NS_APP_ACTIVATION_POLICY_ACCESSORY)
    except Exception:
        log.warning("could not hide the Dock icon", exc_info=True)
