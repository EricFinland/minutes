"""End to end: run `app.py --fake` as its own process and inspect its window from here.

- display affinity is WDA_EXCLUDEFROMCAPTURE (0x11) and the window is a tool window
  (no taskbar button, not in Alt-Tab)
- capture APIs get nothing: PrintWindow returns an all black image for Minutes, while
  the same app with the exclusion switched off (positive control) renders its UI
- a real screenshot does not contain it: a solid magenta window is placed directly
  underneath, and the screenshot of that area must be magenta. Skipped while the
  session is locked, because then the lock screen covers everything
- a second launch exits and leaves the first one running
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).with_name("python.exe")

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x80
WS_EX_APPWINDOW = 0x40000
WDA_EXCLUDEFROMCAPTURE = 0x11
PW_RENDERFULLCONTENT = 2
GA_ROOT = 2

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.WindowFromPoint.restype = wintypes.HWND
user32.WindowFromPoint.argtypes = [wintypes.POINT]
user32.GetAncestor.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]

try:
    user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # physical pixels everywhere
except Exception:
    pass

SETTINGS = {"expanded": True, "window_pos": [160, 160, 380, 460]}
NO_STEALTH = (
    "import sys; sys.path.insert(0, {root!r}); "
    "import minutes.ui.stealth as st; st.exclude_from_capture = lambda hwnd: False; "
    "import app; sys.exit(app.main(['--fake']))"
)


def find_window(pid: int, timeout: float = 25.0) -> int:
    import win32gui
    import win32process

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        found = []

        def cb(hwnd, _):
            if (win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == "Minutes"
                    and win32process.GetWindowThreadProcessId(hwnd)[1] == pid):
                found.append(hwnd)
            return True

        win32gui.EnumWindows(cb, None)
        if found:
            return found[0]
        time.sleep(0.2)
    return 0


def launch(tmp_path: Path, name: str, args: list[str]):
    data = tmp_path / name
    data.mkdir()
    (data / "settings.json").write_text(json.dumps(SETTINGS), encoding="utf-8")
    env = dict(os.environ, MINUTES_DATA_DIR=str(data), PYTHONIOENCODING="utf-8")
    env.pop("QT_QPA_PLATFORM", None)
    log = open(tmp_path / f"{name}.log", "w", encoding="utf-8")
    proc = subprocess.Popen([str(PYTHON), *args], cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT)
    hwnd = find_window(proc.pid)
    if not hwnd:
        proc.kill()
        log.close()
        pytest.fail((tmp_path / f"{name}.log").read_text(encoding="utf-8"))
    time.sleep(1.5)  # first paint plus a few fake events
    return proc, hwnd, env, log


@pytest.fixture(scope="module")
def minutes_app(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("stealth")
    proc, hwnd, env, log = launch(tmp, "app", [str(ROOT / "app.py"), "--fake"])
    yield proc, hwnd, env, tmp
    proc.kill()
    proc.wait(10)
    log.close()


def print_window(hwnd: int):
    import win32gui
    import win32ui
    from PIL import Image

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w, h = right - left, bottom - top
    hdc = win32gui.GetWindowDC(hwnd)
    src = win32ui.CreateDCFromHandle(hdc)
    mem = src.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(src, w, h)
    mem.SelectObject(bmp)
    try:
        assert user32.PrintWindow(hwnd, mem.GetSafeHdc(), PW_RENDERFULLCONTENT)
        info = bmp.GetInfo()
        return Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]),
                                bmp.GetBitmapBits(True), "raw", "BGRX", 0, 1)
    finally:
        win32gui.DeleteObject(bmp.GetHandle())
        mem.DeleteDC()
        src.DeleteDC()
        win32gui.ReleaseDC(hwnd, hdc)


def fraction(img, pred) -> float:
    px = img.load()
    w, h = img.size
    hits = total = 0
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            total += 1
            hits += bool(pred(px[x, y][:3]))
    return hits / max(1, total)


def is_black(p):
    return p == (0, 0, 0)


def is_light(p):
    return sum(p) > 450  # text and button labels


def is_magenta(p):
    return p[0] > 200 and p[1] < 60 and p[2] > 200


def test_affinity_and_tool_window(minutes_app):
    _, hwnd, _, _ = minutes_app
    v = wintypes.DWORD(0)
    assert user32.GetWindowDisplayAffinity(hwnd, ctypes.byref(v))
    assert v.value == WDA_EXCLUDEFROMCAPTURE
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    assert style & WS_EX_TOOLWINDOW
    assert not style & WS_EX_APPWINDOW
    assert user32.GetForegroundWindow() != hwnd  # showing it did not take focus


def test_capture_api_gets_blank_window(minutes_app, tmp_path):
    _, hwnd, _, _ = minutes_app
    shot = print_window(hwnd)
    shot.save(tmp_path / "printwindow_minutes.png")
    assert fraction(shot, is_black) > 0.99
    assert fraction(shot, is_light) == 0

    # Positive control: identical app, exclusion switched off, renders its UI.
    proc, ctrl_hwnd, _, log = launch(tmp_path, "control",
                                     ["-c", NO_STEALTH.format(root=str(ROOT))])
    try:
        v = wintypes.DWORD(0)
        user32.GetWindowDisplayAffinity(ctrl_hwnd, ctypes.byref(v))
        assert v.value == 0
        ctrl = print_window(ctrl_hwnd)
        ctrl.save(tmp_path / "printwindow_control.png")
        assert fraction(ctrl, is_black) < 0.2
        assert fraction(ctrl, is_light) > 0.001
    finally:
        proc.kill()
        proc.wait(10)
        log.close()


class Underlay:
    """A solid magenta topmost window owned by this test process."""

    def __init__(self, rect):
        import win32api
        import win32con
        import win32gui

        self.gui = win32gui
        self.brush = win32gui.CreateSolidBrush(win32api.RGB(255, 0, 255))

        def wndproc(h, msg, wp, lp):
            if msg == win32con.WM_PAINT:
                hdc, ps = win32gui.BeginPaint(h)
                win32gui.FillRect(hdc, win32gui.GetClientRect(h), self.brush)
                win32gui.EndPaint(h, ps)
                return 0
            return win32gui.DefWindowProc(h, msg, wp, lp)

        wc = win32gui.WNDCLASS()
        wc.lpszClassName = f"MinutesCaptureUnderlay{os.getpid()}_{time.time_ns()}"
        wc.lpfnWndProc = wndproc
        wc.hInstance = win32api.GetModuleHandle(None)
        atom = win32gui.RegisterClass(wc)
        left, top, right, bottom = rect
        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE,
            atom, "underlay", win32con.WS_POPUP, left - 20, top - 20,
            right - left + 40, bottom - top + 40, 0, 0, wc.hInstance, None)
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOWNOACTIVATE)

    def put_below(self, hwnd: int) -> None:
        import win32con

        self.gui.SetWindowPos(self.hwnd, hwnd, 0, 0, 0, 0, win32con.SWP_NOMOVE
                              | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
        end = time.monotonic() + 0.6
        while time.monotonic() < end:
            self.gui.PumpWaitingMessages()
            time.sleep(0.02)

    def close(self) -> None:
        self.gui.DestroyWindow(self.hwnd)


def test_screenshot_does_not_contain_window(minutes_app, tmp_path):
    import mss
    from PIL import Image

    _, hwnd, _, _ = minutes_app
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    cx, cy = (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2
    underlay = Underlay((rect.left, rect.top, rect.right, rect.bottom))
    try:
        underlay.put_below(hwnd)
        top = user32.GetAncestor(user32.WindowFromPoint(wintypes.POINT(cx, cy)), GA_ROOT)
        if top != hwnd:
            import win32gui
            pytest.skip(f"desktop not visible (top window: {win32gui.GetWindowText(top)!r}), "
                        "probably locked")
        box = {"left": rect.left + 8, "top": rect.top + 8,
               "width": rect.right - rect.left - 16, "height": rect.bottom - rect.top - 16}
        with mss.mss() as sct:
            grab = sct.grab(box)
        img = Image.frombytes("RGB", grab.size, grab.rgb)
        img.save(tmp_path / "screenshot.png")
        frac = fraction(img, is_magenta)
        assert frac > 0.97, f"window visible in screenshot ({frac:.1%} magenta)"
    finally:
        underlay.close()


def test_second_launch_exits(minutes_app):
    proc, _, env, _ = minutes_app
    second = subprocess.run([str(PYTHON), str(ROOT / "app.py"), "--fake"], cwd=ROOT, env=env,
                            capture_output=True, timeout=30)
    assert second.returncode == 0
    assert proc.poll() is None
