"""Minutes entry point: builds the Recorder and the overlay and wires them together.

    python app.py           real engine (minutes.engine) and post tools (minutes.post)
    python app.py --fake    simulated recorder and post stubs, data in a temp folder

Use run.pyw to launch without a console window.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("minutes")


def _preload_native_libs() -> None:
    """Import onnxruntime before PyQt6.

    Both ship their own copy of the MSVC runtime and whichever loads first wins.
    If Qt is first, importing onnxruntime (used by the voice detector) fails with an
    access violation, so it has to be loaded before any Qt module.
    """
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        log.exception("could not preload onnxruntime")


def _load_backend(fake: bool):
    """Return (recorder factory, post module, problem text or None)."""
    from minutes.ui import dev_fake

    if fake:
        return dev_fake.FakeRecorder, dev_fake.FakePost(), None
    problems = []
    try:
        from minutes.engine import Recorder
        factory = Recorder
    except Exception as exc:
        log.exception("could not import minutes.engine")
        reason = f"Speech engine unavailable: {exc}"
        problems.append(reason)

        def factory(cfg, listener, _reason=reason):
            return dev_fake.UnavailableRecorder(cfg, listener, _reason)
    try:
        import minutes.post as post
    except Exception as exc:
        log.exception("could not import minutes.post")
        post = dev_fake.UnavailablePost(f"Summary tools unavailable: {exc}")
        problems.append(post.reason)
    return factory, post, ("; ".join(problems) or None)


def _install_console_handler(overlay, bridge) -> None:
    """When run from a console: finalize the transcript if that console closes."""
    try:
        import win32api
        import win32con
    except ImportError:
        return
    finalize = {win32con.CTRL_CLOSE_EVENT, win32con.CTRL_LOGOFF_EVENT, win32con.CTRL_SHUTDOWN_EVENT}

    def handler(ctrl_type):
        if ctrl_type in finalize:
            overlay.shutdown_engine()
            return True
        if ctrl_type in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT):
            bridge.quit_requested.emit()  # queued onto the GUI thread
            return True
        return False

    try:
        win32api.SetConsoleCtrlHandler(handler, True)
    except Exception:
        log.debug("no console handler", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Minutes: local meeting transcriber")
    parser.add_argument("--fake", action="store_true",
                        help="simulated recorder and post tools (UI development)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.fake and not os.environ.get("MINUTES_DATA_DIR"):
        os.environ["MINUTES_DATA_DIR"] = str(Path(tempfile.gettempdir()) / "minutes-fake")
    if not args.fake:
        _preload_native_libs()

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication

    from minutes import config
    from minutes.ui import EngineBridge, Overlay, install_capture_guard, stealth
    from minutes.ui.single_instance import SingleInstance, instance_key

    cfg = config.load()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Minutes")
    app.setQuitOnLastWindowClosed(False)
    stealth.prepare_app()  # macOS: no Dock icon or Cmd-Tab entry

    instance = SingleInstance(instance_key(config.data_dir()))
    if not instance.acquire():
        log.info("already running; asked the existing window to show itself")
        return 0

    guard = install_capture_guard(app)  # noqa: F841 (kept alive by its parent)
    factory, post, problem = _load_backend(args.fake)

    overlay = Overlay(cfg, post)
    bridge = EngineBridge()
    bridge.event.connect(overlay.on_engine_event)
    bridge.quit_requested.connect(overlay.quit_app)
    rec = factory(cfg, bridge.listener)
    overlay.attach(rec)

    instance.activated.connect(overlay.show_overlay)
    # Logoff or shutdown: Windows asks running apps to save; finalize the transcript then.
    app.commitDataRequest.connect(lambda *_: overlay.shutdown_engine())
    app.aboutToQuit.connect(overlay.shutdown_engine)
    _install_console_handler(overlay, bridge)

    overlay.start_hotkey()
    overlay.show_overlay()
    if problem:
        overlay.show_status(problem, "error")
    if cfg.get("record_on_launch", True):
        rec.start()

    def cleanup():
        try:
            n = post.cleanup_old_audio(cfg)
            if n:
                log.info("removed %d old audio files", n)
        except Exception:
            log.exception("audio cleanup failed")

    threading.Thread(target=cleanup, name="minutes-cleanup", daemon=True).start()
    code = app.exec()
    overlay.shutdown_engine()
    instance.release()
    return code


if __name__ == "__main__":
    sys.exit(main())
