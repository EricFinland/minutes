"""UI tests against the fake recorder, on Qt's offscreen platform."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from PyQt6.QtCore import QPoint, Qt  # noqa: E402
from PyQt6.QtGui import QGuiApplication  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from minutes import config  # noqa: E402
from minutes.models import ME, THEM, TRANSCRIPT_FILE, Line, event  # noqa: E402
from minutes.ui import EngineBridge, Overlay  # noqa: E402
from minutes.ui.dev_fake import FakePost, FakeRecorder  # noqa: E402
from minutes.ui.hotkey import MOD_ALT, MOD_CONTROL, MOD_SHIFT, parse_hotkey  # noqa: E402
from minutes.ui.single_instance import SingleInstance  # noqa: E402
from minutes.ui.transcript_view import TranscriptView, compact_ts  # noqa: E402

app = QApplication.instance() or QApplication([])


def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        QTest.qWait(20)
    return pred()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINUTES_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def opened():
    return []


@pytest.fixture
def make_overlay(data_dir, opened, tmp_path):
    made = []

    def factory(cfg=None, post=None, save=None):
        cfg = cfg if cfg is not None else config.load()
        ov = Overlay(cfg, post or FakePost(delay=0.05), save_settings=save or config.save,
                     opener=opened.append, startup_folder=tmp_path / "Startup")
        made.append(ov)
        return ov

    yield factory
    for ov in made:
        if ov.rec is not None:
            ov.rec.shutdown()
        ov._quitting = True
        ov.close()
        ov.deleteLater()
    QTest.qWait(10)


def attach_fake(ov, **kw):
    bridge = EngineBridge(ov)
    bridge.event.connect(ov.on_engine_event)
    kw.setdefault("line_interval", 0.03)
    kw.setdefault("load_delay", 0.02)
    rec = FakeRecorder(ov.cfg, bridge.listener, **kw)
    ov.attach(rec)
    return rec


def test_window_builds_collapsed(make_overlay):
    ov = make_overlay()
    ov.show_overlay()
    flags = ov.windowFlags()
    assert ov.windowTitle() == "Minutes"
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.Tool
    assert ov.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert abs(ov.windowOpacity() - 0.92) < 0.01
    assert not ov.expanded
    assert (ov.width(), ov.height()) == (220, 36)
    assert ov.pill_text() == "Paused"


def test_fake_recorder_events_render(make_overlay):
    ov = make_overlay()
    ov.show_overlay()
    rec = attach_fake(ov)
    rec.start()
    assert wait_until(lambda: ov._view.line_count() >= 4)
    rec.stop()
    assert ov._dot.recording is False
    texts = ov._view.block_texts()
    assert any(f" {ME} " in t for t in texts)
    assert any(f" {THEM} " in t for t in texts)
    assert "Okay, looks like everyone's here" in texts[0]
    assert ov._meeting_dir == rec.meeting_dir
    # The meeting clock is showing (mm:ss), not a status string.
    assert ov.pill_text().count(":") == 1


def test_status_and_recording_state(make_overlay):
    ov = make_overlay()
    ov.on_engine_event(event("status", text="Loading speech model...", level="info"))
    assert ov.pill_text() == "Loading speech model..."
    assert ov._status.text() == "Loading speech model..."
    ov.on_engine_event(event("state", recording=True))
    assert ov._dot.recording and ov._rec_btn.kind == "stop"
    ov.on_engine_event(event("status", text="Listening", level="info"))
    assert ov.pill_text() == "Listening"
    ov.on_engine_event(event("status", text="Mic unavailable", level="error"))
    assert ov.pill_text() == "Mic unavailable"
    ov.on_engine_event(event("state", recording=False))
    assert not ov._dot.recording and ov._rec_btn.kind == "record"


def test_activity_indicator(make_overlay):
    ov = make_overlay()
    ov.on_engine_event(event("state", recording=True))
    ov.on_engine_event(event("activity", speaker=ME, active=True))
    assert ov.is_talking(ME) and not ov.is_talking(THEM)
    ov.on_engine_event(event("activity", speaker=THEM, active=True))
    ov.on_engine_event(event("activity", speaker=ME, active=False))
    assert ov.is_talking(THEM) and not ov.is_talking(ME)
    ov.on_engine_event(event("state", recording=False))
    assert not ov.is_talking(THEM)


def test_meeting_events(make_overlay, data_dir):
    ov = make_overlay()
    m1 = config.meetings_dir() / "2026-10-29_090000"
    m1.mkdir(parents=True)
    ov.on_engine_event(event("meeting", meeting_dir=m1))
    ov.on_engine_event(event("line", line=Line(ME, 1.0, 2.0, "hello"), meeting_dir=m1))
    assert ov._view.line_count() == 1
    ov.on_engine_event(event("meeting", meeting_dir=None))
    assert ov._view.line_count() == 1  # still visible to copy after the meeting closes
    assert ov.target_meeting() == m1
    m2 = config.meetings_dir() / "2026-10-29_100000"
    m2.mkdir()
    ov.on_engine_event(event("meeting", meeting_dir=m2))
    assert ov._view.line_count() == 0


def test_copy_transcript(make_overlay):
    ov = make_overlay()
    rec = attach_fake(ov)
    rec.start()
    assert wait_until(lambda: ov._view.line_count() >= 3)
    rec.stop()
    QApplication.clipboard().clear()
    assert ov.copy_transcript()
    clip = QApplication.clipboard().text()
    expected = (rec.meeting_dir / TRANSCRIPT_FILE).read_text(encoding="utf-8")
    assert clip == expected
    assert "] Them: Okay, looks like everyone's here" in clip
    assert ov._copy_btn.text() == "Copied"
    assert wait_until(lambda: ov._copy_btn.text() == "Copy", 3)


def test_copy_with_nothing(make_overlay):
    ov = make_overlay()
    assert not ov.copy_transcript()
    assert "Nothing to copy" in ov.task_message()


def test_open_transcript(make_overlay, opened):
    ov = make_overlay()
    rec = attach_fake(ov)
    rec.start()
    assert wait_until(lambda: ov._view.line_count() >= 1)
    ov.open_transcript()
    assert opened == [str(rec.meeting_dir / TRANSCRIPT_FILE)]


def test_expand_collapse(make_overlay):
    ov = make_overlay()
    ov.show_overlay()
    ov.set_expanded(True)
    assert ov.expanded and ov._body.isVisible()
    assert ov.width() >= 340 and ov.height() >= 240
    assert ov._expand_btn.kind == "collapse"
    ov.resize(420, 500)
    ov.set_expanded(False)
    assert not ov.expanded and not ov._body.isVisible()
    assert (ov.width(), ov.height()) == (220, 36)
    ov.set_expanded(True)
    assert (ov.width(), ov.height()) == (420, 500)


def test_double_click_header_toggles(make_overlay):
    ov = make_overlay()
    ov.show_overlay()
    QTest.mouseDClick(ov, Qt.MouseButton.LeftButton, pos=QPoint(60, 18))
    assert ov.expanded


def test_settings_persist(make_overlay, data_dir):
    ov = make_overlay()
    ov.show_overlay()
    ov.set_expanded(True)
    ov.resize(400, 480)
    ov.move(ov.pos() + QPoint(-30, 20))
    ov.keep_on_screen()
    ov.save_settings()
    saved_pos = ov.pos()

    cfg = config.load()
    assert cfg["expanded"] is True
    assert cfg["window_pos"][:2] == [saved_pos.x(), saved_pos.y()]
    assert cfg["window_pos"][2:] == [400, 480]

    ov2 = make_overlay(cfg)
    assert ov2.expanded
    assert ov2.pos() == saved_pos
    assert (ov2.width(), ov2.height()) == (400, 480)


def test_remembered_monitor_gone(make_overlay):
    cfg = config.load()
    cfg["window_pos"] = [-20000, -20000]
    ov = make_overlay(cfg)
    avail = QGuiApplication.primaryScreen().availableGeometry()
    assert avail.contains(ov.frameGeometry())


def test_out_of_order_lines_sorted():
    view = TranscriptView()
    for start, spk, text in [(1, ME, "a"), (5, THEM, "c"), (3, ME, "b"), (9, THEM, "d")]:
        view.add_line(Line(spk, float(start), float(start) + 1, text))
    assert [t.split()[-1] for t in view.block_texts()] == ["a", "b", "c", "d"]
    assert compact_ts(3725) == "1:02:05"
    assert compact_ts(65) == "01:05"


def test_long_meeting_is_fast():
    view = TranscriptView()
    view.resize(380, 400)
    view.show()
    t0 = time.perf_counter()
    for i in range(4000):
        view.add_line(Line(ME if i % 2 else THEM, i * 1.8, i * 1.8 + 1.5,
                           f"Line {i}: some reasonably long sentence that wraps in the panel."))
    elapsed = time.perf_counter() - t0
    assert view.line_count() == 4000
    assert view.document().blockCount() == 4000
    assert view.at_bottom()
    # The last appends must cost about as much as the first ones (no rebuilds).
    t1 = time.perf_counter()
    for i in range(4000, 4100):
        view.add_line(Line(ME, i * 1.8, i * 1.8 + 1, "tail line"))
    per_line = (time.perf_counter() - t1) / 100
    assert elapsed < 10, elapsed
    assert per_line < 0.01, per_line
    view.close()


def test_scroll_stays_put_when_reading_back():
    view = TranscriptView()
    view.resize(380, 200)
    view.show()
    for i in range(200):
        view.add_line(Line(ME, float(i), float(i), f"line {i}"))
    sb = view.verticalScrollBar()
    sb.setValue(0)
    view.add_line(Line(THEM, 500.0, 501.0, "new"))
    assert sb.value() == 0
    view.close()


def test_summarize_success(make_overlay, opened):
    ov = make_overlay(post=FakePost(delay=0.3))
    rec = attach_fake(ov)
    rec.start()
    assert wait_until(lambda: ov._view.line_count() >= 2)
    assert ov.summarize()
    assert not ov._sum_btn.isEnabled()
    assert ov._sum_btn.text() == "Summarizing..."
    assert not ov.summarize()  # second click while running is ignored
    assert wait_until(lambda: ov.busy_task is None, 5)
    assert ov._sum_btn.isEnabled()
    summary = rec.meeting_dir / "summary.md"
    assert summary.exists()
    assert opened[-1] == str(summary)
    assert ov.task_message() == "Summary ready"


def test_summarize_error_inline(make_overlay, opened):
    ov = make_overlay(post=FakePost(delay=0.05, fail="Ollama is not running"))
    rec = attach_fake(ov)
    rec.start()
    assert wait_until(lambda: ov._view.line_count() >= 1)
    assert ov.summarize()
    assert wait_until(lambda: ov.busy_task is None, 5)
    assert "Summary failed: Ollama is not running" == ov.task_message()
    assert ov._sum_btn.isEnabled()
    assert opened == []


def test_polish(make_overlay, opened):
    ov = make_overlay()
    rec = attach_fake(ov)
    rec.start()
    assert wait_until(lambda: ov._view.line_count() >= 2)
    assert ov.polish()
    assert not ov._polish_action.isEnabled()
    assert wait_until(lambda: ov.busy_task is None, 5)
    assert opened[-1] == str(rec.meeting_dir / "transcript_polished.txt")


def test_new_meeting_and_record_toggle(make_overlay):
    ov = make_overlay()
    rec = attach_fake(ov)
    ov.toggle_recording()
    assert wait_until(lambda: ov._view.line_count() >= 2)
    first = rec.meeting_dir
    ov.new_meeting()
    assert wait_until(lambda: ov._meeting_dir != first)
    assert ov._meeting_dir == rec.meeting_dir
    ov.toggle_recording()
    assert wait_until(lambda: not ov._dot.recording)
    assert not rec.recording


def test_quit_and_close(make_overlay, monkeypatch):
    ov = make_overlay()
    rec = attach_fake(ov)
    rec.start()
    ov.show_overlay()
    ov.close()  # closing only hides
    assert not ov.isVisible()
    assert rec.shutdown_calls == 0
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: None))
    ov.quit_app()
    ov.shutdown_engine()
    assert rec.shutdown_calls == 1
    meta = (rec_dir := sorted(config.meetings_dir().iterdir())[-1]) / "meta.json"
    assert '"ended": null' not in meta.read_text(encoding="utf-8"), rec_dir


def test_start_with_windows_toggle(make_overlay, tmp_path):
    from minutes.ui import startup

    folder = tmp_path / "Startup"
    ov = make_overlay()
    ov.set_start_with_windows(True)
    lnk = folder / startup.SHORTCUT_NAME
    assert lnk.exists() and ov._startup_action.isChecked()
    info = startup.read_shortcut(lnk)
    assert info["target"].lower().endswith("pythonw.exe")
    assert info["arguments"] == f'"{ROOT / "run.pyw"}"'
    assert Path(info["workdir"]) == ROOT
    ov.set_start_with_windows(False)
    assert not lnk.exists() and not ov._startup_action.isChecked()


def test_parse_hotkey():
    assert parse_hotkey("ctrl+alt+m") == (MOD_CONTROL | MOD_ALT, ord("M"))
    assert parse_hotkey("Ctrl + Shift + F9") == (MOD_CONTROL | MOD_SHIFT, 0x78)
    assert parse_hotkey("ctrl+alt") is None
    assert parse_hotkey("ctrl+bogus") is None


def test_single_instance(tmp_path):
    key = f"MinutesTest-{os.getpid()}-{time.time_ns()}"
    first = SingleInstance(key)
    assert first.acquire()
    hits = []
    first.activated.connect(lambda: hits.append(1))
    second = SingleInstance(key)
    assert not second.acquire()
    assert wait_until(lambda: hits, 3)
    first.release()


def test_no_em_dash_in_ui_files():
    files = list((ROOT / "minutes" / "ui").glob("*.py")) + [
        ROOT / "app.py", ROOT / "run.pyw", ROOT / "make_shortcut.py"] + list(
        (ROOT / "tests").glob("test_ui*.py"))
    for f in files:
        assert chr(0x2014) not in f.read_text(encoding="utf-8"), f
