"""The Minutes window: a small pill that expands into a live transcript panel."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QEvent, QObject, QPoint, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QSizeGrip, QSizePolicy,
    QVBoxLayout, QWidget,
)

from minutes import config
from minutes.models import ME, THEM, TRANSCRIPT_FILE, SUMMARY_FILE, POLISHED_TRANSCRIPT_FILE
from minutes.models import Event, meeting_title, read_meta

from . import stealth, startup, theme
from .hotkey import GlobalHotkey
from .transcript_view import TranscriptView, compact_ts
from .widgets import IconButton, RecordDot

log = logging.getLogger(__name__)

QWIDGETSIZE_MAX = 16777215


def _is_native_platform() -> bool:
    """False under Qt's offscreen platform (tests), where there is no real window."""
    return QGuiApplication.platformName() in ("windows", "cocoa")


def open_path(path) -> None:
    """Open a file or folder with the system's default app."""
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


class EngineBridge(QObject):
    """Engine listener that hops events onto the GUI thread through a queued signal."""

    event = pyqtSignal(object)
    quit_requested = pyqtSignal()

    def listener(self, ev: Event) -> None:
        try:
            self.event.emit(ev)
        except RuntimeError:
            pass  # window already gone during shutdown


class CaptureGuard(QObject):
    """Excludes every top-level window (menus, tooltips) from capture as it is shown."""

    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if ev.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow():
            if _is_native_platform():
                stealth.exclude_from_capture(int(obj.winId()))
        return False


def install_capture_guard(app: QApplication) -> CaptureGuard:
    guard = CaptureGuard(app)
    app.installEventFilter(guard)
    return guard


class _TaskSignals(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)


class Overlay(QWidget):
    COLLAPSED = QSize(220, 36)
    DEFAULT_EXPANDED = QSize(380, 460)
    MIN_EXPANDED = QSize(340, 240)
    HEADER_H = 36

    hotkey_fired = pyqtSignal()
    hotkey_failed = pyqtSignal(str)

    def __init__(self, cfg: dict, post: Any, *,
                 save_settings: Optional[Callable[[dict], None]] = None,
                 opener: Optional[Callable[[str], None]] = None,
                 startup_folder: Optional[Path] = None):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.cfg = cfg
        self.post = post
        self.rec: Any = None
        self._save_fn = save_settings or config.save
        self._open_fn = opener or open_path
        self._startup_folder = startup_folder

        self._recording = False
        self._meeting_dir: Optional[Path] = None
        self._meeting_open = False
        self._meeting_started: Optional[float] = None
        self._status_text = ""
        self._status_level = "info"
        self._status_pinned = False
        self._active = {ME: False, THEM: False}
        self._task: Optional[str] = None
        self._task_signals: Optional[_TaskSignals] = None
        self._expanded = False
        self._expanded_size = QSize(self.DEFAULT_EXPANDED)
        self._drag_offset: Optional[QPoint] = None
        self._quitting = False
        self._shut_down = False
        self._hotkey: Optional[GlobalHotkey] = None
        self._time_color = ""
        self.stealth_status: dict = {}

        self.setWindowTitle("Minutes")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowOpacity(0.92)
        self.setStyleSheet(theme.STYLESHEET)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.setTimerType(Qt.TimerType.VeryCoarseTimer)
        self._tick_timer.timeout.connect(self._refresh_pill)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(self.save_settings)
        self._task_clear_timer = QTimer(self)
        self._task_clear_timer.setSingleShot(True)
        self._task_clear_timer.timeout.connect(self._clear_task_message)
        self._unpin_timer = QTimer(self)
        self._unpin_timer.setSingleShot(True)
        self._unpin_timer.timeout.connect(self._unpin_status)

        self._build()
        self._restore()

        self.hotkey_fired.connect(self.toggle_visible)
        self.hotkey_failed.connect(lambda msg: self.show_task_message(msg, "warn"))
        app = QGuiApplication.instance()
        if app is not None:
            app.screenRemoved.connect(lambda *_: QTimer.singleShot(0, self.keep_on_screen))

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(0)
        self._root = root

        header = QWidget(self)
        header.setFixedHeight(self.HEADER_H)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        self._dot = RecordDot(header)
        self._time = QLabel("", header)
        self._time.setObjectName("time")
        self._time.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._me_lbl = QLabel(ME, header)
        self._them_lbl = QLabel(THEM, header)
        for lbl in (self._me_lbl, self._them_lbl):
            lbl.setObjectName("talk")
        self._me_lbl.setToolTip("You are speaking")
        self._them_lbl.setToolTip("Someone on the call is speaking")
        self._rec_btn = IconButton("record", "Start recording", header)
        self._rec_btn.clicked.connect(self.toggle_recording)
        self._expand_btn = IconButton("expand", "Show transcript", header)
        self._expand_btn.clicked.connect(lambda: self.set_expanded(not self._expanded))
        hl.addWidget(self._dot)
        hl.addSpacing(2)
        hl.addWidget(self._time, 1)
        hl.addWidget(self._me_lbl)
        hl.addWidget(self._them_lbl)
        hl.addSpacing(4)
        hl.addWidget(self._rec_btn)
        hl.addWidget(self._expand_btn)
        root.addWidget(header)

        body = QWidget(self)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        self._status = QLabel("", body)
        self._status.setObjectName("status")
        self._status.setWordWrap(True)
        divider = QFrame(body)
        divider.setObjectName("divider")
        divider.setFixedHeight(1)
        self._view = TranscriptView(body)
        self._task_lbl = QLabel("", body)
        self._task_lbl.setObjectName("task")
        self._task_lbl.setWordWrap(True)
        self._task_lbl.hide()

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 2, 0, 0)
        bar.setSpacing(6)
        self._copy_btn = self._button("Copy", "Copy the whole transcript", self.copy_transcript)
        self._open_btn = self._button("Open", "Open transcript.txt", self.open_transcript)
        self._sum_btn = self._button("Summarize", "Write summary.md with the local model", self.summarize)
        self._new_btn = self._button("New meeting", "Finish this meeting and start a new one",
                                     self.new_meeting)
        self._more_btn = IconButton("more", "More", body, size=26)
        self._more_btn.clicked.connect(self._show_menu)
        for b in (self._copy_btn, self._open_btn, self._sum_btn, self._new_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        bar.addWidget(self._more_btn)

        bl.addWidget(self._status)
        bl.addWidget(divider)
        bl.addWidget(self._view, 1)
        bl.addWidget(self._task_lbl)
        bl.addLayout(bar)
        root.addWidget(body, 1)
        self._body = body

        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(14, 14)

        self._menu = self._make_menu(self)
        self._menu.addAction("Open meetings folder", self.open_meetings_folder)
        self._prev_menu = self._make_menu(self._menu)
        self._prev_menu.setTitle("Previous meetings")
        self._prev_menu.aboutToShow.connect(self._fill_previous)
        self._menu.addMenu(self._prev_menu)
        self._menu.addSeparator()
        self._polish_action = self._menu.addAction("Polish transcript", self.polish)
        self._startup_action = self._menu.addAction(startup.label())
        self._startup_action.setCheckable(True)
        self._startup_action.triggered.connect(self.set_start_with_windows)
        self._menu.addSeparator()
        hk = (self.cfg.get("hotkey_toggle") or "ctrl+alt+m").title()
        if sys.platform == "darwin":
            hk = hk.replace("Alt", "Option").replace("Win", "Cmd")
        self._menu.addAction(f"Hide	{hk}", self.hide)
        self._menu.addAction("Quit Minutes", self.quit_app)
        self._menu.aboutToShow.connect(self._refresh_menu)

        self._set_talk_style(ME, False)
        self._set_talk_style(THEM, False)

    def _button(self, text: str, tip: str, slot: Callable[[], Any]) -> QPushButton:
        b = QPushButton(text, self)
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.clicked.connect(lambda _=False: slot())
        return b

    @staticmethod
    def _make_menu(parent: QWidget) -> QMenu:
        m = QMenu(parent)
        m.setWindowFlags(m.windowFlags() | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        m.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        return m

    # --------------------------------------------------------------- settings
    def _restore(self) -> None:
        wp = self.cfg.get("window_pos")
        pos = None
        if isinstance(wp, (list, tuple)) and len(wp) >= 2 and all(
                isinstance(v, (int, float)) for v in wp):
            pos = QPoint(int(wp[0]), int(wp[1]))
            if len(wp) >= 4:
                self._expanded_size = QSize(max(self.MIN_EXPANDED.width(), int(wp[2])),
                                            max(self.MIN_EXPANDED.height(), int(wp[3])))
        self.set_expanded(bool(self.cfg.get("expanded")), save=False)
        self.move(pos if pos is not None else self._default_pos())
        self.keep_on_screen()

    def _default_pos(self) -> QPoint:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return QPoint(100, 100)
        avail = screen.availableGeometry()
        return QPoint(avail.right() - self.width() - 24, avail.top() + 80)

    def save_settings(self) -> None:
        if self._expanded:
            self._expanded_size = self.size()
        pos = self.pos()
        self.cfg["window_pos"] = [pos.x(), pos.y(),
                                  self._expanded_size.width(), self._expanded_size.height()]
        self.cfg["expanded"] = self._expanded
        try:
            self._save_fn(self.cfg)
        except Exception:
            log.exception("could not save settings")

    def keep_on_screen(self) -> bool:
        """Move the window fully onto a screen; returns True if it moved."""
        geo = self.frameGeometry()
        probe = QPoint(geo.x() + min(geo.width(), self.COLLAPSED.width()) // 2,
                       geo.y() + self.HEADER_H // 2)
        screen = QGuiApplication.screenAt(probe)
        if screen is None:
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                return False
            avail = screen.availableGeometry()
            target = QPoint(avail.right() - geo.width() - 24, avail.top() + 80)
        else:
            avail = screen.availableGeometry()
            target = QPoint(geo.x(), geo.y())
        x = min(max(target.x(), avail.left()), avail.right() - geo.width() + 1)
        y = min(max(target.y(), avail.top()), avail.bottom() - geo.height() + 1)
        if (x, y) != (geo.x(), geo.y()):
            self.move(x, y)
            return True
        return False

    # --------------------------------------------------------------- expanded
    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, on: bool, save: bool = True) -> None:
        if self._expanded and not on:
            self._expanded_size = self.size()
        target = QSize(self._expanded_size)
        self._expanded = False  # keep resizeEvent from recording intermediate sizes
        self._body.setVisible(on)
        self._grip.setVisible(on)
        if on:
            self._root.setContentsMargins(14, 0, 8, 10)
            self.setMinimumSize(self.MIN_EXPANDED)
            self.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
            self.resize(target)
            self._expanded = True
            self._expanded_size = target
            self._expand_btn.set_kind("collapse", "Hide transcript")
        else:
            self._root.setContentsMargins(12, 0, 5, 0)
            self.setFixedSize(self.COLLAPSED)
            self._expand_btn.set_kind("expand", "Show transcript")
        self._apply_native()
        if self.isVisible():
            self.keep_on_screen()
        if on:
            self._view.scroll_to_end()
        self._refresh_pill()
        self.update()
        if save:
            self.save_settings()

    # ----------------------------------------------------------------- native
    def _apply_native(self) -> None:
        if not _is_native_platform():
            return
        hwnd = int(self.winId())
        # Collapsed pill never takes focus; the expanded panel may, so text can be selected.
        self.stealth_status = stealth.apply(hwnd, no_activate=not self._expanded)

    def event(self, ev: QEvent) -> bool:
        if ev.type() == QEvent.Type.WinIdChange:
            QTimer.singleShot(0, self._apply_native)
        return super().event(ev)

    def showEvent(self, ev):
        self._apply_native()
        super().showEvent(ev)
        self._update_timer()
        self._refresh_pill()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self._update_timer()

    def closeEvent(self, ev):
        if self._quitting:
            ev.accept()
        else:
            ev.ignore()
            self.hide()

    def show_overlay(self) -> None:
        self.winId()
        self._apply_native()
        self.keep_on_screen()
        self.show()

    def toggle_visible(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show_overlay()

    def start_hotkey(self) -> None:
        spec = self.cfg.get("hotkey_toggle") or "ctrl+alt+m"
        self._hotkey = GlobalHotkey(spec, self.hotkey_fired.emit, self.hotkey_failed.emit)
        self._hotkey.start()

    # --------------------------------------------------------- paint and drag
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = 12.0 if self._expanded else r.height() / 2
        p.setPen(QPen(theme.BORDER, 1))
        p.setBrush(QColor(theme.BG))
        p.drawRoundedRect(r, radius, radius)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._grip.move(self.width() - self._grip.width() - 2, self.height() - self._grip.height() - 2)
        self._grip.raise_()
        if self._expanded:
            self._expanded_size = self.size()
            if self.isVisible():
                self._save_timer.start()
        self._refresh_pill()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()
        else:
            super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag_offset is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_offset)
            e.accept()

    def mouseReleaseEvent(self, e):
        if self._drag_offset is not None:
            self._drag_offset = None
            self.keep_on_screen()
            self.save_settings()
            e.accept()

    def mouseDoubleClickEvent(self, e):
        if e.position().y() <= self.HEADER_H:
            self.set_expanded(not self._expanded)
            e.accept()

    # ----------------------------------------------------------------- engine
    def attach(self, rec: Any) -> None:
        self.rec = rec
        self._set_recording(bool(getattr(rec, "recording", False)))
        md = getattr(rec, "meeting_dir", None)
        if md:
            self._set_meeting(Path(md))
            try:
                self._view.set_lines(rec.lines())
            except Exception:
                log.exception("could not load current lines")

    def on_engine_event(self, ev: Event) -> None:
        try:
            kind = ev.kind
            d = ev.data
            if kind == "line":
                self._on_line(d.get("line"), d.get("meeting_dir"))
            elif kind == "state":
                self._set_recording(bool(d.get("recording")))
            elif kind == "meeting":
                md = d.get("meeting_dir")
                self._set_meeting(Path(md) if md else None)
            elif kind == "status":
                self.show_status(str(d.get("text", "")), str(d.get("level", "info")))
            elif kind == "activity":
                self._set_active(d.get("speaker"), bool(d.get("active")))
        except Exception:
            log.exception("bad engine event %r", ev)

    def _on_line(self, line, md) -> None:
        if line is None:
            return
        if md is not None and Path(md) != self._meeting_dir:
            self._set_meeting(Path(md))
        elif not self._meeting_open and self._meeting_dir is not None:
            self._meeting_open = True
            self._update_timer()
        self._maybe_unpin_progress()
        self._view.add_line(line)

    def _set_recording(self, on: bool) -> None:
        self._recording = on
        self._dot.set_recording(on)
        if on:
            self._rec_btn.set_kind("stop", "Stop recording")
        else:
            self._rec_btn.set_kind("record", "Start recording")
            for spk in (ME, THEM):
                self._set_active(spk, False)
        self._refresh_pill()

    def _read_started(self, md: Path) -> float:
        started = read_meta(md).get("started")
        if isinstance(started, str):
            try:
                return datetime.fromisoformat(started).timestamp()
            except ValueError:
                pass
        try:
            return datetime.strptime(md.name[:17], "%Y-%m-%d_%H%M%S").timestamp()
        except ValueError:
            return time.time()

    def _set_meeting(self, md: Optional[Path]) -> None:
        if md is None:
            self._meeting_open = False
        else:
            if md != self._meeting_dir:
                self._view.clear_lines()
                self._meeting_dir = md
                self._meeting_started = self._read_started(md)
            self._meeting_open = True
            self._status.setToolTip(meeting_title(md))
        self._update_timer()
        self._refresh_pill()

    def show_status(self, text: str, level: str = "info") -> None:
        level = level if level in theme.LEVEL_COLORS else "info"
        self._status_text = text
        self._status_level = level
        self._status_pinned = bool(text) and (
            level in ("warn", "error") or text.rstrip().endswith(("...", chr(0x2026))))
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {theme.LEVEL_COLORS[level]};")
        if level == "warn":
            self._unpin_timer.start(30000)
        else:
            self._unpin_timer.stop()
        self._refresh_pill()

    def _unpin_status(self) -> None:
        self._status_pinned = False
        self._refresh_pill()

    def _maybe_unpin_progress(self) -> None:
        # Speech is flowing, so an "in progress" info status (model loading) is over.
        if self._status_pinned and self._status_level == "info":
            self._unpin_status()

    def _set_active(self, speaker: Any, active: bool) -> None:
        if speaker not in self._active:
            return
        if active:
            self._maybe_unpin_progress()
        if self._active[speaker] != active:
            self._active[speaker] = active
            self._set_talk_style(speaker, active)

    def _set_talk_style(self, speaker: str, active: bool) -> None:
        lbl = self._me_lbl if speaker == ME else self._them_lbl
        color = (theme.ME if speaker == ME else theme.THEM) if active else theme.FAINT
        lbl.setStyleSheet(f"color: {color};")
        lbl.setProperty("active", active)

    def is_talking(self, speaker: str) -> bool:
        return self._active.get(speaker, False)

    def _update_timer(self) -> None:
        if self._meeting_open and self.isVisible():
            if not self._tick_timer.isActive():
                self._tick_timer.start()
        else:
            self._tick_timer.stop()

    def pill_text(self) -> str:
        if self._status_pinned and self._status_text:
            return self._status_text
        if self._meeting_open and self._meeting_started is not None:
            return compact_ts(time.time() - self._meeting_started)
        return "Listening" if self._recording else "Paused"

    def _refresh_pill(self) -> None:
        text = self.pill_text()
        pinned = bool(self._status_pinned and self._status_text)
        if self._me_lbl.isHidden() != pinned:
            # A status needs the room more than the talk indicator does.
            self._me_lbl.setHidden(pinned)
            self._them_lbl.setHidden(pinned)
            self._time.parentWidget().layout().activate()
        if pinned:
            color = theme.LEVEL_COLORS[self._status_level]
        elif self._meeting_open and self._recording:
            color = theme.TEXT
        else:
            color = theme.MUTED
        width = max(24, self._time.width())
        shown = self._time.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, width)
        if shown != self._time.text():
            self._time.setText(shown)
        self._time.setToolTip(self._status_text or text)
        if color != self._time_color:
            self._time_color = color
            self._time.setStyleSheet(f"color: {color};")

    # ---------------------------------------------------------------- actions
    def toggle_recording(self) -> None:
        if self.rec is None:
            return
        try:
            if self._recording:
                self.rec.stop()
            else:
                self.rec.start()
        except Exception as exc:
            log.exception("record toggle failed")
            self.show_status(f"Could not change recording: {exc}", "error")

    def new_meeting(self) -> None:
        if self.rec is None:
            return
        try:
            self.rec.new_meeting()
        except Exception as exc:
            log.exception("new meeting failed")
            self.show_task_message(f"Could not start a new meeting: {exc}", "error")

    def target_meeting(self) -> Optional[Path]:
        if self._meeting_dir is not None:
            return self._meeting_dir
        try:
            meetings = self.post.list_meetings(self.cfg)
            return Path(meetings[0]) if meetings else None
        except Exception:
            log.exception("list_meetings failed")
            return None

    def copy_transcript(self) -> bool:
        d = self.target_meeting()
        if d is None:
            self.show_task_message("Nothing to copy yet", "info")
            return False
        try:
            text = self.post.transcript_text(d)
        except Exception as exc:
            self.show_task_message(f"Could not read the transcript: {exc}", "error")
            return False
        if not text or not text.strip():
            self.show_task_message("Nothing to copy yet", "info")
            return False
        QApplication.clipboard().setText(text)
        self._copy_btn.setText("Copied")
        QTimer.singleShot(1500, lambda: self._copy_btn.setText("Copy"))
        return True

    def open_transcript(self) -> None:
        d = self.target_meeting()
        path = d / TRANSCRIPT_FILE if d is not None else None
        if path is None or not path.exists():
            self.show_task_message("No transcript yet", "info")
            return
        self._open(path)

    def open_meetings_folder(self) -> None:
        self._open(config.meetings_dir())

    def _open(self, path: Path) -> None:
        try:
            self._open_fn(str(path))
        except OSError as exc:
            self.show_task_message(f"Could not open {Path(path).name}: {exc}", "error")

    def summarize(self) -> bool:
        return self._run_task("summarize", self.post.summarize, SUMMARY_FILE,
                              "Summarizing...", "Summary ready", "Summary failed")

    def polish(self) -> bool:
        return self._run_task("polish", self.post.polish, POLISHED_TRANSCRIPT_FILE,
                              "Polishing transcript...", "Polished transcript ready",
                              "Polish failed")

    def _run_task(self, name: str, fn: Callable, default_file: str, busy: str,
                  ok: str, failed: str) -> bool:
        if self._task is not None:
            return False
        d = self.target_meeting()
        if d is None:
            self.show_task_message("No meeting yet", "info")
            return False
        self._task = name
        sig = _TaskSignals(self)
        sig.progress.connect(lambda msg: self.show_task_message(msg, "info", sticky=True))
        sig.done.connect(lambda res: self._task_finished(Path(res) if res else d / default_file,
                                                         None, ok, failed))
        sig.failed.connect(lambda err: self._task_finished(None, err, ok, failed))
        self._task_signals = sig
        cfg = self.cfg

        def worker():
            try:
                result = fn(d, cfg, on_progress=sig.progress.emit)
            except Exception as exc:
                log.exception("%s failed", name)
                msg = str(exc) or type(exc).__name__
                try:
                    sig.failed.emit(msg)
                except RuntimeError:
                    pass
            else:
                try:
                    sig.done.emit(result)
                except RuntimeError:
                    pass

        threading.Thread(target=worker, name=f"minutes-{name}", daemon=True).start()
        self.show_task_message(busy, "info", sticky=True)
        self._update_actions()
        return True

    def _task_finished(self, path: Optional[Path], err: Optional[str], ok: str, failed: str) -> None:
        self._task = None
        if self._task_signals is not None:
            self._task_signals.deleteLater()
            self._task_signals = None
        self._update_actions()
        if err is not None:
            self.show_task_message(f"{failed}: {err}", "error")
        else:
            self.show_task_message(ok, "ok")
            if path is not None:
                self._open(path)

    @property
    def busy_task(self) -> Optional[str]:
        return self._task

    def _update_actions(self) -> None:
        busy = self._task is not None
        self._sum_btn.setEnabled(not busy)
        self._sum_btn.setText("Summarizing..." if self._task == "summarize" else "Summarize")
        self._polish_action.setEnabled(not busy)
        self._polish_action.setText("Polishing..." if self._task == "polish" else "Polish transcript")

    def show_task_message(self, text: str, level: str = "info", sticky: bool = False) -> None:
        color = theme.LEVEL_COLORS.get(level, theme.MUTED)
        self._task_lbl.setStyleSheet(f"color: {color};")
        self._task_lbl.setText(text)
        self._task_lbl.setVisible(bool(text))
        if sticky:
            self._task_clear_timer.stop()
        else:
            self._task_clear_timer.start(30000 if level == "error" else 6000)

    def _clear_task_message(self) -> None:
        if self._task is None:
            self._task_lbl.clear()
            self._task_lbl.hide()

    def task_message(self) -> str:
        return self._task_lbl.text()

    # ------------------------------------------------------------------- menu
    def _show_menu(self) -> None:
        self._menu.adjustSize()
        hint = self._menu.sizeHint()
        btn = self._more_btn
        pos = btn.mapToGlobal(QPoint(btn.width() - hint.width(), -hint.height() - 4))
        self._menu.popup(pos)

    def _refresh_menu(self) -> None:
        try:
            self._startup_action.setChecked(startup.is_enabled(self._startup_folder))
        except Exception:
            self._startup_action.setChecked(False)
        self._update_actions()

    def _fill_previous(self) -> None:
        m = self._prev_menu
        m.clear()
        try:
            meetings = list(self.post.list_meetings(self.cfg))[:12]
        except Exception:
            meetings = []
        if not meetings:
            a = m.addAction("No meetings yet")
            a.setEnabled(False)
            return
        for d in meetings:
            d = Path(d)
            m.addAction(meeting_title(d), lambda d=d: self._open_meeting(d))

    def _open_meeting(self, d: Path) -> None:
        t = d / TRANSCRIPT_FILE
        self._open(t if t.exists() else d)

    def set_start_with_windows(self, enabled: bool) -> None:
        try:
            state = startup.set_enabled(enabled, self._startup_folder)
        except Exception as exc:
            log.exception("startup toggle failed")
            self.show_task_message(f"Could not change startup setting: {exc}", "error")
            state = startup.is_enabled(self._startup_folder)
        self._startup_action.setChecked(state)

    # ------------------------------------------------------------------- quit
    def shutdown_engine(self) -> None:
        """Finalize the transcript. Safe to call more than once (quit, logoff)."""
        if self._shut_down:
            return
        self._shut_down = True
        self.save_settings()
        if self._hotkey is not None:
            self._hotkey.stop()
        self._tick_timer.stop()
        if self.rec is not None:
            try:
                self.rec.shutdown()
            except Exception:
                log.exception("recorder shutdown failed")

    def quit_app(self) -> None:
        self._quitting = True
        self.save_settings()
        self.hide()
        self.shutdown_engine()
        QApplication.quit()
