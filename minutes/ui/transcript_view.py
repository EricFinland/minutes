"""Live transcript: one text block per line, appended in place (never rebuilt).

QPlainTextEdit keeps layout lazy and per block, so thousands of lines stay cheap.
Lines can arrive slightly out of order (the engine holds "Me" lines briefly for echo
checks), so a new line is inserted at its sorted position, which is almost always at
or near the end.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QTextBlockFormat, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QFrame, QPlainTextEdit

from minutes.models import ME, Line

from . import theme


def compact_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    h, m, sec = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class TranscriptView(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                     | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.setPlaceholderText("Listening. The transcript appears here.")
        self.viewport().setAutoFillBackground(False)
        self.document().setDocumentMargin(2)
        self._starts: list[float] = []

        self._fmt_time = QTextCharFormat()
        self._fmt_time.setForeground(QColor(theme.FAINT))
        self._fmt_time.setFont(theme.font(11))
        self._fmt_me = QTextCharFormat()
        self._fmt_me.setForeground(QColor(theme.ME))
        self._fmt_me.setFont(theme.font(12, QFont.Weight.DemiBold))
        self._fmt_them = QTextCharFormat(self._fmt_me)
        self._fmt_them.setForeground(QColor(theme.THEM))
        self._fmt_text = QTextCharFormat()
        self._fmt_text.setForeground(QColor(theme.TEXT))
        self._fmt_text.setFont(theme.font(13))
        self._block = QTextBlockFormat()
        self._block.setBottomMargin(5)

    def line_count(self) -> int:
        return len(self._starts)

    def clear_lines(self) -> None:
        self.clear()
        self._starts.clear()

    def set_lines(self, lines: Iterable[Line]) -> None:
        self.setUpdatesEnabled(False)
        try:
            self.clear_lines()
            for ln in sorted(lines, key=lambda l: l.start):
                self.add_line(ln, follow=False)
        finally:
            self.setUpdatesEnabled(True)
        self.scroll_to_end()

    def at_bottom(self) -> bool:
        sb = self.verticalScrollBar()
        return sb.value() >= sb.maximum() - 4

    def scroll_to_end(self) -> None:
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def add_line(self, line: Line, follow: bool = True) -> None:
        stick = follow and self.at_bottom()
        idx = bisect_right(self._starts, line.start)
        doc = self.document()
        cur = QTextCursor(doc)
        if not self._starts:
            cur.movePosition(QTextCursor.MoveOperation.Start)
        elif idx >= len(self._starts):
            cur.movePosition(QTextCursor.MoveOperation.End)
            cur.insertBlock()
        else:
            cur.setPosition(doc.findBlockByNumber(idx).position())
            cur.insertBlock()
            cur.movePosition(QTextCursor.MoveOperation.PreviousBlock)
        cur.setBlockFormat(self._block)
        cur.insertText(compact_ts(line.start) + "  ", self._fmt_time)
        cur.insertText(line.speaker, self._fmt_me if line.speaker == ME else self._fmt_them)
        cur.insertText("  " + line.text.strip(), self._fmt_text)
        self._starts.insert(idx, line.start)
        if stick:
            self.scroll_to_end()

    def block_texts(self) -> list[str]:
        doc = self.document()
        return [doc.findBlockByNumber(i).text() for i in range(len(self._starts))]
