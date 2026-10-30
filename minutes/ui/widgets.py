"""Small painted widgets: crisp at any DPI and no icon font dependency."""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QAbstractButton, QWidget

from . import theme


class RecordDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._recording = False
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    @property
    def recording(self) -> bool:
        return self._recording

    def set_recording(self, on: bool) -> None:
        if on != self._recording:
            self._recording = on
            self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        c = QPointF(self.width() / 2, self.height() / 2)
        if self._recording:
            halo = QColor(theme.RECORDING)
            halo.setAlpha(60)
            p.setBrush(halo)
            p.drawEllipse(c, 6.5, 6.5)
            p.setBrush(QColor(theme.RECORDING))
        else:
            p.setBrush(QColor(theme.STOPPED))
        p.drawEllipse(c, 4.0, 4.0)


class IconButton(QAbstractButton):
    """Flat round-cornered icon button. kind: record, stop, expand, collapse, more."""

    def __init__(self, kind: str, tip: str = "", parent=None, size: int = 24):
        super().__init__(parent)
        self._kind = kind
        self._hover = False
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if tip:
            self.setToolTip(tip)

    @property
    def kind(self) -> str:
        return self._kind

    def set_kind(self, kind: str, tip: str | None = None) -> None:
        self._kind = kind
        if tip is not None:
            self.setToolTip(tip)
        self.update()

    def sizeHint(self) -> QSize:
        return self.size()

    def enterEvent(self, e):
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        if self._hover or self.isDown():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 40 if self.isDown() else 26))
            p.drawRoundedRect(r, 6, 6)
        c = r.center()
        k = self._kind
        if k == "record":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.RECORDING))
            p.drawEllipse(c, 5.0, 5.0)
        elif k == "stop":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#D6D7DB"))
            p.drawRoundedRect(QRectF(c.x() - 4.5, c.y() - 4.5, 9, 9), 2, 2)
        elif k in ("expand", "collapse"):
            pen = QPen(QColor("#B9BBC2"), 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            dy = 2.0 if k == "expand" else -2.0
            p.drawPolyline([QPointF(c.x() - 4, c.y() - dy), QPointF(c.x(), c.y() + dy),
                            QPointF(c.x() + 4, c.y() - dy)])
        elif k == "more":
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#B9BBC2"))
            for dx in (-5, 0, 5):
                p.drawEllipse(QPointF(c.x() + dx, c.y()), 1.5, 1.5)
