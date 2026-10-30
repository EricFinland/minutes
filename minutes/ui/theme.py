"""Colors, fonts and the stylesheet. Dark and quiet."""
from __future__ import annotations

from PyQt6.QtGui import QColor, QFont

BG = "#17181C"
BORDER = QColor(255, 255, 255, 24)
TEXT = "#E6E7EA"
MUTED = "#8B8E97"
FAINT = "#4C4F57"
ME = "#7DB4FF"       # soft blue
THEM = "#E8B86C"     # soft amber
RECORDING = "#FF4D4F"
STOPPED = "#6B6E76"
WARN = "#E8B86C"
ERROR = "#FF7A70"
OK = "#7FCB9A"

LEVEL_COLORS = {"info": MUTED, "warn": WARN, "error": ERROR, "ok": OK}

FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI Variable", "Segoe UI",
                 ".AppleSystemUIFont", "Helvetica Neue"]


def font(pixel_size: int = 12, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPixelSize(pixel_size)
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferVerticalHinting)
    return f


STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    font-family: "Segoe UI Variable Text", "Segoe UI", ".AppleSystemUIFont", "Helvetica Neue";
    font-size: 12px;
}}
QLabel#time {{ font-size: 12px; font-weight: 600; }}
QLabel#status, QLabel#task {{ color: {MUTED}; font-size: 11px; }}
QLabel#talk {{ font-size: 10px; font-weight: 600; padding: 0px 1px; }}
QFrame#divider {{ background: rgba(255, 255, 255, 0.06); border: none; }}
QPushButton {{
    color: #D6D7DB;
    background: rgba(255, 255, 255, 0.06);
    border: none;
    border-radius: 6px;
    padding: 4px 10px;
    min-height: 18px;
}}
QPushButton:hover {{ background: rgba(255, 255, 255, 0.11); }}
QPushButton:pressed {{ background: rgba(255, 255, 255, 0.16); }}
QPushButton:disabled {{ color: #666970; background: rgba(255, 255, 255, 0.03); }}
QPlainTextEdit {{
    background: transparent;
    border: none;
    font-size: 13px;
    selection-background-color: #34445F;
    selection-color: {TEXT};
}}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px 0px 2px 0px; }}
QScrollBar::handle:vertical {{ background: rgba(255, 255, 255, 0.14); border-radius: 3px; min-height: 28px; margin: 0px 2px; }}
QScrollBar::handle:vertical:hover {{ background: rgba(255, 255, 255, 0.24); }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ height: 0px; }}
QMenu {{
    background: #1F2025;
    border: 1px solid #2E3037;
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 18px 6px 12px; border-radius: 5px; color: {TEXT}; }}
QMenu::item:selected {{ background: rgba(255, 255, 255, 0.08); }}
QMenu::item:disabled {{ color: #666970; }}
QMenu::separator {{ height: 1px; background: #2E3037; margin: 4px 6px; }}
QMenu::indicator {{ width: 12px; height: 12px; left: 4px; }}
QToolTip {{ background: #1F2025; color: {TEXT}; border: 1px solid #2E3037; padding: 4px 6px; }}
QSizeGrip {{ background: transparent; width: 14px; height: 14px; }}
"""
