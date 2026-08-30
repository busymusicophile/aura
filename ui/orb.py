"""
AURA — presence orb (Phase 3).

A small frameless, translucent, always-on-top glow that shows what AURA is doing:
idle, listening, thinking, speaking. It is the only always-visible part of the
system, so it stays deliberately small and quiet.

It also renders the Phase 4 "silent flag". When someone other than Keerthana is
present and an action is waiting, nothing is spoken aloud - the orb turns to the
flagged colour and shows the pending prompt as text only she is looking at. That
is the whole point: the notification must not be audible to the room.

The palette matches the Phase 9 control panel - warm sepia: dusty rose at rest,
amber while thinking, rust when something needs her attention.
"""

from __future__ import annotations

import math
import sys
from enum import Enum

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QRadialGradient
from PySide6.QtWidgets import QApplication, QWidget


class OrbState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FLAGGED = "flagged"
    ERROR = "error"


# (core colour, glow colour, pulse period in seconds)
#
# Warm sepia, matching the Phase 9 control panel: dusty rose at rest, sage when
# listening, amber while thinking, aged paper while speaking, rust for anything
# needing attention. The hues sit close together on purpose, so pulse rate is
# the second signal - idle breathes slowly, thinking is quick, error is urgent.
_PALETTE: dict[OrbState, tuple[QColor, QColor, float]] = {
    OrbState.IDLE: (QColor(179, 147, 132), QColor(84, 66, 58), 5.0),
    OrbState.LISTENING: (QColor(138, 154, 123), QColor(66, 78, 58), 2.4),
    OrbState.THINKING: (QColor(212, 178, 76), QColor(104, 84, 34), 1.1),
    OrbState.SPEAKING: (QColor(232, 221, 208), QColor(126, 106, 90), 1.8),
    OrbState.FLAGGED: (QColor(163, 86, 74), QColor(84, 40, 34), 2.0),
    OrbState.ERROR: (QColor(186, 74, 60), QColor(96, 32, 26), 0.7),
}

_PAPER = QColor(232, 221, 208)
_INK = QColor(36, 29, 26)


class PresenceOrb(QWidget):
    """The always-on-top state indicator."""

    clicked = Signal()
    flag_dismissed = Signal()

    ORB_SIZE = 96
    LABEL_HEIGHT = 22
    FLAG_WIDTH = 300

    def __init__(self) -> None:
        super().__init__()
        self._state = OrbState.IDLE
        self._phase = 0.0
        self._flag_text = ""
        self._drag_origin: QPoint | None = None

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # keeps it off the taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._resize_for_state()
        self._move_to_corner()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 fps is plenty for a slow pulse

    # ------------------------------------------------------------------ setup
    def _resize_for_state(self) -> None:
        if self._flag_text:
            self.resize(self.FLAG_WIDTH, self.ORB_SIZE + self.LABEL_HEIGHT * 3)
        else:
            self.resize(self.ORB_SIZE, self.ORB_SIZE + self.LABEL_HEIGHT)

    def _move_to_corner(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(area.right() - self.width() - 28, area.bottom() - self.height() - 28)

    # ------------------------------------------------------------------- api
    def set_state(self, state: OrbState) -> None:
        if state != self._state:
            self._state = state
            self.update()

    def show_flag(self, text: str) -> None:
        """Display a silent, text-only prompt. Never spoken aloud."""
        self._flag_text = text
        self._state = OrbState.FLAGGED
        self._resize_for_state()
        self._move_to_corner()
        self.update()

    def clear_flag(self) -> None:
        if self._flag_text:
            self._flag_text = ""
            self._state = OrbState.IDLE
            self._resize_for_state()
            self._move_to_corner()
            self.update()
            self.flag_dismissed.emit()

    @property
    def state(self) -> OrbState:
        return self._state

    # ---------------------------------------------------------------- render
    def _tick(self) -> None:
        _, _, period = _PALETTE[self._state]
        self._phase = (self._phase + 0.033 / period) % 1.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001, N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        core, glow, _ = _PALETTE[self._state]
        # Breathing brightness. Idle sits dimmer so it fades into the desktop.
        swell = 0.5 + 0.5 * math.sin(self._phase * 2 * math.pi)
        intensity = (0.35 + 0.30 * swell) if self._state == OrbState.IDLE else (0.55 + 0.45 * swell)

        cx = self.ORB_SIZE / 2 if not self._flag_text else self.ORB_SIZE / 2 + 6
        cy = self.ORB_SIZE / 2
        radius = self.ORB_SIZE / 2 - 6

        gradient = QRadialGradient(cx, cy, radius)
        bright = QColor(core)
        bright.setAlphaF(min(1.0, 0.85 * intensity))
        mid = QColor(glow)
        mid.setAlphaF(0.45 * intensity)
        edge = QColor(glow)
        edge.setAlphaF(0.0)

        gradient.setColorAt(0.0, bright)
        gradient.setColorAt(0.45, mid)
        gradient.setColorAt(1.0, edge)

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2)
        )

        # Solid inner bead so the orb reads clearly on a light desktop too.
        bead = QColor(core)
        bead.setAlphaF(0.9)
        painter.setBrush(bead)
        painter.drawEllipse(int(cx - radius * 0.28), int(cy - radius * 0.28),
                            int(radius * 0.56), int(radius * 0.56))

        painter.setFont(QFont("Georgia", 8))
        painter.setPen(QColor(200, 200, 200, 210))
        if not self._flag_text:
            painter.drawText(
                0, self.ORB_SIZE, self.ORB_SIZE, self.LABEL_HEIGHT,
                Qt.AlignCenter, self._state.value,
            )
        else:
            self._paint_flag(painter)

    def _paint_flag(self, painter: QPainter) -> None:
        """The silent notification card."""
        left = self.ORB_SIZE + 4
        width = self.width() - left - 6
        height = self.height() - 12

        path = QPainterPath()
        path.addRoundedRect(left, 6, width, height, 10, 10)

        backdrop = QColor(46, 38, 34)
        backdrop.setAlphaF(0.93)
        painter.fillPath(path, backdrop)

        border = QColor(_PALETTE[OrbState.FLAGGED][0])
        border.setAlphaF(0.55)
        painter.setPen(border)
        painter.drawPath(path)

        painter.setPen(_PAPER)
        painter.setFont(QFont("Georgia", 9))
        painter.drawText(
            int(left + 12), 14, int(width - 24), int(height - 28),
            Qt.AlignLeft | Qt.TextWordWrap, self._flag_text,
        )

        painter.setFont(QFont("Georgia", 7))
        painter.setPen(QColor(150, 150, 150, 190))
        painter.drawText(
            int(left + 12), int(height - 14), int(width - 24), 16,
            Qt.AlignLeft, "click to dismiss",
        )

    # ----------------------------------------------------------------- input
    def mousePressEvent(self, event) -> None:  # noqa: ANN001, N802
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001, N802
        if self._drag_origin is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001, N802
        moved = False
        if self._drag_origin is not None:
            start = self._drag_origin
            now = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            moved = (start - now).manhattanLength() > 4
        self._drag_origin = None
        if not moved and event.button() == Qt.LeftButton:
            if self._flag_text:
                self.clear_flag()
            else:
                self.clicked.emit()
        event.accept()


def demo() -> int:
    """Cycle every state so the look can be checked without the full system."""
    app = QApplication(sys.argv)
    orb = PresenceOrb()
    orb.show()

    sequence = [
        (OrbState.IDLE, 2200),
        (OrbState.LISTENING, 2200),
        (OrbState.THINKING, 2200),
        (OrbState.SPEAKING, 2200),
        (OrbState.ERROR, 1600),
    ]
    elapsed = 0
    for state, hold in sequence:
        QTimer.singleShot(elapsed, lambda s=state: orb.set_state(s))
        elapsed += hold

    QTimer.singleShot(
        elapsed,
        lambda: orb.show_flag(
            "Someone else is here — go ahead, or remind me later?\n\n"
            "Pending: draft reply to Amma"
        ),
    )
    QTimer.singleShot(elapsed + 5000, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(demo())
