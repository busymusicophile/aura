"""
AURA — control panel (Phase 9).

The house at a glance: device status by room, who is present, recent activity,
and manual overrides.

Aesthetic: warm sepia dark academia, modelled on a Notion study dashboard. Aged
paper on dark walnut, dusty-rose section bars with dark text, muted tag chips,
serif throughout. Four ideas do most of the work:

* Section headers are filled bars, not underlined text - they read as tabs of a
  ledger and give the page its horizontal rhythm.
* The `| title |` pipe framing on every header, which is what makes it feel
  handwritten rather than generated.
* State is carried by small filled chips, so status is scannable without
  reading - amber for on, muted grey for off, rust for unreachable.
* Nothing is pure white or pure black. Everything sits in a narrow warm band,
  so the few coloured chips carry all the emphasis.

At this point no smart-home hardware exists, so the panel runs against
`MockBackend`. `get_backend()` returns the real Home Assistant client the moment
one is reachable, and nothing in this file changes.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from aura import config
from aura.home.registry import Device, DeviceKind, by_room, get_backend
from aura.safety import audit

# --------------------------------------------------------------------------
# Palette — warm sepia
# --------------------------------------------------------------------------

INK = "#241d1a"          # dark walnut, the page
CARD = "#2e2622"         # raised surface
CARD_HI = "#3a302a"      # hover
BORDER = "#403530"       # hairline
PAPER = "#e8ddd0"        # warm off-white, primary text
MUTED = "#a89685"        # secondary text
FAINT = "#7a6b5e"        # tertiary

BAR = "#b39384"          # dusty rose section bar
BAR_TEXT = "#2b211c"     # dark text on the bar
ROOM_BAR = "#6d5a4e"     # smaller room bar
QUOTE_BG = "#3d2a26"     # maroon-tinted epigraph block

AMBER = "#d4b24c"        # a device that is on
SLATE = "#5c5149"        # a device that is off
RUST = "#a3564a"         # unreachable / attention
SAGE = "#8a9a7b"         # healthy system

SERIF = "Georgia"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {INK};
    color: {PAPER};
    font-family: {SERIF};
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: transparent; width: 8px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QFrame#card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}
/* QLabel is a QWidget, so the rule above paints an opaque box behind every
   label. Only the bars, chips and quote are meant to have a fill. */
QLabel {{ background: transparent; }}

QLabel#title {{
    color: {PAPER};
    font-size: 30px;
    letter-spacing: 2px;
}}
QLabel#subtitle {{ color: {FAINT}; font-size: 11px; letter-spacing: 1px; }}

/* Filled section bar with dark text - the signature of this look. */
QLabel#bar {{
    background-color: {BAR};
    color: {BAR_TEXT};
    font-size: 13px;
    letter-spacing: 1px;
    padding: 5px 12px;
    border-radius: 3px;
}}
QLabel#roombar {{
    background-color: {ROOM_BAR};
    color: {PAPER};
    font-size: 10px;
    letter-spacing: 2px;
    padding: 3px 10px;
    border-radius: 3px;
}}
QLabel#quote {{
    background-color: {QUOTE_BG};
    color: {MUTED};
    font-size: 12px;
    font-style: italic;
    padding: 10px 14px;
    border-radius: 3px;
}}
QLabel#device {{ color: {PAPER}; font-size: 13px; }}
QLabel#meta {{ color: {FAINT}; font-size: 11px; }}
QLabel#body {{ color: {MUTED}; font-size: 12px; }}
QLabel#banner {{ border-radius: 4px; }}

QPushButton {{
    background-color: {CARD};
    color: {MUTED};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 5px 14px;
    font-family: {SERIF};
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton:hover {{ background-color: {CARD_HI}; color: {PAPER}; }}
"""

# Chip styling is applied per-widget, since each carries its own colour.
_CHIP = (
    "background-color: {bg}; color: {fg}; font-size: 10px; "
    "letter-spacing: 1px; padding: 2px 9px; border-radius: 3px;"
)

BANNER_FILE = config.DATA_DIR / "banner.jpg"

EPIGRAPH = (
    '"Mi\'ija, in a world as wrong as this one, all we can do is to make things '
    'as right as we can." — Barbara Kingsolver'
)


def bar_label(text: str) -> QLabel:
    """A filled section bar, framed in pipes."""
    label = QLabel(f"| {text} |")
    label.setObjectName("bar")
    label.setSizePolicy(label.sizePolicy().horizontalPolicy(), label.sizePolicy().verticalPolicy())
    return label


def chip(text: str, bg: str, fg: str = "#241d1a") -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(_CHIP.format(bg=bg, fg=fg))
    label.setAlignment(Qt.AlignCenter)
    return label


def card(title: str = "") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(11)
    if title:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(bar_label(title))
        row.addStretch(1)
        layout.addLayout(row)
    return frame, layout


class DeviceRow(QWidget):
    """One device: name, a state chip, and a toggle."""

    def __init__(self, device: Device, on_toggle) -> None:  # noqa: ANN001
        super().__init__()
        self.device = device
        self._on_toggle = on_toggle

        row = QHBoxLayout(self)
        row.setContentsMargins(2, 3, 0, 3)
        row.setSpacing(10)

        self.name = QLabel(device.name)
        self.name.setObjectName("device")
        row.addWidget(self.name, 3)

        self.chip = QLabel()
        self.chip.setAlignment(Qt.AlignCenter)
        self.chip.setMinimumWidth(74)
        row.addWidget(self.chip, 0)

        self.button = QPushButton()
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setMinimumWidth(58)
        self.button.clicked.connect(self._clicked)
        row.addWidget(self.button, 0)

        self.refresh(device)

    def _clicked(self) -> None:
        if self.device.reachable:
            self._on_toggle(self.device)

    def refresh(self, device: Device) -> None:
        self.device = device

        if not device.reachable or device.state == "unavailable":
            self.chip.setText("unreachable")
            self.chip.setStyleSheet(_CHIP.format(bg=RUST, fg="#f0e4da"))
            self.button.setText("—")
            self.button.setEnabled(False)
        elif device.kind == DeviceKind.SENSOR:
            self.chip.setText(device.state)
            self.chip.setStyleSheet(_CHIP.format(bg=SLATE, fg=PAPER))
            self.button.setText("")
            self.button.setEnabled(False)
        elif device.is_on:
            self.chip.setText("on")
            self.chip.setStyleSheet(_CHIP.format(bg=AMBER, fg=BAR_TEXT))
            self.button.setText("turn off")
            self.button.setEnabled(True)
        else:
            self.chip.setText("off")
            self.chip.setStyleSheet(_CHIP.format(bg=SLATE, fg=MUTED))
            self.button.setText("turn on")
            self.button.setEnabled(True)


class ControlPanel(QMainWindow):
    def __init__(self, backend: Any = None, perception: Any = None) -> None:
        super().__init__()
        self.backend = backend or get_backend()
        self.perception = perception
        self._rows: dict[str, DeviceRow] = {}

        self.setWindowTitle("AURA")
        self.resize(1180, 820)
        self.setStyleSheet(STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        banner = self._banner()
        if banner is not None:
            outer.addWidget(banner)

        body = QWidget()
        outer.addWidget(body)
        inner = QVBoxLayout(body)
        inner.setContentsMargins(34, 22, 34, 26)
        inner.setSpacing(16)

        inner.addLayout(self._header())

        epigraph = QLabel(EPIGRAPH)
        epigraph.setObjectName("quote")
        epigraph.setWordWrap(True)
        inner.addWidget(epigraph)

        columns = QHBoxLayout()
        columns.setSpacing(16)
        columns.addWidget(self._presence_column(), 2)
        columns.addWidget(self._devices_column(), 4)
        columns.addWidget(self._activity_column(), 3)
        inner.addLayout(columns)

        inner.addLayout(self._rooms_strip())

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(4000)
        self.refresh()

    # ---------------------------------------------------------------- banner
    def _banner(self) -> QWidget | None:
        """A header image, like the reference dashboard.

        Optional: drop any image at C:\\AURA\\data\\banner.jpg and it appears
        here. Without one, a warm gradient stands in rather than leaving an
        obvious empty slot.
        """
        strip = QLabel()
        strip.setObjectName("banner")
        strip.setFixedHeight(120)
        strip.setScaledContents(True)

        if BANNER_FILE.exists():
            pixmap = QPixmap(str(BANNER_FILE))
            if not pixmap.isNull():
                strip.setPixmap(pixmap)
                return strip

        strip.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            f"stop:0 #1b1512, stop:0.45 {QUOTE_BG}, stop:1 #1b1512);"
        )
        return strip

    # ---------------------------------------------------------------- header
    def _header(self) -> QHBoxLayout:
        bar = QHBoxLayout()

        left = QVBoxLayout()
        left.setSpacing(3)
        title = QLabel("| home <3 |")
        title.setObjectName("title")
        left.addWidget(title)
        self.subtitle = QLabel()
        self.subtitle.setObjectName("subtitle")
        left.addWidget(self.subtitle)
        bar.addLayout(left)

        bar.addStretch(1)

        self.clock = QLabel()
        self.clock.setObjectName("body")
        self.clock.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar.addWidget(self.clock)

        refresh = QPushButton("refresh")
        refresh.setCursor(Qt.PointingHandCursor)
        refresh.clicked.connect(self.refresh)
        bar.addWidget(refresh)

        all_off = QPushButton("all lights off")
        all_off.setCursor(Qt.PointingHandCursor)
        all_off.clicked.connect(self._all_lights_off)
        bar.addWidget(all_off)

        return bar

    # -------------------------------------------------------------- presence
    def _presence_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        frame, inner = card("who's here")
        self.presence_body = QLabel()
        self.presence_body.setObjectName("body")
        self.presence_body.setWordWrap(True)
        inner.addWidget(self.presence_body)
        layout.addWidget(frame)

        frame2, inner2 = card("system")
        self.system_rows = QVBoxLayout()
        self.system_rows.setSpacing(6)
        inner2.addLayout(self.system_rows)
        layout.addWidget(frame2)

        layout.addStretch(1)
        return column

    # --------------------------------------------------------------- devices
    def _devices_column(self) -> QWidget:
        frame, layout = card("the house")
        self.backend_label = QLabel()
        self.backend_label.setObjectName("meta")
        layout.addWidget(self.backend_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.device_layout = QVBoxLayout(holder)
        self.device_layout.setContentsMargins(0, 4, 10, 0)
        self.device_layout.setSpacing(5)
        scroll.setWidget(holder)
        layout.addWidget(scroll)

        self._build_devices()
        return frame

    def _build_devices(self) -> None:
        devices = self.backend.devices()
        for room, items in by_room(devices).items():
            header = QHBoxLayout()
            header.setContentsMargins(0, 8, 0, 2)
            label = QLabel(room.upper())
            label.setObjectName("roombar")
            header.addWidget(label)
            header.addStretch(1)
            self.device_layout.addLayout(header)

            for device in items:
                row = DeviceRow(device, self._toggle)
                self._rows[device.id] = row
                self.device_layout.addWidget(row)
        self.device_layout.addStretch(1)

    # ----------------------------------------------------------- rooms strip
    def _rooms_strip(self) -> QHBoxLayout:
        """One small card per room, echoing the weekday columns in the design.

        Redundant with the device list on purpose: this answers "is anything on
        in the kitchen" at a glance, without reading a list.
        """
        strip = QHBoxLayout()
        strip.setSpacing(12)
        self._room_cards: dict[str, QLabel] = {}

        for room in by_room(self.backend.devices()):
            frame = QFrame()
            frame.setObjectName("card")
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 10, 12, 12)
            box.setSpacing(8)

            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            label = QLabel(room.upper())
            label.setObjectName("roombar")
            header.addWidget(label)
            header.addStretch(1)
            box.addLayout(header)

            summary = QLabel()
            summary.setObjectName("body")
            box.addWidget(summary)

            self._room_cards[room] = summary
            strip.addWidget(frame, 1)

        return strip

    def _refresh_rooms(self) -> None:
        for room, devices in by_room(self.backend.devices()).items():
            summary = getattr(self, "_room_cards", {}).get(room)
            if summary is None:
                continue
            on = [d for d in devices if d.is_on]
            dead = [d for d in devices if not d.reachable]
            if on:
                text = ", ".join(d.name.lower() for d in on)
            else:
                text = "all off"
            if dead:
                text += f"\n{len(dead)} unreachable"
            summary.setText(text)

    # -------------------------------------------------------------- activity
    def _activity_column(self) -> QWidget:
        frame, layout = card("recent activity")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.activity_layout = QVBoxLayout(holder)
        self.activity_layout.setContentsMargins(0, 4, 10, 0)
        self.activity_layout.setSpacing(7)
        self.activity_layout.addStretch(1)
        scroll.setWidget(holder)
        layout.addWidget(scroll)
        return frame

    # --------------------------------------------------------------- actions
    def _toggle(self, device: Device) -> None:
        target = "off" if device.is_on else "on"
        try:
            updated = self.backend.set_state(device.id, target)
            self._rows[device.id].refresh(updated)
            audit.record(
                audit.Event.DEVICE_COMMAND,
                actor="panel",
                detail={"device": device.name, "room": device.room, "state": target},
                outcome="ok",
            )
        except Exception as exc:  # noqa: BLE001
            audit.record(
                audit.Event.ACTION_FAILED,
                actor="panel",
                detail={"device": device.name, "error": str(exc)},
                outcome="failed",
            )
        self.refresh()

    def _all_lights_off(self) -> None:
        for device in self.backend.devices():
            if device.kind == DeviceKind.LIGHT and device.is_on and device.reachable:
                try:
                    self.backend.set_state(device.id, "off")
                except Exception:  # noqa: BLE001
                    pass
        self.refresh()

    # --------------------------------------------------------------- refresh
    def refresh(self) -> None:
        now = datetime.now()
        self.clock.setText(now.strftime("%A, %d %B  ·  %H:%M"))
        self.backend_label.setText(f"source — {self.backend.describe_backend()}")

        devices = self.backend.devices()
        for device in devices:
            row = self._rows.get(device.id)
            if row is not None:
                row.refresh(device)

        on_count = sum(1 for d in devices if d.is_on)
        unreachable = sum(1 for d in devices if not d.reachable)
        self.subtitle.setText(
            f"{len(devices)} devices  ·  {on_count} on"
            + (f"  ·  {unreachable} unreachable" if unreachable else "")
        )

        self._refresh_presence()
        self._refresh_rooms()
        self._refresh_system()
        self._refresh_activity()

    def _refresh_presence(self) -> None:
        if self.perception is None:
            self.presence_body.setText(
                "Perception is not running.\n\nStart the assistant to see who is "
                "in the room."
            )
            return
        state = self.perception.state()
        if not state.people:
            self.presence_body.setText("Nobody visible.")
            return
        lines = []
        for person in state.people:
            label = person.name if person.is_known else "unrecognised"
            lines.append(f"{label} — {person.seconds_present() / 60:.0f} min")
        self.presence_body.setText("\n".join(lines))

    def _refresh_system(self) -> None:
        from aura.runtime import gpu_report

        while self.system_rows.count():
            item = self.system_rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

        info = gpu_report()
        entries = [
            ("gpu", "ready" if info.get("torch_cuda") else "cpu only",
             SAGE if info.get("torch_cuda") else RUST),
            ("vision", "cuda" if info.get("onnx_cuda") else "cpu",
             SAGE if info.get("onnx_cuda") else RUST),
        ]
        if "vram_free_gb" in info:
            entries.append(
                ("vram", f"{info['vram_free_gb']} / {info.get('vram_gb')} GB free", SLATE)
            )

        for name, value, colour in entries:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            key = QLabel(name)
            key.setObjectName("meta")
            key.setMinimumWidth(56)
            row.addWidget(key)
            row.addWidget(chip(value, colour, BAR_TEXT if colour is not SLATE else PAPER))
            row.addStretch(1)
            self.system_rows.addLayout(row)

    def _refresh_activity(self) -> None:
        while self.activity_layout.count() > 1:
            item = self.activity_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for entry in reversed(audit.read(limit=16)):
            stamp = entry.get("ts", "")[11:16]
            event = entry.get("event", "").replace("_", " ")
            detail = entry.get("detail", {}) or {}
            summary = (
                detail.get("summary") or detail.get("device") or detail.get("name")
                or detail.get("kind") or entry.get("outcome", "")
            )
            text = f"{stamp}  ·  {event}"
            if summary:
                text += f"  —  {str(summary)[:40]}"
            label = QLabel(text)
            label.setObjectName("meta")
            label.setWordWrap(True)
            self.activity_layout.insertWidget(self.activity_layout.count() - 1, label)


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA control panel (Phase 9)")
    parser.add_argument("--mock", action="store_true", help="force mock devices")
    args = parser.parse_args()

    bootstrap("panel")

    app = QApplication(sys.argv)
    app.setFont(QFont(SERIF, 10))

    backend = None
    if args.mock:
        from aura.home.registry import MockBackend

        backend = MockBackend()

    panel = ControlPanel(backend=backend)
    panel.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
