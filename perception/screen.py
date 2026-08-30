"""
AURA — screen context (Phase 2).

Reads what is on screen, on demand only. Never on a timer.

That is a deliberate privacy decision, not a performance one. Continuously OCRing
the screen would mean everything Keerthana ever looks at - messages, documents,
banking pages - flowing through the pipeline and into memory. Instead the screen
is read only when a request needs it ("what am I looking at?"), and the result is
redacted before it goes anywhere.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from aura.safety import redaction

_TESSERACT_HINTS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


@dataclass
class ScreenContext:
    window_title: str
    process_name: str
    text: str
    ocr_available: bool
    redacted: bool

    def summary(self, limit: int = 600) -> str:
        parts = [f"Active window: {self.window_title or 'unknown'}"]
        if self.process_name:
            parts.append(f"Application: {self.process_name}")
        if self.text:
            body = self.text[:limit]
            parts.append(f"Visible text:\n{body}{'...' if len(self.text) > limit else ''}")
        elif not self.ocr_available:
            parts.append("(screen text unavailable - Tesseract is not installed)")
        return "\n".join(parts)


def find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _TESSERACT_HINTS:
        if candidate.exists():
            return str(candidate)
    return None


def active_window() -> tuple[str, str]:
    """(title, process name) of the foreground window."""
    title, process = "", ""
    try:
        import win32gui
        import win32process
        import psutil

        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd) or ""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid:
            try:
                process = psutil.Process(pid).name()
            except psutil.Error:
                process = ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("active window lookup failed: {}", exc)
    return title, process


def grab_screen(monitor: int = 1) -> np.ndarray:
    """Screenshot the given monitor as a BGR array."""
    import mss

    with mss.mss() as sct:
        target = sct.monitors[monitor]
        shot = sct.grab(target)
        frame = np.array(shot)  # BGRA
    return frame[:, :, :3]


def ocr(image: np.ndarray) -> tuple[str, bool]:
    """Extract text. Returns (text, ocr_available)."""
    exe = find_tesseract()
    if not exe:
        return "", False
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = exe
        text = pytesseract.image_to_string(image)
        return text.strip(), True
    except Exception as exc:  # noqa: BLE001
        logger.error("OCR failed: {}", exc)
        return "", False


def read_screen(include_text: bool = True, monitor: int = 1) -> ScreenContext:
    """Capture current screen context, redacted."""
    title, process = active_window()

    raw_text = ""
    ocr_available = True
    if include_text:
        raw_text, ocr_available = ocr(grab_screen(monitor))

    # The screen is the single most likely place to encounter excluded data -
    # a banking tab, an OTP notification, a password manager. Filter before
    # this text reaches memory, the model, or the log.
    title_result = redaction.redact(title)
    text_result = redaction.redact(raw_text)
    was_redacted = not (title_result.is_clean and text_result.is_clean)
    if was_redacted:
        logger.warning(
            "redacted screen content: {} {}", title_result.summary(), text_result.summary()
        )

    return ScreenContext(
        window_title=title_result.text,
        process_name=process,
        text=text_result.text,
        ocr_available=ocr_available,
        redacted=was_redacted,
    )


def main() -> int:
    from aura.runtime import bootstrap

    bootstrap("screen")
    exe = find_tesseract()
    print(f"tesseract: {exe or 'NOT INSTALLED (titles only)'}\n")
    context = read_screen()
    print(context.summary())
    if context.redacted:
        print("\n[excluded data was filtered out of this capture]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
