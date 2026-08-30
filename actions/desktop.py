"""
AURA — desktop control (Phase 5).

Launching applications, focusing windows, and reading what is open, on Windows.

Application launching resolves through a small alias table first, then PATH, then
the Start Menu shortcuts. The Start Menu lookup is what makes "open spotify" work
without hardcoding install paths - shortcuts are where Windows already records
that mapping.

Closing a window is treated as irreversible. It is not, strictly, but it can
discard unsaved work, and the confirmation prompt says so.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from aura.actions.base import Action, ActionBroker, ActionKind

# Spoken name -> executable or URI.
APP_ALIASES: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "explorer": "explorer.exe",
    "files": "explorer.exe",
    "file explorer": "explorer.exe",
    "terminal": "wt.exe",
    "command prompt": "cmd.exe",
    "powershell": "powershell.exe",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "control panel": "control.exe",
    "chrome": "chrome.exe",
    "edge": "msedge.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "vs code": "code.exe",
    "vscode": "code.exe",
}

_START_MENU_DIRS = [
    Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
]


@dataclass
class WindowInfo:
    title: str
    process: str
    handle: int
    visible: bool


def resolve_app(name: str) -> str | None:
    """Find something launchable for a spoken application name."""
    import shutil

    key = name.strip().lower()
    if key in APP_ALIASES:
        target = APP_ALIASES[key]
        if target.endswith(":"):  # a URI like ms-settings:
            return target
        found = shutil.which(target)
        return found or target

    found = shutil.which(key) or shutil.which(f"{key}.exe")
    if found:
        return found

    # Start Menu shortcuts: how Windows already maps names to programs.
    for root in _START_MENU_DIRS:
        if not root.exists():
            continue
        for shortcut in root.rglob("*.lnk"):
            if shortcut.stem.lower() == key:
                return str(shortcut)
    for root in _START_MENU_DIRS:
        if not root.exists():
            continue
        for shortcut in root.rglob("*.lnk"):
            if key in shortcut.stem.lower():
                return str(shortcut)
    return None


def list_windows(visible_only: bool = True) -> list[WindowInfo]:
    """Every top-level window currently open."""
    import psutil
    import win32gui
    import win32process

    results: list[WindowInfo] = []

    def collect(hwnd: int, _: Any) -> None:
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        visible = bool(win32gui.IsWindowVisible(hwnd))
        if visible_only and not visible:
            return
        process = ""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid).name()
        except Exception:  # noqa: BLE001
            pass
        results.append(WindowInfo(title=title, process=process, handle=hwnd, visible=visible))

    win32gui.EnumWindows(collect, None)
    return results


def focus_window(match: str) -> bool:
    """Bring the first window whose title contains `match` to the front."""
    import win32con
    import win32gui

    needle = match.lower()
    for window in list_windows():
        if needle in window.title.lower():
            try:
                win32gui.ShowWindow(window.handle, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(window.handle)
                logger.info("focused: {}", window.title)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error("could not focus {}: {}", window.title, exc)
                return False
    logger.warning("no window matching '{}'", match)
    return False


def _launch(target: str) -> str:
    if target.endswith(":"):
        os.startfile(target)  # noqa: S606 - a Windows URI, not a shell string
    elif target.lower().endswith(".lnk"):
        os.startfile(target)  # noqa: S606
    else:
        subprocess.Popen([target], shell=False)
    return target


def _close(match: str) -> bool:
    import win32con
    import win32gui

    needle = match.lower()
    for window in list_windows():
        if needle in window.title.lower():
            win32gui.PostMessage(window.handle, win32con.WM_CLOSE, 0, 0)
            logger.info("closed: {}", window.title)
            return True
    return False


class DesktopController:
    """Proposes desktop actions through the broker."""

    def propose_launch(
        self, app_name: str, broker: ActionBroker, state: Any = None
    ) -> Action:
        target = resolve_app(app_name)
        if target is None:
            action = Action(
                kind=ActionKind.DESKTOP,
                summary=f"Open {app_name}",
                run=lambda: None,
            )
            action.error = f"could not find an application called '{app_name}'"
            logger.warning(action.error)
            return action

        return broker.propose(
            Action(
                kind=ActionKind.DESKTOP,
                summary=f"Open {app_name}",
                preview=f"Launch: {target}",
                run=lambda: _launch(target),
                reversible=True,
            ),
            state,
        )

    def propose_close(
        self, window_match: str, broker: ActionBroker, state: Any = None
    ) -> Action:
        return broker.propose(
            Action(
                kind=ActionKind.DESKTOP,
                summary=f"Close the window matching '{window_match}'",
                preview="Any unsaved work in that window may be lost.",
                run=lambda: _close(window_match),
                reversible=False,
            ),
            state,
        )

    def focus(self, window_match: str) -> bool:
        """Focusing changes nothing and needs no confirmation."""
        return focus_window(window_match)

    def windows(self) -> list[WindowInfo]:
        return list_windows()


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA desktop control (Phase 5)")
    parser.add_argument("--list", action="store_true", help="list open windows")
    parser.add_argument("--resolve", metavar="APP", help="resolve an app name")
    parser.add_argument("--open", metavar="APP", help="propose opening an app")
    parser.add_argument("--focus", metavar="TEXT", help="focus a window by title")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    bootstrap("desktop")

    if args.list:
        for window in list_windows():
            print(f"{window.process:<24} {window.title[:70]}")
        return 0

    if args.resolve:
        print(resolve_app(args.resolve) or "not found")
        return 0

    if args.focus:
        return 0 if focus_window(args.focus) else 1

    if args.open:
        broker = ActionBroker()
        controller = DesktopController()
        action = controller.propose_launch(args.open, broker)
        if action.error:
            print(action.error)
            return 1
        print("\n--- proposed ---")
        print(action.describe())
        print("----------------")
        answer = "y" if args.yes else input("confirm? [y/N] ").strip().lower()
        if answer != "y":
            broker.reject(action.id, "declined at prompt")
            print("cancelled - nothing was launched")
            return 0
        done = broker.confirm(action.id)
        print(f"failed: {done.error}" if done.error else f"launched: {done.result}")
        return 1 if done.error else 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
