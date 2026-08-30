"""
AURA — browser control (Phase 5).

Playwright-driven Chromium. Every navigation is proposed through the action
broker first, so nothing opens without a yes.

The browser runs non-headless and persists its profile under the AURA data
directory. Persisting matters: a fresh context every time means logged-out
everything, and a visible window matters because she should be able to see what
AURA is doing in her name rather than having it happen invisibly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from loguru import logger

from aura import config
from aura.actions.base import Action, ActionBroker, ActionKind
from aura.safety import redaction

_SEARCH_URL = "https://duckduckgo.com/?q={query}"

# Spoken shorthand -> real URL, so "open youtube" works.
KNOWN_SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "wikipedia": "https://www.wikipedia.org",
    "maps": "https://maps.google.com",
    "drive": "https://drive.google.com",
    "calendar": "https://calendar.google.com",
    "whatsapp": "https://web.whatsapp.com",
    "kite": "https://kite.zerodha.com",
    "chatgpt": "https://chat.openai.com",
}


@dataclass
class PageSnapshot:
    url: str
    title: str
    text: str


class BrowserController:
    """Owns one persistent Chromium context."""

    def __init__(self, headless: bool = False) -> None:
        self.headless = headless
        self.profile_dir = config.DATA_DIR / "browser"
        self._playwright: Any = None
        self._context: Any = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- launch
    def _ensure(self) -> Any:
        with self._lock:
            if self._context is not None:
                return self._context

            from playwright.sync_api import sync_playwright

            self.profile_dir.mkdir(parents=True, exist_ok=True)
            self._playwright = sync_playwright().start()
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=self.headless,
                viewport={"width": 1280, "height": 800},
                args=["--disable-blink-features=AutomationControlled"],
            )
            logger.info("browser ready (profile: {})", self.profile_dir)
            return self._context

    def _page(self) -> Any:
        context = self._ensure()
        if context.pages:
            return context.pages[-1]
        return context.new_page()

    # ----------------------------------------------------------- raw actions
    def _open(self, url: str) -> PageSnapshot:
        page = self._page()
        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        return PageSnapshot(url=page.url, title=page.title(), text="")

    def _read(self, limit: int = 4000) -> PageSnapshot:
        page = self._page()
        body = page.inner_text("body")[:limit]
        # Web pages are untrusted input and routinely contain things like card
        # forms and OTP text. Filter before this reaches memory or the model.
        return PageSnapshot(
            url=page.url,
            title=redaction.redact(page.title()).text,
            text=redaction.redact(body).text,
        )

    def close(self) -> None:
        with self._lock:
            if self._context is not None:
                try:
                    self._context.close()
                except Exception:  # noqa: BLE001
                    pass
                self._context = None
            if self._playwright is not None:
                try:
                    self._playwright.stop()
                except Exception:  # noqa: BLE001
                    pass
                self._playwright = None

    # -------------------------------------------------------------- proposals
    def propose_open(
        self, target: str, broker: ActionBroker, state: Any = None
    ) -> Action:
        """Propose opening a site. Accepts a shorthand, a domain or a full URL."""
        url = resolve_target(target)
        action = Action(
            kind=ActionKind.BROWSER,
            summary=f"Open {url} in the browser",
            preview=f"Navigate to: {url}",
            run=lambda: self._open(url),
            reversible=True,
        )
        return broker.propose(action, state)

    def propose_search(
        self, query: str, broker: ActionBroker, state: Any = None
    ) -> Action:
        clean = redaction.redact(query).text
        url = _SEARCH_URL.format(query=clean.replace(" ", "+"))
        action = Action(
            kind=ActionKind.BROWSER,
            summary=f'Search the web for "{clean}"',
            preview=f"Search URL: {url}",
            run=lambda: self._open(url),
            reversible=True,
        )
        return broker.propose(action, state)

    def read_current(self) -> PageSnapshot:
        """Reading is not an action - it changes nothing outside the machine."""
        return self._read()


def resolve_target(target: str) -> str:
    """Turn 'youtube', 'example.com' or a full URL into a URL."""
    text = target.strip().lower().removeprefix("open ").strip()
    if text in KNOWN_SITES:
        return KNOWN_SITES[text]
    if text.startswith(("http://", "https://")):
        return target.strip()
    if "." in text and " " not in text:
        return f"https://{text}"
    return _SEARCH_URL.format(query=text.replace(" ", "+"))


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA browser control (Phase 5)")
    parser.add_argument("target", nargs="?", default="wikipedia")
    parser.add_argument("--yes", action="store_true", help="confirm without prompting")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--read", action="store_true", help="print page text after opening")
    args = parser.parse_args()

    bootstrap("browser")
    broker = ActionBroker()
    controller = BrowserController(headless=args.headless)

    action = controller.propose_open(args.target, broker)
    print("\n--- proposed ---")
    print(action.describe())
    print("----------------")

    answer = "y" if args.yes else input("confirm? [y/N] ").strip().lower()
    if answer != "y":
        broker.reject(action.id, "declined at prompt")
        print("cancelled - nothing was opened")
        return 0

    done = broker.confirm(action.id)
    if done.error:
        print(f"failed: {done.error}")
        return 1
    print(f"opened: {done.result.title}")

    if args.read:
        snapshot = controller.read_current()
        print(f"\n--- {snapshot.title} ---\n{snapshot.text[:800]}")

    controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
