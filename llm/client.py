"""
AURA — local LLM client (Phase 1).

Thin wrapper over Ollama running qwen3:8b. Local only; no cloud endpoint is ever
contacted.

Everything sent to the model passes through the redaction filter first. That is
enforced here rather than at each call site, so a future feature cannot forget to
do it - the filter is not optional and there is no bypass argument.

Everything the model returns is stripped of emoji, for the same reason. She was
explicit that she does not want AURA using them, and the persona's system
prompt already says so as an instruction - but an instruction is a request to a
small model, not a guarantee, and it was in fact observed ignoring it (a plain
"how's it going" answered with "I 😊"). This is the one place every reply passes
through regardless of caller - `complete()`, `chat()` and `stream()` are used
directly by the assistant, the persona builder, message drafting and the
autonomy engine, not only through AuraBrain - so stripping here is the only
place a single fix covers all of them.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from ollama import Client, ResponseError

from aura import config
from aura.safety import redaction

# Same glyph ranges as aura.voice.tts._UNSPEAKABLE and aura.persona.profile._EMOJI.
# Kept as its own copy rather than a shared import - three modules independently
# decided emoji do not belong in their output, for three different reasons, and
# tying them to one shared constant would make future changes to any one of
# them accidentally affect the other two.
_NO_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE0F\U0000200D"
    "]+"
)


@dataclass
class Message:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """A rolling message list with a fixed system prompt."""

    system: str
    messages: list[Message] = field(default_factory=list)
    max_turns: int = 20

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))
        # Keep the tail; the long-term store is what remembers the rest.
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2 :]

    def payload(self, extra_system: str = "") -> list[dict[str, str]]:
        system = self.system if not extra_system else f"{self.system}\n\n{extra_system}"
        return [{"role": "system", "content": system}] + [
            m.as_dict() for m in self.messages
        ]


class LLMError(RuntimeError):
    pass


# Substrings that mark a crashed-and-restartable llama-server, as opposed to a
# genuine rejection (unknown model, malformed request) which must not be retried.
_TRANSIENT_MARKERS = (
    "0xc0000409",                      # STATUS_STACK_BUFFER_OVERRUN on Windows
    "cuda error",
    "shared object initialization failed",
    "llama-server process has terminated",
    "llama runner process has terminated",
    "no longer running",
    "connection refused",
    "connection reset",
    "remote end closed",
    "timed out",
)


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class LLMClient:
    """Ollama chat wrapper with redaction enforced on the way in."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float | None = None,
    ) -> None:
        s = config.SETTINGS.llm
        self.model = model or s.model
        self.host = host or s.host
        self.temperature = s.temperature if temperature is None else temperature
        self._client = Client(host=self.host)
        self.max_retries = 3
        self.retry_delay = 1.5
        self._supports_think: bool | None = None

    # ------------------------------------------------------------------ util
    def _options(self, **overrides: Any) -> dict[str, Any]:
        cfg = config.SETTINGS.llm
        opts = {
            "temperature": self.temperature,
            "num_ctx": cfg.context_tokens,
            # A ceiling on the reply and a repetition penalty. Without both, a
            # small model under a long system prompt has been observed to spiral
            # into thousands of tokens of open or repetitive generation with
            # nothing to stop it - a real multi-minute hang, not a theoretical one.
            "num_predict": cfg.max_reply_tokens,
            "repeat_penalty": cfg.repeat_penalty,
            "repeat_last_n": cfg.repeat_last_n,
        }
        opts.update(overrides)
        return opts

    @staticmethod
    def _content(response: Any) -> str:
        """Ollama returns a pydantic object; older versions return a dict."""
        message = getattr(response, "message", None)
        if message is not None:
            return getattr(message, "content", "") or ""
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "") or ""
        return ""

    @staticmethod
    def _clean(text: str) -> str:
        """Strip qwen3's reasoning block if the server leaves it inline."""
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        return _NO_EMOJI.sub("", text).strip()

    def _scrub_payload(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        scrubbed: list[dict[str, str]] = []
        for m in messages:
            result = redaction.redact(m["content"])
            if not result.is_clean:
                logger.warning(
                    "redacted excluded data before it reached the model: {}",
                    result.summary(),
                )
            scrubbed.append({"role": m["role"], "content": result.text})
        return scrubbed

    def _call(self, messages: list[dict[str, str]], think: bool, stream: bool) -> Any:
        """Call ollama, degrading gracefully if the server rejects `think`."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": self._options(),
            "stream": stream,
        }
        if self._supports_think is not False:
            kwargs["think"] = think
        try:
            result = self._client.chat(**kwargs)
            self._supports_think = True
            return result
        except (ResponseError, TypeError) as exc:
            if self._supports_think is None and "think" in str(exc).lower():
                logger.debug("server rejected `think`; retrying without it")
                self._supports_think = False
                kwargs.pop("think", None)
                return self._client.chat(**kwargs)
            raise

    def _call_with_retry(
        self, messages: list[dict[str, str]], think: bool, stream: bool
    ) -> Any:
        """Retry transient llama-server crashes.

        ollama's llama-server intermittently dies during CUDA initialisation
        with `exit status 0xc0000409: CUDA error: shared object initialization
        failed`. It is not deterministic and not VRAM exhaustion - it has been
        observed with 5GB of the 6GB card free, and the identical request
        succeeds on the very next attempt against the same runner.

        ollama respawns llama-server on the following request, so a retry is
        genuinely all that is needed. Retrying is limited to errors that look
        like that crash: a bad model name or a malformed request must fail
        immediately rather than being tried three times.
        """
        last: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._call(messages, think, stream)
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if not _is_transient(exc):
                    raise
                last = exc
                if attempt == self.max_retries:
                    break
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    "llama-server crashed ({}). retrying in {:.1f}s "
                    "[attempt {}/{}]",
                    str(exc).split(":")[0][:60], delay, attempt + 1, self.max_retries,
                )
                time.sleep(delay)

        raise LLMError(
            "ollama's llama-server crashed on every attempt. This is usually "
            "transient - try again, or restart Ollama from the system tray. "
            f"Last error: {last}"
        ) from last

    # ------------------------------------------------------------------- api
    def complete(
        self,
        user: str,
        system: str = "",
        think: bool | None = None,
    ) -> str:
        """One-shot completion."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        use_think = config.SETTINGS.llm.think if think is None else think
        try:
            response = self._call_with_retry(self._scrub_payload(messages), use_think, stream=False)
        except ResponseError as exc:
            raise LLMError(f"ollama refused the request: {exc}") from exc
        except ConnectionError as exc:
            raise LLMError(
                f"cannot reach ollama at {self.host} - is the server running?"
            ) from exc
        return self._clean(self._content(response))

    def chat(
        self,
        conversation: Conversation,
        extra_system: str = "",
        think: bool | None = None,
    ) -> str:
        use_think = config.SETTINGS.llm.think if think is None else think
        payload = self._scrub_payload(conversation.payload(extra_system))
        try:
            response = self._call_with_retry(payload, use_think, stream=False)
        except ResponseError as exc:
            raise LLMError(f"ollama refused the request: {exc}") from exc
        return self._clean(self._content(response))

    def stream(
        self,
        conversation: Conversation,
        extra_system: str = "",
        think: bool | None = None,
    ) -> Iterator[str]:
        """Yield reply chunks as they arrive, for a responsive CLI and TTS."""
        use_think = config.SETTINGS.llm.think if think is None else think
        payload = self._scrub_payload(conversation.payload(extra_system))
        in_reasoning = False
        try:
            for part in self._call_with_retry(payload, use_think, stream=True):
                chunk = self._content(part)
                if not chunk:
                    continue
                # Suppress any reasoning block that leaks into the stream.
                if "<think>" in chunk:
                    in_reasoning = True
                    chunk = chunk.split("<think>", 1)[0]
                if in_reasoning:
                    if "</think>" not in chunk:
                        continue
                    chunk = chunk.split("</think>", 1)[1]
                    in_reasoning = False
                # An emoji is a single Unicode scalar value and ollama streams
                # complete decoded characters per chunk, so unlike the excluded-
                # data filter (aura.safety.redaction.StreamRedactor) this needs
                # no cross-chunk buffering - a plain per-chunk strip is safe.
                chunk = _NO_EMOJI.sub("", chunk)
                if chunk:
                    yield chunk
        except ResponseError as exc:
            raise LLMError(f"ollama refused the request: {exc}") from exc

    # ------------------------------------------------------------ diagnostics
    def health(self) -> dict[str, Any]:
        """Check the server is up and the model is present."""
        try:
            listing = self._client.list()
            models = [
                getattr(m, "model", None) or m.get("model", "")
                for m in getattr(listing, "models", None) or listing.get("models", [])
            ]
            return {
                "ok": self.model in models,
                "host": self.host,
                "model": self.model,
                "available": models,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "host": self.host, "error": str(exc)}
