"""
AURA — conversational core (Phase 1).

Ties persona, memory and the local LLM into one loop. Phase 3 reuses `AuraBrain`
behind voice; this module is the text-only proof that the core works.

Run it:
    C:\\AURA\\venv\\Scripts\\python.exe -m aura.chat

Quit, restart, and ask about something from the previous session - if it
remembers, Phase 1 is doing its job.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

from loguru import logger
from rich.console import Console
from rich.panel import Panel

from aura import config
from aura.llm.client import Conversation, LLMClient, LLMError
from aura.memory.store import KIND_CORRECTION, MemoryStore
from aura.persona import profile
from aura.runtime import bootstrap
from aura.safety import audit, redaction

console = Console()

# Phrases that mark an explicit correction. Kept deliberately tight: a false
# positive permanently over-weights an ordinary remark, which is worse than
# missing one, since she can always use /correct.
_CORRECTION_PATTERNS = [
    re.compile(r"(?i)\bno[,;]?\s+(?:i\s+)?(?:meant|mean)\b"),
    re.compile(r"(?i)\bthat'?s\s+(?:not\s+right|wrong|incorrect)\b"),
    re.compile(r"(?i)\bactually[,;]?\s+(?:it'?s|its|i|the)\b"),
    re.compile(r"(?i)\bi\s+(?:already\s+)?told\s+you\b"),
    re.compile(r"(?i)\bstop\s+(?:saying|calling|doing)\b"),
    re.compile(r"(?i)\bcorrection\b"),
    re.compile(r"(?i)\bremember\s+that\b"),
]


def looks_like_correction(text: str) -> bool:
    return any(p.search(text) for p in _CORRECTION_PATTERNS)


@dataclass
class BrainReply:
    text: str
    used_memories: int
    was_correction: bool


class AuraBrain:
    """Persona + memory + LLM. The reusable core behind every interface."""

    def __init__(self, tier_instruction: str = "", speaker: str | None = None) -> None:
        self.persona = profile.load()
        self.memory = MemoryStore()
        self.llm = LLMClient()
        self.speaker = speaker or config.SETTINGS.primary_user
        self.tier_instruction = tier_instruction
        self.conversation = Conversation(
            system=self.persona.system_prompt(tier_instruction)
        )

    # ------------------------------------------------------------------ core
    def _recall(self, user_text: str) -> str:
        try:
            return self.memory.recall_block(user_text)
        except Exception as exc:  # noqa: BLE001 - memory must never break chat
            logger.error("memory recall failed: {}", exc)
            return ""

    def respond(self, user_text: str, stream: bool = False):
        """Produce a reply. Returns BrainReply, or a generator when streaming."""
        clean = redaction.redact(user_text)
        if not clean.is_clean:
            console.print(
                f"[yellow]Filtered before processing: {clean.summary()}[/yellow]"
            )
            audit.record(
                audit.Event.REDACTION,
                actor=self.speaker,
                detail={"where": "chat_input", "categories": clean.summary()},
            )

        text = clean.text
        recall = self._recall(text)
        self.conversation.add("user", text)

        if stream:
            return self._respond_streaming(text, recall)

        try:
            reply = self.llm.chat(self.conversation, extra_system=recall)
        except LLMError as exc:
            self.conversation.messages.pop()
            raise

        return self._finish(text, reply, recall)

    def _respond_streaming(self, text: str, recall: str):
        # Filter the outgoing stream too. A secret can be split across token
        # boundaries ("4111 1111 " then "1111 1111"), so scanning each chunk on
        # its own would find nothing and let the whole value through.
        redactor = redaction.StreamRedactor()
        emitted: list[str] = []

        for chunk in self.llm.stream(self.conversation, extra_system=recall):
            safe = redactor.feed(chunk)
            if safe:
                emitted.append(safe)
                yield safe

        tail = redactor.flush()
        if tail:
            emitted.append(tail)
            yield tail

        if not redactor.is_clean:
            logger.warning("redacted excluded data from a streamed reply")
            audit.record(
                audit.Event.REDACTION,
                actor=self.speaker,
                detail={"where": "stream_output"},
            )

        self._finish(text, "".join(emitted), recall)

    def _finish(self, user_text: str, reply: str, recall: str) -> BrainReply:
        reply = reply.strip()
        self.conversation.add("assistant", reply)

        was_correction = looks_like_correction(user_text)
        if was_correction:
            self.memory.add_correction(user_text, speaker=self.speaker)
        self.memory.add_exchange(user_text, reply, speaker=self.speaker)

        audit.record(
            audit.Event.MEMORY_WRITE,
            actor=self.speaker,
            detail={"kind": KIND_CORRECTION if was_correction else "exchange"},
        )
        return BrainReply(
            text=reply,
            used_memories=recall.count("\n- ") if recall else 0,
            was_correction=was_correction,
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_BANNER = """[bold]AURA[/bold] — local, private, offline.
Commands: [cyan]/memories[/cyan] [cyan]/stats[/cyan] [cyan]/correct <text>[/cyan] [cyan]/forget <id>[/cyan] [cyan]/quit[/cyan]"""


def _handle_command(cmd: str, brain: AuraBrain) -> bool:
    """Returns True if the loop should continue, False to exit."""
    parts = cmd.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit", "/q"):
        return False

    if name == "/stats":
        stats = brain.memory.stats()
        health = brain.llm.health()
        console.print(
            Panel(
                f"memories : {stats}\n"
                f"persona  : {brain.persona.path.name} "
                f"(template={brain.persona.is_template}, reviewed={brain.persona.reviewed})\n"
                f"model    : {health.get('model')} ok={health.get('ok')}",
                title="status",
                border_style="dim",
            )
        )
    elif name == "/memories":
        rows = brain.memory.recent(limit=10)
        if not rows:
            console.print("[dim]nothing stored yet[/dim]")
        for m in rows:
            console.print(
                f"[dim]{m.timestamp}[/dim] [cyan]{m.kind:<10}[/cyan] "
                f"{m.text[:100]}{'...' if len(m.text) > 100 else ''}"
            )
    elif name == "/correct":
        if not arg:
            console.print("[yellow]usage: /correct <what AURA got wrong>[/yellow]")
        else:
            brain.memory.add_correction(arg, speaker=brain.speaker)
            console.print("[green]correction stored - it now outranks older memories[/green]")
    elif name == "/forget":
        if not arg:
            console.print("[yellow]usage: /forget <memory-id>[/yellow]")
        else:
            brain.memory.delete(arg.strip())
            console.print("[green]forgotten[/green]")
    else:
        console.print(f"[yellow]unknown command: {name}[/yellow]")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="AURA text chat (Phase 1)")
    parser.add_argument("--no-stream", action="store_true", help="wait for full replies")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    bootstrap("chat", level="DEBUG" if args.debug else "INFO")
    audit.record(audit.Event.STARTUP, detail={"interface": "cli_chat"})

    console.print(Panel(_BANNER, border_style="dim"))

    try:
        brain = AuraBrain()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]failed to start: {exc}[/red]")
        return 1

    health = brain.llm.health()
    if not health.get("ok"):
        console.print(
            f"[red]ollama is not serving {health.get('model')} "
            f"at {health.get('host')}.[/red]"
        )
        console.print("[dim]start it with: ollama serve[/dim]")
        return 1

    if brain.persona.is_template or not brain.persona.reviewed:
        console.print(
            "[yellow]Persona not built from writing samples yet - AURA will sound "
            "generic. Run: python -m aura.persona.profile --build-from <folder>[/yellow]"
        )

    console.print(
        f"[dim]{brain.memory.count()} memories loaded from previous sessions[/dim]\n"
    )

    while True:
        try:
            user_text = console.input("[bold cyan]you ›[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_text:
            continue
        if user_text.startswith("/"):
            if not _handle_command(user_text, brain):
                break
            continue

        try:
            if args.no_stream:
                reply = brain.respond(user_text)
                console.print(f"[bold green]aura ›[/bold green] {reply.text}")
                if reply.was_correction:
                    console.print("[dim](stored as a correction)[/dim]")
            else:
                console.print("[bold green]aura ›[/bold green] ", end="")
                for chunk in brain.respond(user_text, stream=True):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                console.print()
        except LLMError as exc:
            console.print(f"[red]{exc}[/red]")
        except Exception as exc:  # noqa: BLE001
            logger.exception("unexpected failure")
            console.print(f"[red]unexpected: {exc}[/red]")

    console.print("[dim]memory saved. bye.[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
