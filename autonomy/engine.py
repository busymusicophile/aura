"""
AURA — proactive suggestions (Phase 8).

Watches context and occasionally offers something unprompted, drawing on persona
and memory. Full access tier only, and off by default.

On emotional inference
----------------------
The brief asked to scope the false-positive risk before reacting to inferred
emotional state. Having scoped it, this engine deliberately **does not infer
emotional state at all**, and that is a design decision rather than an omission.

Facial-expression classifiers report roughly 60-75% agreement with human labels
on posed datasets and materially worse on spontaneous, non-Western faces. Voice
"stress" detection is worse still and confounded by accent, room acoustics and
head cold. At those rates, a system checking in every ten minutes generates
several wrong reads a day.

And the errors are not symmetric. "You seem stressed, want to talk about it?"
when she is simply concentrating is not a small miss - it is presumptuous,
slightly unsettling, and it teaches her that AURA's guesses about her inner state
are unreliable. That impression is expensive to undo and it poisons trust in the
suggestions that *are* well-founded.

So every trigger here fires on facts AURA actually knows: elapsed time, what is
on screen, what was said, what is in memory. "You have been at this for two
hours" is checkable and she can disagree with it. "You seem upset" is not.

If emotional awareness is wanted later, the honest version is her telling AURA
how she feels and AURA remembering it - not a camera guessing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from aura import config
from aura.safety import audit


@dataclass
class Context:
    """Everything a trigger is allowed to look at."""

    now: datetime
    session_seconds: float
    active_window: str
    active_process: str
    idle_seconds: float
    presence_names: list[str]
    tier: str
    last_utterance: str = ""
    memory: Any = None


@dataclass
class Suggestion:
    trigger: str
    message: str
    urgency: str = "low"  # low | normal
    created: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )


@dataclass
class Trigger:
    """One proactive rule."""

    name: str
    description: str
    condition: Callable[[Context], bool]
    prompt: str
    cooldown_seconds: float = 3600.0
    max_per_day: int = 3
    enabled: bool = True

    _last_fired: float = field(default=0.0, repr=False)
    _fired_today: int = field(default=0, repr=False)
    _day: str = field(default="", repr=False)

    def ready(self, now: float, today: str) -> bool:
        if not self.enabled:
            return False
        if self._day != today:
            self._day = today
            self._fired_today = 0
        if self._fired_today >= self.max_per_day:
            return False
        return (now - self._last_fired) >= self.cooldown_seconds

    def mark_fired(self, now: float, today: str) -> None:
        self._last_fired = now
        self._day = today
        self._fired_today += 1


# --------------------------------------------------------------------------
# Built-in triggers — all fact-based
# --------------------------------------------------------------------------


def _long_session(ctx: Context) -> bool:
    return ctx.session_seconds > 2 * 3600 and ctx.idle_seconds < 300


def _very_late(ctx: Context) -> bool:
    return ctx.now.hour in (1, 2, 3, 4) and ctx.idle_seconds < 600


def _back_after_absence(ctx: Context) -> bool:
    return ctx.idle_seconds > 4 * 3600


def _same_window_a_long_time(ctx: Context) -> bool:
    return ctx.session_seconds > 90 * 60 and bool(ctx.active_window)


DEFAULT_TRIGGERS: list[Trigger] = [
    Trigger(
        name="long_session",
        description="Working continuously for over two hours",
        condition=_long_session,
        prompt=(
            "She has been working continuously for over two hours. Offer a short, "
            "un-nagging observation - one sentence, easy to ignore. Do not "
            "moralise about health and do not tell her what to do."
        ),
        cooldown_seconds=2 * 3600,
        max_per_day=2,
    ),
    Trigger(
        name="very_late",
        description="Still active between 1am and 5am",
        condition=_very_late,
        prompt=(
            "It is the small hours and she is still up. Say one brief, warm thing. "
            "Do not lecture her about sleep - she knows what time it is."
        ),
        cooldown_seconds=6 * 3600,
        max_per_day=1,
    ),
    Trigger(
        name="back_after_absence",
        description="Returning after several hours away",
        condition=_back_after_absence,
        prompt=(
            "She is back after several hours away. Greet her briefly in her own "
            "register. If memory holds something genuinely worth picking back up, "
            "mention it in a few words. Otherwise just say hello and stop."
        ),
        cooldown_seconds=3 * 3600,
        max_per_day=3,
    ),
    Trigger(
        name="deep_focus",
        description="Ninety minutes in a single application",
        condition=_same_window_a_long_time,
        prompt=(
            "She has been in the same application for a long stretch. Offer one "
            "short, concrete, useful remark about what she is doing, drawing on "
            "memory if it is relevant. If you have nothing genuinely useful to "
            "add, say nothing at all."
        ),
        cooldown_seconds=3 * 3600,
        max_per_day=2,
        enabled=False,  # opt-in: the most likely to feel intrusive
    ),
]

# The model is told it may decline. A suggestion nobody needed is worse than
# silence, so refusing must be a first-class option rather than a failure.
_DECLINE = "NOTHING"

_SYSTEM = (
    "You are deciding whether to say something unprompted to Keerthana. "
    "Unprompted remarks are intrusive by default, so the bar is high.\n\n"
    f"If you have nothing genuinely worth saying, reply with exactly {_DECLINE}. "
    "That is the correct answer most of the time and choosing it is not a "
    "failure.\n\n"
    "If you do speak: one or two sentences, her register, no preamble, no "
    "question she has to answer, nothing about her emotional state - you cannot "
    "see that and guessing is worse than silence."
)


class AutonomyEngine:
    """Evaluates triggers and produces suggestions. Off unless switched on."""

    def __init__(
        self,
        brain: Any = None,
        triggers: list[Trigger] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.brain = brain
        self.triggers = triggers if triggers is not None else list(DEFAULT_TRIGGERS)
        self.enabled = config.SETTINGS.autonomy_enabled if enabled is None else enabled
        self.on_suggestion: Callable[[Suggestion], None] | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._history: list[Suggestion] = []
        self._session_start = time.time()

    # ---------------------------------------------------------------- toggle
    def enable(self) -> None:
        self.enabled = True
        logger.info("autonomy enabled")
        audit.record(audit.Event.STARTUP, detail={"autonomy": "enabled"})

    def disable(self) -> None:
        self.enabled = False
        logger.info("autonomy disabled")
        audit.record(audit.Event.STARTUP, detail={"autonomy": "disabled"})

    # ------------------------------------------------------------- evaluation
    def evaluate(self, ctx: Context) -> Suggestion | None:
        """Check triggers and, if one fires, ask the model what to say."""
        if not self.enabled:
            return None

        # Full tier only: never volunteer anything with someone else in the room.
        if ctx.tier != "full":
            return None

        now = time.time()
        today = ctx.now.date().isoformat()

        for trigger in self.triggers:
            if not trigger.ready(now, today):
                continue
            try:
                if not trigger.condition(ctx):
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.error("trigger {} raised: {}", trigger.name, exc)
                continue

            message = self._compose(trigger, ctx)
            trigger.mark_fired(now, today)

            if message is None:
                logger.debug("trigger {} fired but the model declined", trigger.name)
                return None

            suggestion = Suggestion(trigger=trigger.name, message=message)
            self._history.append(suggestion)
            audit.record(
                audit.Event.ACTION_PROPOSED,
                detail={"kind": "suggestion", "trigger": trigger.name},
                tier=ctx.tier,
            )
            logger.info("suggestion [{}]: {}", trigger.name, message)

            if self.on_suggestion:
                try:
                    self.on_suggestion(suggestion)
                except Exception:  # noqa: BLE001
                    logger.exception("suggestion callback failed")
            return suggestion

        return None

    def _compose(self, trigger: Trigger, ctx: Context) -> str | None:
        if self.brain is None:
            return trigger.description

        recall = ""
        if ctx.memory is not None:
            try:
                recall = ctx.memory.recall_block(ctx.active_window or trigger.name, k=3)
            except Exception:  # noqa: BLE001
                recall = ""

        prompt = (
            f"{trigger.prompt}\n\n"
            f"Time: {ctx.now.strftime('%A %H:%M')}\n"
            f"Session length: {ctx.session_seconds / 3600:.1f} hours\n"
            f"On screen: {ctx.active_window or 'unknown'}\n"
        )
        if recall:
            prompt += f"\n{recall}\n"

        try:
            reply = self.brain.llm.complete(
                user=prompt,
                system=f"{self.brain.persona.system_prompt()}\n\n{_SYSTEM}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("could not compose suggestion: {}", exc)
            return None

        text = reply.strip()
        if not text or _DECLINE in text.upper()[:20]:
            return None
        return text

    # ------------------------------------------------------------ background
    def start(self, context_fn: Callable[[], Context], interval: float = 300.0) -> None:
        """Poll for trigger conditions on a background thread."""
        if self._thread and self._thread.is_alive():
            return

        def loop() -> None:
            while not self._stop.wait(interval):
                if not self.enabled:
                    continue
                try:
                    self.evaluate(context_fn())
                except Exception:  # noqa: BLE001
                    logger.exception("autonomy evaluation failed")

        self._stop.clear()
        self._thread = threading.Thread(target=loop, name="aura-autonomy", daemon=True)
        self._thread.start()
        logger.info("autonomy loop started (every {:.0f}s, enabled={})", interval, self.enabled)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def history(self, limit: int = 20) -> list[Suggestion]:
        return self._history[-limit:]

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "triggers": {
                t.name: {
                    "enabled": t.enabled,
                    "description": t.description,
                    "fired_today": t._fired_today,
                    "max_per_day": t.max_per_day,
                }
                for t in self.triggers
            },
            "suggestions_this_session": len(self._history),
        }


def build_context(
    perception_state: Any = None,
    tier: str = "full",
    session_start: float | None = None,
    memory: Any = None,
) -> Context:
    """Assemble a Context from live perception, or sensible defaults."""
    from aura.perception import screen

    title, process = screen.active_window()
    started = session_start or time.time()

    return Context(
        now=datetime.now().astimezone(),
        session_seconds=time.time() - started,
        active_window=title,
        active_process=process,
        idle_seconds=_idle_seconds(),
        presence_names=list(getattr(perception_state, "names", []) or []),
        tier=tier,
        memory=memory,
    )


def _idle_seconds() -> float:
    """Seconds since the last keyboard or mouse input."""
    try:
        import ctypes

        class LastInput(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = LastInput()
        info.cbSize = ctypes.sizeof(LastInput)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
            return millis / 1000.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA autonomy engine (Phase 8)")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate triggers without calling the model")
    parser.add_argument("--simulate", metavar="TRIGGER", help="force one trigger")
    args = parser.parse_args()

    bootstrap("autonomy")
    engine = AutonomyEngine()

    if args.status or not (args.dry_run or args.simulate):
        import json

        print(json.dumps(engine.status(), indent=2))
        print(f"\nidle for {_idle_seconds():.0f}s")
        return 0

    if args.dry_run:
        engine.enable()
        ctx = build_context()
        print(f"session {ctx.session_seconds / 60:.1f}min, idle {ctx.idle_seconds:.0f}s, "
              f"window '{ctx.active_window[:50]}'")
        for trigger in engine.triggers:
            try:
                fires = trigger.condition(ctx)
            except Exception as exc:  # noqa: BLE001
                fires = f"error: {exc}"
            print(f"  {trigger.name:<22} enabled={trigger.enabled!s:<5} fires={fires}")
        return 0

    if args.simulate:
        from aura.chat import AuraBrain

        engine = AutonomyEngine(brain=AuraBrain(), enabled=True)
        trigger = next((t for t in engine.triggers if t.name == args.simulate), None)
        if trigger is None:
            print(f"no trigger called {args.simulate}")
            return 1
        ctx = build_context()
        ctx.session_seconds = 3 * 3600
        message = engine._compose(trigger, ctx)
        print(f"\n[{trigger.name}] -> {message or '(declined - said nothing)'}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
