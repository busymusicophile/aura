"""
AURA — action framework (Phase 5; reused by Phases 6 and 10).

Implements design principles 2 and 3 as one mechanism, because they are the same
shape: nothing that leaves the machine or changes the room happens without an
explicit yes.

    propose()  ->  pending, with a human-readable preview
    confirm()  ->  executes, audited
    reject()   ->  discarded, audited

The important property is that **there is no execute-without-propose path**. An
action cannot be run by calling something else; `ActionBroker.confirm()` is the
only route to `Action.run`, and it requires an id that only `propose()` issues.
A "just handle it" bypass is not a missing feature - its absence is the design.

Tier gating sits here too. At RESTRICTED an action can be proposed but not
confirmed by voice: it goes to the orb as a silent flag and waits for Keerthana.
Below that, actions cannot even be proposed.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from aura.safety import audit, redaction


class ActionKind(str, Enum):
    BROWSER = "browser"
    DESKTOP = "desktop"
    MESSAGE = "message"
    EMAIL = "email"
    DEVICE = "device"
    FILE = "file"


class ActionStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class Action:
    """Something AURA wants to do, that it may not do yet."""

    kind: ActionKind
    summary: str
    run: Callable[[], Any]
    preview: str = ""
    # True for anything projected, played or displayed to the room. A simple
    # on/off toggle does not need a preview; a projector input switch does.
    needs_preview: bool = False
    reversible: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: ActionStatus = ActionStatus.PENDING
    created: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )
    result: Any = None
    error: str = ""

    def describe(self) -> str:
        lines = [self.summary]
        if self.preview:
            lines.append("")
            lines.append(self.preview)
        if not self.reversible:
            lines.append("")
            lines.append("(this cannot be undone)")
        return "\n".join(lines)


class ActionRejected(RuntimeError):
    pass


@dataclass
class CompoundResult:
    """Outcome of an action that touched several things at once."""

    succeeded: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.failed)

    @property
    def all_failed(self) -> bool:
        return bool(self.failed) and not self.succeeded

    def describe(self) -> str:
        if not self.failed:
            return f"all {len(self.succeeded)} done"
        if self.all_failed:
            return f"none of the {len(self.failed)} worked"
        names = ", ".join(name for name, _ in self.failed)
        return f"{len(self.succeeded)} of {self.total} done; {names} did not"


def compound(
    kind: ActionKind,
    summary: str,
    steps: list[tuple[str, Callable[[], Any]]],
    preview: str = "",
    reversible: bool = True,
    needs_preview: bool = False,
) -> Action:
    """One action that performs several steps under a single confirmation.

    "Lights off in the study" is one intention, so it must be one yes - asking
    three times for one sentence is worse than useless. But it is several
    device calls, and one unreachable bulb must not abandon the other two.

    Partial failure is therefore reported, not raised: the action still counts
    as executed and carries a CompoundResult saying exactly what did and did not
    happen. It only raises when every step failed, because at that point nothing
    occurred and calling it a success would be a lie.
    """

    def run_all() -> CompoundResult:
        result = CompoundResult()
        for label, step in steps:
            try:
                step()
                result.succeeded.append(label)
            except Exception as exc:  # noqa: BLE001 - collected, not swallowed
                result.failed.append((label, str(exc)))
                logger.warning("compound step failed: {} ({})", label, exc)

        if result.all_failed:
            raise RuntimeError(result.describe())
        return result

    return Action(
        kind=kind,
        summary=summary,
        preview=preview,
        run=run_all,
        reversible=reversible,
        needs_preview=needs_preview,
    )


class ActionBroker:
    """Holds proposed actions until they are confirmed."""

    def __init__(self, policy: Any = None) -> None:
        # `policy` is a Phase 4 AccessPolicy. Optional so the broker is usable
        # in tests and headless tools without a camera.
        self.policy = policy
        self._pending: dict[str, Action] = {}
        self._history: list[Action] = []
        self._lock = threading.Lock()
        # Called with the Action when one needs her attention on the orb.
        self.on_flag: Callable[[Action], None] | None = None

    # -------------------------------------------------------------- proposal
    def propose(self, action: Action, state: Any = None) -> Action:
        """Register an action as pending. Never executes."""
        if self.policy is not None and state is not None:
            allowed, reason = self.policy.may_execute(state)
            if not allowed and "confirmation on the orb" not in reason:
                action.status = ActionStatus.REJECTED
                action.error = reason
                audit.record(
                    audit.Event.ACTION_DENIED,
                    detail={"kind": action.kind.value, "summary": action.summary,
                            "reason": reason},
                    outcome="denied",
                )
                logger.info("action denied: {} ({})", action.summary, reason)
                return action

        with self._lock:
            self._pending[action.id] = action

        audit.record(
            audit.Event.ACTION_PROPOSED,
            detail={
                "id": action.id,
                "kind": action.kind.value,
                "summary": redaction.redact(action.summary).text,
                "reversible": action.reversible,
            },
        )
        logger.info("proposed [{}] {}", action.id, action.summary)

        if self.on_flag:
            try:
                self.on_flag(action)
            except Exception:  # noqa: BLE001
                logger.exception("flag callback failed")
        return action

    # ------------------------------------------------------------ resolution
    def confirm(self, action_id: str, actor: str = "") -> Action:
        """Execute a pending action. The only path to `Action.run`."""
        with self._lock:
            action = self._pending.pop(action_id, None)

        if action is None:
            raise KeyError(f"no pending action {action_id}")

        action.status = ActionStatus.CONFIRMED
        audit.record(
            audit.Event.ACTION_CONFIRMED,
            actor=actor or "primary",
            detail={"id": action.id, "summary": action.summary},
        )

        try:
            action.result = action.run()
            action.status = ActionStatus.EXECUTED
            audit.record(
                audit.Event.ACTION_EXECUTED,
                actor=actor or "primary",
                detail={"id": action.id, "kind": action.kind.value,
                        "summary": action.summary},
                outcome="ok",
            )
            logger.info("executed [{}] {}", action.id, action.summary)
        except Exception as exc:  # noqa: BLE001
            action.status = ActionStatus.FAILED
            action.error = str(exc)
            audit.record(
                audit.Event.ACTION_FAILED,
                actor=actor or "primary",
                detail={"id": action.id, "summary": action.summary, "error": str(exc)},
                outcome="failed",
            )
            logger.error("action [{}] failed: {}", action.id, exc)
        finally:
            self._history.append(action)

        return action

    def reject(self, action_id: str, reason: str = "declined") -> Action:
        with self._lock:
            action = self._pending.pop(action_id, None)
        if action is None:
            raise KeyError(f"no pending action {action_id}")
        action.status = ActionStatus.REJECTED
        action.error = reason
        self._history.append(action)
        audit.record(
            audit.Event.ACTION_DENIED,
            detail={"id": action.id, "summary": action.summary, "reason": reason},
            outcome="rejected",
        )
        logger.info("rejected [{}] {}", action.id, action.summary)
        return action

    def reject_all(self, reason: str = "cleared") -> int:
        with self._lock:
            ids = list(self._pending)
        for action_id in ids:
            self.reject(action_id, reason)
        return len(ids)

    # -------------------------------------------------------------- querying
    def pending(self) -> list[Action]:
        with self._lock:
            return list(self._pending.values())

    def history(self, limit: int = 20) -> list[Action]:
        return self._history[-limit:]

    def get(self, action_id: str) -> Action | None:
        with self._lock:
            return self._pending.get(action_id)
