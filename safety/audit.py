"""
AURA — audit trail (design principle 7).

Every action beyond plain conversation is appended here: timestamped, local,
human-readable, append-only. JSON Lines so it stays greppable by eye and
parseable by the control panel.

Everything written passes through the redaction filter first. The audit log must
never become the one place excluded data ends up stored.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from aura import config
from aura.safety import redaction

_LOCK = threading.Lock()


class Event(str, Enum):
    ACTION_PROPOSED = "action_proposed"
    ACTION_CONFIRMED = "action_confirmed"
    ACTION_EXECUTED = "action_executed"
    ACTION_DENIED = "action_denied"
    ACTION_FAILED = "action_failed"
    TIER_CHANGE = "tier_change"
    PERSON_SEEN = "person_seen"
    PERSON_ENROLLED = "person_enrolled"
    DEFLECTION = "deflection"
    REDACTION = "redaction"
    MEMORY_WRITE = "memory_write"
    DEVICE_COMMAND = "device_command"
    REMOTE_ACCESS = "remote_access"
    STARTUP = "startup"
    ERROR = "error"


def _scrub(value: Any) -> Any:
    """Recursively redact strings inside arbitrary detail payloads."""
    if isinstance(value, str):
        return redaction.redact(value).text
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def record(
    event: Event,
    actor: str = "system",
    detail: dict[str, Any] | None = None,
    tier: str | None = None,
    outcome: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one audit entry. Returns the entry that was written."""
    entry: dict[str, Any] = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event": event.value,
        "actor": actor,
        "pid": os.getpid(),
    }
    if tier:
        entry["tier"] = tier
    if outcome:
        entry["outcome"] = outcome
    if detail:
        entry["detail"] = _scrub(detail)

    target = path or config.AUDIT_LOG
    target.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(entry, ensure_ascii=False)
    with _LOCK:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return entry


def read(limit: int = 100, path: Path | None = None) -> list[dict[str, Any]]:
    """Most recent entries, newest last. Used by the Phase 9 control panel."""
    target = path or config.AUDIT_LOG
    if not target.exists():
        return []
    with open(target, encoding="utf-8") as fh:
        lines = fh.readlines()
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
