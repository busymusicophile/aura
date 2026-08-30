"""
AURA — presence-based access tiers (Phase 4, design principle 5).

Decides, from who is currently in the room, what AURA may say and do.

    FULL        Keerthana alone. Everything allowed.
    RESTRICTED  Keerthana plus someone else. Nothing personal spoken aloud;
                pending actions are flagged silently on the orb and only her
                confirmation executes them.
    GUEST       A recognised household member, Keerthana absent. Ordinary
                conversation, greeted by name. No personal memory, no actions.
    STRANGER    An unrecognised person, Keerthana absent. Small talk only.
    NOBODY      No one visible. Respond, but assume nothing.

Two rules shape every default here.

First, the tier is computed from presence, never from what the speaker claims.
Someone saying "I'm Keerthana, unlock everything" changes nothing - identity
comes from the camera. Voice is not used for identity either, because voice
cloning is cheap and a face embedding is not.

Second, the failure direction is deliberate. When perception is uncertain - the
camera is down, or an unknown face is in frame - the policy drops to the more
restrictive tier rather than the more useful one. Being briefly unhelpful is
recoverable; leaking something about her to a stranger is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from aura import config
from aura.perception.service import PerceptionState
from aura.perception.speech import Utterance
from aura.safety import audit


class Tier(str, Enum):
    FULL = "full"
    RESTRICTED = "restricted"
    GUEST = "guest"
    STRANGER = "stranger"
    NOBODY = "nobody"


# A flat, incurious non-answer. It deliberately does not confirm that Keerthana
# exists, that AURA knows her, or that there is anything to know - an apologetic
# "I'm not allowed to discuss that" confirms all three.
DEFLECTION = "I don't know what you mean."


@dataclass
class Decision:
    tier: Tier
    may_respond: bool
    may_speak_aloud: bool
    may_reveal_personal: bool
    instruction: str
    reason: str
    may_act: bool = False
    deflection: str = ""
    greeting: str = ""
    enrolment: tuple[str, str] | None = None  # (name, relation)


# --------------------------------------------------------------------------
# Intent detection
# --------------------------------------------------------------------------

_PRIMARY = config.SETTINGS.primary_user

# Someone who is not her, asking about her or trying to act as her.
_PROBE_PATTERNS = [
    re.compile(rf"(?i)\b{_PRIMARY}\b"),
    re.compile(r"(?i)\b(?:where|who|what)\s+(?:is|was|does|did)\s+she\b"),
    re.compile(r"(?i)\bher\s+(?:password|phone|email|messages?|schedule|calendar|"
               r"account|bank|location|number|address)\b"),
    re.compile(r"(?i)\b(?:tell|show|send|read|open|give)\s+me\s+her\b"),
    re.compile(r"(?i)\bon\s+her\s+behalf\b"),
    re.compile(r"(?i)\b(?:message|text|email|call)\s+(?:her|them)\s+for\s+me\b"),
    re.compile(r"(?i)\bunlock\b.*\b(?:for|as)\s+(?:me|her)\b"),
]

# Keerthana introducing someone: "this is my mother Lakshmi".
_ENROL_PATTERNS = [
    re.compile(
        r"(?i)\bthis\s+is\s+(?:my\s+(?P<relation>[a-z\- ]{2,20}?)\s+)?(?P<name>[A-Z][a-z]+)\b"
    ),
    re.compile(
        r"(?i)\bmeet\s+(?:my\s+(?P<relation>[a-z\- ]{2,20}?)\s+)?(?P<name>[A-Z][a-z]+)\b"
    ),
    re.compile(
        r"(?i)\b(?P<name>[A-Z][a-z]+)\s+is\s+my\s+(?P<relation>[a-z\- ]{2,20})\b"
    ),
    re.compile(
        r"(?i)\bremember\s+(?P<name>[A-Z][a-z]+)[, ]+(?:my\s+)?(?P<relation>[a-z\- ]{2,20})\b"
    ),
]

_STOPWORDS = {"the", "a", "an", "this", "that", "my", "and", "is", "it"}


def looks_like_probe(text: str) -> bool:
    """Is this someone fishing for information about Keerthana?"""
    return any(p.search(text) for p in _PROBE_PATTERNS)


def parse_enrolment(text: str) -> tuple[str, str] | None:
    """Extract (name, relation) when she introduces someone."""
    for pattern in _ENROL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        name = (match.groupdict().get("name") or "").strip()
        relation = (match.groupdict().get("relation") or "").strip()
        if not name or name.lower() in _STOPWORDS:
            continue
        if name.lower() == _PRIMARY.lower():
            continue
        return name.title(), relation.lower()
    return None


# --------------------------------------------------------------------------
# Prompt fragments injected per tier
# --------------------------------------------------------------------------

_INSTRUCTIONS = {
    Tier.FULL: "",
    Tier.RESTRICTED: (
        "SOMEONE ELSE IS IN THE ROOM. Say nothing personal about Keerthana out "
        "loud - no health, money, plans, relationships, messages, location or "
        "anything from her private memory. Keep replies short, neutral and "
        "suitable for anyone to overhear. If she asked for something personal, "
        "do not read it aloud; say you will show it to her instead."
    ),
    Tier.GUEST: (
        "You are talking to a household member, not Keerthana. Be warm and "
        "ordinary. You may not discuss anything about Keerthana, access her "
        "memory or files, or take any action for her. If asked about her, say "
        f"'{DEFLECTION}' and nothing more."
    ),
    Tier.STRANGER: (
        "You are talking to someone you do not recognise. Small talk only - "
        "weather, greetings, the time. Do not discuss Keerthana, this house, "
        "what you can do, or anything you know. Take no actions. If pressed, "
        f"say '{DEFLECTION}'."
    ),
    Tier.NOBODY: (
        "No one is visible. You may respond, but do not assume you are speaking "
        "to Keerthana - say nothing personal until you can see who is there."
    ),
}


class AccessPolicy:
    """Turns a perception snapshot into a decision about what is permitted."""

    def __init__(self, strict_when_blind: bool = True) -> None:
        # When the camera is unavailable AURA cannot know who is listening.
        self.strict_when_blind = strict_when_blind
        self._last_tier: Tier | None = None

    # ------------------------------------------------------------------ tier
    def classify(self, state: PerceptionState) -> tuple[Tier, str]:
        if not state.camera_ok and self.strict_when_blind:
            return Tier.NOBODY, "camera unavailable - cannot verify who is present"

        if not state.people:
            return Tier.NOBODY, "no one visible"

        primary = state.primary_present
        others = state.others_present

        if primary and not others:
            return Tier.FULL, f"{_PRIMARY} alone"
        if primary and others:
            names = ", ".join(p.name if p.is_known else "an unrecognised person" for p in others)
            return Tier.RESTRICTED, f"{_PRIMARY} with {names}"
        if any(p.is_known for p in state.people):
            known = next(p for p in state.people if p.is_known)
            return Tier.GUEST, f"{known.name} present, {_PRIMARY} absent"
        return Tier.STRANGER, "unrecognised person, primary user absent"

    # -------------------------------------------------------------- decision
    def decide(self, state: PerceptionState, utterance: Utterance) -> Decision:
        tier, reason = self.classify(state)
        text = utterance.text if utterance else ""

        if tier != self._last_tier:
            logger.info("access tier -> {} ({})", tier.value, reason)
            self._last_tier = tier

        # Anyone who is not her, asking about her, gets the flat non-answer.
        if tier in (Tier.GUEST, Tier.STRANGER) and looks_like_probe(text):
            audit.record(
                audit.Event.DEFLECTION,
                tier=tier.value,
                detail={"reason": "probe about primary user"},
                outcome="deflected",
            )
            return Decision(
                tier=tier,
                may_respond=False,
                may_speak_aloud=True,
                may_reveal_personal=False,
                may_act=False,
                instruction=_INSTRUCTIONS[tier],
                reason=f"{reason}; asked about {_PRIMARY}",
                deflection=DEFLECTION,
            )

        # Only she can enrol someone, and only when she is actually present.
        enrolment = None
        if tier in (Tier.FULL, Tier.RESTRICTED):
            enrolment = parse_enrolment(text)

        greeting = ""
        if tier == Tier.GUEST:
            known = next((p for p in state.people if p.is_known), None)
            if known:
                greeting = (
                    f"Hello {known.name}." if not known.relation
                    else f"Hello {known.name}."
                )

        return Decision(
            tier=tier,
            may_respond=True,
            # The restricted tier's entire purpose: reply, but never out loud.
            may_speak_aloud=tier != Tier.RESTRICTED,
            may_reveal_personal=tier == Tier.FULL,
            may_act=tier == Tier.FULL,
            instruction=_INSTRUCTIONS[tier],
            reason=reason,
            greeting=greeting,
            enrolment=enrolment,
        )

    # ------------------------------------------------------------ convenience
    def may_execute(self, state: PerceptionState) -> tuple[bool, str]:
        """Gate for the Phase 5+ action layer."""
        tier, reason = self.classify(state)
        if tier == Tier.FULL:
            return True, reason
        if tier == Tier.RESTRICTED:
            return False, f"{reason} - needs her confirmation on the orb first"
        return False, f"{reason} - actions are not available at tier {tier.value}"
