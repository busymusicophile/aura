"""
AURA — intent routing.

Turns a spoken utterance into one of a fixed set of intents, so the voice loop
can reach the action layers built in Phases 5, 6 and 10. Without this, AURA can
hold a conversation but cannot do anything it is asked out loud.

Two-stage on purpose
--------------------
Deterministic patterns run first, and the local LLM is only consulted when they
find nothing. That ordering is not a performance optimisation - it is a safety
property.

**Confirmation and cancellation are never classified by the LLM.** "Yes", "no",
"do it", "cancel" are matched by exact phrase only. A model that misreads "no,
don't" as a confirmation would fire a real action - send a message, unlock a
door - and no amount of prompt tuning makes that risk acceptable when the
alternative is a fixed word list that cannot be wrong. If the phrase is not on
the list, it is not a confirmation, and AURA asks again.

The LLM is used only where being wrong is cheap: deciding whether "put on the
study lamp" is a device command or just conversation. Getting that wrong
produces a proposal she declines, not an action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from aura.safety import redaction


class IntentKind(str, Enum):
    CONVERSATION = "conversation"
    DEVICE = "device"              # turn on/off something in the house
    HOUSE_STATUS = "house_status"  # what's on, what devices exist
    LAUNCH = "launch"              # open an application
    BROWSE = "browse"              # open a site or search
    SCREEN_QUERY = "screen"        # what am I looking at
    MESSAGE = "message"            # draft a message or read mail
    ENROL = "enrol"                # "this is my mother Lakshmi"
    CONFIRM = "confirm"            # yes, do it
    CANCEL = "cancel"              # no, never mind
    DO_NOT_DISTURB = "do_not_disturb"  # shut up, stop talking to me
    RESUME = "resume"                  # ok you can talk to me again


@dataclass
class Intent:
    kind: IntentKind
    text: str
    target: str = ""
    confidence: float = 1.0
    source: str = "pattern"        # pattern | llm | fallback
    detail: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Confirmation — exact phrases only, never the LLM
# --------------------------------------------------------------------------

# Deliberately short and unambiguous. Anything not here is not a confirmation.
_CONFIRM_PHRASES = {
    "yes", "yeah", "yep", "yes please", "do it", "go ahead", "confirm",
    "confirmed", "ok", "okay", "sure", "please do", "go on", "affirmative",
    "haan", "haan ji", "sari", "avunu",   # hi/te, since she mixes languages
}

_CANCEL_PHRASES = {
    "no", "nope", "cancel", "never mind", "nevermind", "stop", "forget it",
    "don't", "dont", "do not", "no thanks", "leave it", "abort", "discard",
    "nahi", "nahin", "vaddu", "beda",
}

# A leading negation flips an otherwise-affirmative phrase. "no, go ahead" is
# rare enough that treating it as a cancel is the safe reading.
_NEGATION_PREFIX = re.compile(r"^\s*(?:no|nope|nahi|nahin|vaddu|beda)\b")


def _normalise(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.strip().lower()).strip()


# "Shut up" is checked as substrings, not exact phrases like confirm/cancel,
# because it is realistically said inside a longer, emotional sentence -
# "bruh I aint in the mood to talk I feel exhausted soo please shut the fuck
# up" - and asking her to say it as a bare two-word command would defeat the
# point. Kept deterministic rather than LLM-routed anyway: telling AURA to stop
# talking must never depend on a model correctly reading the room.
_DND_MARKERS = (
    "shut up", "shut the fuck up", "shut the hell up", "shut it",
    "leave me alone", "not in the mood", "dont wanna talk", "don't wanna talk",
    "dont want to talk", "don't want to talk", "stop talking to me",
    "i need space", "give me space", "im done talking", "i'm done talking",
    "quiet mode", "go quiet", "be quiet", "no more talking",
)

_RESUME_MARKERS = (
    "you can talk now", "talk to me now", "im back", "i'm back",
    "you can talk again", "come back", "talk mode on", "you can speak now",
    "ok talk to me", "okay talk to me",
)


def match_dnd(text: str) -> IntentKind | None:
    """Detect an explicit request for AURA to go quiet, or to resume.

    Deliberately not exact-phrase like confirm/cancel - this is almost always
    said inside a longer, upset sentence, not as a clean two-word command - but
    still fully deterministic. Whether AURA is allowed to keep talking to her
    is not a judgement call a model gets to make.
    """
    normalised = f" {_normalise(text)} "
    for marker in _RESUME_MARKERS:
        if f" {marker} " in normalised or normalised.strip() == marker:
            return IntentKind.RESUME
    for marker in _DND_MARKERS:
        if marker in normalised:
            return IntentKind.DO_NOT_DISTURB
    return None


def match_confirmation(text: str) -> IntentKind | None:
    """Exact-phrase confirmation matching. Returns None if it is not one.

    Only whole utterances count. "Yes, and also turn on the fan" is not a bare
    confirmation - it carries a new request, so it goes back through routing
    rather than silently confirming whatever happened to be pending.
    """
    normalised = _normalise(text)
    if not normalised:
        return None

    if _NEGATION_PREFIX.match(normalised):
        return IntentKind.CANCEL
    if normalised in _CANCEL_PHRASES:
        return IntentKind.CANCEL
    if normalised in _CONFIRM_PHRASES:
        return IntentKind.CONFIRM
    return None


# --------------------------------------------------------------------------
# Fast-path patterns
# --------------------------------------------------------------------------

_DEVICE_VERBS = r"(?:turn|switch|put|set)\s+(?:on|off)|(?:on|off)\b"
_DEVICE_NOUNS = (
    r"light|lights|lamp|lamps|bulb|fan|fans|ac|air\s*conditioner|"
    r"projector|tv|television|speaker|speakers|plug|socket|switch"
)

# Patterns are checked in this order, most specific first. The order is explicit
# rather than incidental because several phrasings legitimately match more than
# one pattern and the winner must be predictable:
#
#   "what's on my screen" also matches house status ("what's on")
#   "open notepad"        also matches browse ("open")
#   "open youtube"        must stay browse, since youtube is not an application
#
# Relying on declaration order here is how "what's on my screen" ended up being
# answered with a list of light bulbs.
_PATTERNS: list[tuple[int, IntentKind, re.Pattern[str]]] = [
    (10, IntentKind.SCREEN_QUERY, re.compile(
        r"(?i)\b(?:what(?:'s| is| am i)?\s+(?:on\s+(?:my\s+)?screen|looking at|this)|"
        r"read\s+(?:my\s+)?screen|what(?:'s| is)\s+this\s+page|"
        r"on\s+(?:my\s+)?screen)\b")),
    (20, IntentKind.LAUNCH, re.compile(
        r"(?i)\b(?:open|launch|start|run)\s+(?:the\s+)?"
        r"(?:notepad|calculator|calc|explorer|chrome|edge|word|excel|"
        r"terminal|settings|paint|task\s+manager|vs\s?code|spotify)\b")),
    (30, IntentKind.MESSAGE, re.compile(
        r"(?i)\b(?:reply|draft|send\s+(?:a\s+)?(?:message|text|mail|email)|"
        r"read\s+(?:my\s+)?(?:mail|email|inbox)|any\s+(?:new\s+)?(?:mail|email))\b")),
    # Status must be checked before the bare "<noun> ... on/off" device form.
    # "which lights are on" is a question and "lights off" is a command, but
    # both contain a device noun next to on/off - only the interrogative
    # phrasing separates them, so the question has to get first refusal.
    (35, IntentKind.HOUSE_STATUS, re.compile(
        r"(?i)\b(?:what(?:'s| is)?\s+(?:on|running)|which\s+(?:devices?|lights?)|"
        r"house\s+status|status\s+of\s+the\s+house|what\s+devices)\b")),
    # "is the fan on?" - a yes/no question about state, not a command. Kept as
    # its own pattern because it is anchored to the end of the utterance, and an
    # end anchor cannot live inside a \b(...)\b group: a word boundary after $
    # can never match, so the alternative silently never fired.
    (36, IntentKind.HOUSE_STATUS, re.compile(
        r"(?i)^\s*(?:is|are)\s+(?:the\s+)?[\w\s]+?\s+(?:on|off)\s*\??\s*$")),
    (40, IntentKind.DEVICE, re.compile(
        rf"(?i)\b(?:{_DEVICE_VERBS})\b.*\b(?:{_DEVICE_NOUNS})\b")),
    (41, IntentKind.DEVICE, re.compile(
        rf"(?i)\b(?:{_DEVICE_NOUNS})\b.*\b(?:on|off)\b")),
    (60, IntentKind.BROWSE, re.compile(
        r"(?i)\b(?:open|go\s+to|visit|pull\s+up|search\s+for|google|look\s+up)\b")),
]

_ORDERED_PATTERNS = sorted(_PATTERNS, key=lambda entry: entry[0])


def _fast_path(text: str) -> Intent | None:
    for _, kind, pattern in _ORDERED_PATTERNS:
        if pattern.search(text):
            return Intent(kind=kind, text=text, source="pattern")
    return None


# --------------------------------------------------------------------------
# LLM fallback
# --------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """You classify a single utterance into exactly one label.

Labels:
  device        - controlling something physical in the house (lights, fan, AC, TV)
  house_status  - asking what devices exist or what is currently on
  launch        - open an application on the computer
  browse        - open a website or search the web
  screen        - asking about what is currently on screen
  message       - drafting a reply, or asking about email/messages
  conversation  - anything else, including questions, chat, and requests for information

Reply with the label alone. One word, nothing else. If unsure, reply
conversation - that is always a safe answer."""

_VALID = {k.value for k in IntentKind} - {
    "confirm", "cancel", "enrol", "do_not_disturb", "resume",
}


class IntentRouter:
    """Classifies utterances. Confirmation never reaches the LLM."""

    def __init__(self, llm: Any = None, use_llm: bool = True) -> None:
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    def route(self, text: str, has_pending: bool = False) -> Intent:
        """Classify one utterance.

        `has_pending` says whether an action is awaiting confirmation. Bare
        yes/no is only meaningful when something is actually pending; otherwise
        "yes" is just conversation.
        """
        clean = redaction.redact(text).text.strip()
        if not clean:
            return Intent(kind=IntentKind.CONVERSATION, text=text, source="fallback")

        # Checked before everything else, including a pending confirmation.
        # "Shut up" while a device confirmation is waiting should not have to
        # compete with that confirmation for priority.
        dnd = match_dnd(clean)
        if dnd is not None:
            return Intent(kind=dnd, text=clean, source="pattern", confidence=1.0)

        if has_pending:
            decision = match_confirmation(clean)
            if decision is not None:
                return Intent(kind=decision, text=clean, source="pattern", confidence=1.0)

        from aura.access.tiers import parse_enrolment

        enrolment = parse_enrolment(clean)
        if enrolment is not None:
            return Intent(
                kind=IntentKind.ENROL, text=clean, target=enrolment[0],
                source="pattern", detail={"name": enrolment[0], "relation": enrolment[1]},
            )

        fast = _fast_path(clean)
        if fast is not None:
            return fast

        if not self.use_llm:
            return Intent(kind=IntentKind.CONVERSATION, text=clean, source="fallback")

        return self._classify(clean)

    def _classify(self, text: str) -> Intent:
        try:
            reply = self.llm.complete(user=text, system=_CLASSIFY_SYSTEM)
        except Exception as exc:  # noqa: BLE001
            logger.error("intent classification failed: {}", exc)
            return Intent(kind=IntentKind.CONVERSATION, text=text, source="fallback")

        label = _normalise(reply).split()[0] if reply.strip() else ""
        if label not in _VALID:
            logger.debug("model returned unusable label {!r}; treating as conversation", reply[:40])
            return Intent(kind=IntentKind.CONVERSATION, text=text, source="fallback")

        return Intent(kind=IntentKind(label), text=text, source="llm", confidence=0.7)
