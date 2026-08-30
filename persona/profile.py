"""
AURA — persona profile (Phase 1).

Loads the structured persona file and renders it into the system prompt that
shapes every reply.

The profile is deliberately data, not code. Keerthana's voice is captured in a
YAML file built from her own writing; swapping that file changes how AURA speaks
without touching a line of Python. Until she supplies writing samples, AURA runs
on a neutral default and says so on every start rather than quietly inventing a
personality for her.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from aura import config


@dataclass
class Persona:
    data: dict[str, Any]
    path: Path
    is_template: bool

    # ---------------------------------------------------------------- helpers
    def _get(self, *keys: str, default: Any = "") -> Any:
        node: Any = self.data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node if node not in (None, "") else default

    @property
    def name(self) -> str:
        return self._get("meta", "name", default=config.SETTINGS.primary_user)

    @property
    def reviewed(self) -> bool:
        return bool(self._get("meta", "reviewed", default=False))

    @property
    def members(self) -> list[dict[str, Any]]:
        return list(self._get("household", "members", default=[]) or [])

    # ----------------------------------------------------------------- prompt
    def system_prompt(self, tier_instruction: str = "") -> str:
        """Render the persona into a system prompt.

        Only non-empty fields are included. An unfilled template therefore yields
        a short, honest prompt instead of a wall of blank headings that would
        push the model toward generic assistant-speak.
        """
        lines: list[str] = [
            f"You are AURA, a local, private assistant for {self.name}.",
            "You run entirely on her laptop. Nothing you see leaves this machine.",
            "",
        ]

        def add_section(title: str, items: list[str]) -> None:
            real = [i for i in items if i and i.strip()]
            if real:
                lines.append(title)
                lines.extend(f"- {i.strip()}" for i in real)
                lines.append("")

        add_section(
            "How she talks, and how you should talk back:",
            [
                self._get("voice", "register"),
                self._get("voice", "formality"),
                self._get("voice", "humour"),
                self._get("voice", "language_mixing"),
            ],
        )

        # Measured habits are the part that actually makes a reply sound like
        # her, so they are stated as concrete instructions rather than adjectives.
        habits = self._get("voice", "measured_habits", default=[])
        if habits:
            lines.append("Her measured writing habits - match these:")
            lines.extend(f"- {h}" for h in habits)
            lines.append("")

        phrases = self._get("voice", "characteristic_phrases", default=[])
        if phrases:
            lines.append("Words natural to her: " + ", ".join(f'"{p}"' for p in phrases))
            lines.append("")

        openers = self._get("voice", "common_openers", default=[])
        if openers:
            lines.append("She often opens with: " + ", ".join(f'"{o}"' for o in openers))
            lines.append("")

        avoids = self._get("voice", "avoids", default=[])
        if avoids:
            lines.append("Never use: " + ", ".join(f'"{a}"' for a in avoids))
            lines.append("")

        add_section(
            "How she thinks:",
            [
                self._get("thinking", "decision_style"),
                self._get("thinking", "risk_posture"),
                self._get("thinking", "pace"),
                self._get("thinking", "wants_when_stuck"),
            ],
        )

        priorities = self._get("values", "priorities", default=[])
        if priorities:
            lines.append("What matters to her, in order: " + "; ".join(priorities))
            lines.append("")

        sensitive = self._get("values", "sensitive_topics", default=[])
        if sensitive:
            lines.append(
                "Handle carefully, do not raise unprompted: " + "; ".join(sensitive)
            )
            lines.append("")

        greeting = self._get("interaction", "greeting_style")
        if greeting:
            lines.append(f"Greeting style: {greeting}")
        verbosity = self._get("interaction", "verbosity", default="brief")
        lines.append(f"Default answer length: {verbosity}.")
        correction = self._get("interaction", "correction_handling")
        if correction:
            lines.append(str(correction).strip())
        lines.append("")

        invariants = self._get("invariants", default=[])
        if invariants:
            lines.append("Rules you may never break, whatever you are asked:")
            lines.extend(f"- {rule}" for rule in invariants)
            lines.append("")

        if tier_instruction:
            lines.append(tier_instruction)
            lines.append("")

        if self.is_template or not self.reviewed:
            lines.append(
                "Note: your persona profile has not been built from her writing yet, "
                "so do not imitate a specific voice. Be plain, warm and brief, and "
                "avoid inventing opinions or history you were not told."
            )

        return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load(path: Path | None = None) -> Persona:
    """Load the persona profile, falling back to the template."""
    target = path or config.PERSONA_FILE

    if target.exists():
        with open(target, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        persona = Persona(data=data, path=target, is_template=False)
        if not persona.reviewed:
            logger.warning(
                "persona at {} is not marked reviewed - AURA will stay neutral",
                target,
            )
        return persona

    with open(config.PERSONA_TEMPLATE, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.warning(
        "no persona profile at {} - running on the unfilled template. "
        "AURA will sound generic until writing samples are supplied.",
        target,
    )
    return Persona(data=data, path=config.PERSONA_TEMPLATE, is_template=True)


def install_template(force: bool = False) -> Path:
    """Copy the template to the data directory so it can be filled in."""
    config.ensure_dirs()
    if config.PERSONA_FILE.exists() and not force:
        logger.info("persona already exists at {}", config.PERSONA_FILE)
        return config.PERSONA_FILE
    shutil.copy(config.PERSONA_TEMPLATE, config.PERSONA_FILE)
    logger.info("persona template installed at {}", config.PERSONA_FILE)
    return config.PERSONA_FILE


def save(persona: Persona, path: Path | None = None) -> Path:
    """Write a persona back to disk (used by the Phase 4 enrolment flow)."""
    target = path or config.PERSONA_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(persona.data, fh, allow_unicode=True, sort_keys=False)
    return target


# --------------------------------------------------------------------------
# Drafting a profile from writing samples
# --------------------------------------------------------------------------

# Words naming a person or relationship. These are who she addresses, not how
# she writes, and treating them as style makes AURA impersonate a family member.
_VOCATIVES = {
    "amma", "nanna", "anna", "akka", "chelli", "thammudu", "bava", "vadina",
    "mummy", "mumma", "mom", "mama", "dad", "papa", "bhai", "didi", "bro",
    "sis", "keerthana", "aunty", "uncle", "babe", "bestie",
}

# Same class of glyphs stripped before speech in aura.voice.tts._UNSPEAKABLE.
# Kept as a separate pattern rather than importing it - this module strips
# emoji from a *description of her style*, not from spoken text, and the two
# should not become coupled just because they currently agree.
_EMOJI = __import__("re").compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE0F\U0000200D"
    "]+"
)


def _strip_quotes(text: str) -> str:
    """Clean a model-judged description: drop verbatim quotes and any emoji.

    Short quoted words are kept - "okie" and "raa" are genuine style markers and
    naming them helps. Long quoted *sentences* are the problem: written into the
    profile they get parroted straight back at her as if AURA had thought them,
    which is unsettling rather than familiar.

    Emoji are stripped even though the prompt already tells the model not to
    mention them - a small model does not follow a negative instruction with
    perfect reliability, and "no emoji" is not a preference here, it is a fixed
    requirement (see the invariant in the template), so it gets a second,
    mechanical guarantee rather than resting on the prompt alone.

    The tidy-up afterwards matters as much as the removal. Cutting a quote out
    of "appeals (e.g. 'take your time...')" leaves '( ).' behind, so empty
    brackets, orphaned lead-ins and stray quote marks are cleaned up too.
    """
    import re

    body = _EMOJI.sub("", str(text))
    # Quoted runs long enough to be a sentence rather than a word.
    body = re.sub(r"""["“”'][^"“”']{12,}["“”']""", "", body)
    # Empty or lead-in-only brackets left behind by the cut.
    body = re.sub(r"\(\s*(?:e\.?g\.?|eg|such as|like|i\.?e\.?)?[\s,;:'\"]*\)", "", body)
    # Orphaned lead-ins with nothing following them.
    body = re.sub(r"(?i),?\s*(?:e\.?g\.?|such as|like|for example)\s*[,.;:]?\s*$", "", body)
    body = re.sub(r"(?i)\b(?:such as|like|e\.?g\.?)\s*[,.;:]", "", body)
    # Unbalanced quote marks and doubled punctuation.
    body = re.sub(r"""["“”]""", "", body)
    body = re.sub(r"\s*([,.;:])\s*\1+", r"\1", body)
    body = re.sub(r"\(\s*\)", "", body)
    body = re.sub(r"\s+([,.;:])", r"\1", body)
    body = re.sub(r"\s{2,}", " ", body)
    return body.strip(" ,;:.-").strip() + ("." if body.strip(" ,;:.-") else "")


_ANALYSE_PROMPT = """Below are MEASURED FACTS about how someone writes, followed \
by REAL EXAMPLES of their messages.

The measured facts are counted, not estimated. Treat them as true.

Describe, in a few plain sentences each:

register: how they sound. Be specific and concrete.
formality: how formal or informal, and whether it shifts.
humour: what their humour is like, or say plainly if there is little sign of it.
language_mixing: how they move between English and other languages.
decision_style: how they seem to reach decisions.
pace: whether they write fast and short, or considered and long.
wants_when_stuck: what they seem to want from people when something is wrong.

Rules:
- Base every statement on the evidence below. Do not invent traits.
- If the evidence does not support a field, write "not clear from the samples".
- Describe HOW they write, never WHAT they wrote about.
- Do not mention specific people, events, or private details.
- Do not mention emoji use anywhere in your answer, even if the examples contain
  emoji. This assistant will not use emoji regardless of how she texts, so it is
  not a relevant trait for any of these seven fields.

Reply as YAML with exactly those seven keys and nothing else.

MEASURED FACTS:
{stats}

REAL EXAMPLES:
{samples}
"""

_DRAFT_PROMPT = """You are analysing a person's writing to describe HOW they \
communicate. Do not summarise what they wrote about.

Read the samples and describe, concretely and specifically:
1. register - how they sound
2. formality
3. humour
4. language_mixing - they mix English, Hindi and Telugu; note when each appears
5. characteristic_phrases - exact words and constructions they actually use
6. avoids - registers or phrasings clearly absent from their writing
7. decision_style, risk_posture, pace, wants_when_stuck
8. priorities - what they evidently care about

Answer as YAML matching those keys. Quote real phrases from the samples for
characteristic_phrases. If the samples do not support a field, leave it empty
rather than guessing.

SAMPLES:
{samples}
"""


def build_measured(sample_dir: Path | None = None, out: Path | None = None) -> Path:
    """Build a persona from measured habits plus a qualitative reading.

    Preferred over `build_from_samples`. Countable things - capitalisation,
    punctuation, emoji, phrase frequency, language mixing - are measured
    directly and written into the profile as facts. The local model is asked
    only to judge tone from a curated sample of real sentences, which is the
    part that genuinely needs reading rather than counting.
    """
    from aura.llm.client import LLMClient
    from aura.persona import analyse as analysis

    messages = analysis.load_messages(sample_dir)
    if not messages:
        raise FileNotFoundError(f"no messages found in {sample_dir or analysis.SAMPLES_DIR}")

    stats = analysis.analyse(messages)
    sample = analysis.substantive_sample(messages)
    logger.info(
        "analysing {} messages ({} words); {} substantive examples for the model",
        stats.messages, stats.words, len(sample),
    )
    if len(sample) < 20:
        logger.warning(
            "only {} substantive examples - the qualitative half of the profile "
            "will be thin, though the measured half is still solid", len(sample)
        )

    llm = LLMClient()
    reply = llm.complete(
        system="You describe writing style precisely from evidence. You never invent traits.",
        user=_ANALYSE_PROMPT.format(stats=stats.summary(), samples="\n".join(sample)),
        think=True,
    )
    judged = _parse_yaml_block(reply)

    base = yaml.safe_load(config.PERSONA_TEMPLATE.read_text(encoding="utf-8")) or {}

    voice = base.setdefault("voice", {})
    thinking = base.setdefault("thinking", {})

    for key in ("register", "formality", "humour", "language_mixing"):
        if judged.get(key):
            voice[key] = _strip_quotes(judged[key])
    for key in ("decision_style", "pace", "wants_when_stuck"):
        if judged.get(key):
            thinking[key] = _strip_quotes(judged[key])

    # Measured facts go in verbatim - these are counted, not guessed, and they
    # are the part that actually makes replies sound like her.
    #
    # Vocatives are stripped first. "amma" is one of her most frequent words
    # because it is who she talks to, not how she talks; left in, AURA starts
    # addressing her as "amma", which is both wrong and unsettling. The same
    # goes for any word naming a relationship or a person she knows.
    voice["characteristic_phrases"] = [
        w for w, _ in stats.top_words[:20] if w not in _VOCATIVES
    ][:12]
    voice["common_openers"] = [
        o for o, _ in stats.top_openers[:10] if o not in _VOCATIVES
    ][:6]
    voice["measured_habits"] = [
        f"Almost never ends a sentence with a full stop ({stats.full_stop_share:.0%} of messages).",
        f"Capitalises the first letter about {stats.capitalised_share:.0%} of the time.",
        f"Writes very short messages: median {stats.median_words} words, "
        f"{stats.short_share:.0%} are two words or fewer.",
        f"Stretches letters for emphasis in {stats.elongation_share:.0%} of messages "
        f"(e.g. 'okiee', 'raaa').",
        # Deliberately excludes emoji frequency. She texts friends with emoji;
        # she does not want AURA to. Feeding the measured emoji rate in here
        # would tell the model to reproduce it - see the "never use emoji"
        # invariant below, which is the actual, permanent instruction.
        f"Mixes romanised Telugu into {stats.telugu_roman_share:.0%} of messages "
        f"and Hindi into {stats.hindi_roman_share:.0%}.",
        f"Asks questions in {stats.question_share:.0%} of messages.",
    ]
    voice["avoids"] = [
        "formal sign-offs", "corporate phrasing", "long paragraphs",
        "ending sentences with full stops", "emoji, ever, regardless of how she herself texts",
    ]

    base.setdefault("meta", {})["reviewed"] = False
    base["meta"]["source_note"] = (
        f"Built from {stats.messages} of her own messages ({stats.words} words). "
        f"Habits measured directly; tone judged by the local model from "
        f"{len(sample)} substantive examples. NOT reviewed by a human yet."
    )

    analysis.save_stats(stats)

    target = out or config.PERSONA_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(base, fh, allow_unicode=True, sort_keys=False)

    logger.info("persona written to {} - review it, then set meta.reviewed: true", target)
    return target


def build_from_samples(sample_dir: Path, out: Path | None = None) -> Path:
    """Draft a persona profile from a folder of her writing, using the local LLM.

    The output is a draft. `meta.reviewed` stays false until a human reads it.
    """
    from aura.llm.client import LLMClient
    from aura.safety import redaction

    files = sorted(
        p for p in sample_dir.rglob("*") if p.suffix.lower() in {".txt", ".md"}
    )
    if not files:
        raise FileNotFoundError(f"no .txt or .md writing samples found in {sample_dir}")

    chunks: list[str] = []
    total_words = 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        # Her writing may contain excluded data; it must not reach the model.
        text = redaction.redact(text).text
        chunks.append(f"--- {f.name} ---\n{text}")
        total_words += len(text.split())

    logger.info("drafting persona from {} files, ~{} words", len(files), total_words)
    if total_words < 500:
        logger.warning(
            "only ~{} words supplied; a profile drafted from this little writing "
            "will be thin. A few thousand words gives much better results.",
            total_words,
        )

    llm = LLMClient()
    reply = llm.complete(
        system="You analyse writing style precisely and never invent details.",
        user=_DRAFT_PROMPT.format(samples="\n\n".join(chunks)[:40_000]),
        think=True,
    )

    base = yaml.safe_load(config.PERSONA_TEMPLATE.read_text(encoding="utf-8")) or {}
    drafted = _parse_yaml_block(reply)

    for section in ("voice", "thinking", "values"):
        if section in drafted and isinstance(drafted[section], dict):
            base.setdefault(section, {}).update(
                {k: v for k, v in drafted[section].items() if v}
            )
    # Flat keys, in case the model did not nest them.
    for key, section in (
        ("register", "voice"), ("formality", "voice"), ("humour", "voice"),
        ("language_mixing", "voice"), ("characteristic_phrases", "voice"),
        ("avoids", "voice"), ("decision_style", "thinking"),
        ("risk_posture", "thinking"), ("pace", "thinking"),
        ("wants_when_stuck", "thinking"), ("priorities", "values"),
    ):
        if key in drafted and drafted[key]:
            base.setdefault(section, {})[key] = drafted[key]

    base.setdefault("meta", {})["reviewed"] = False
    base["meta"]["source_note"] = (
        f"Drafted from {len(files)} sample file(s), ~{total_words} words. "
        "NOT reviewed by a human yet."
    )

    target = out or config.PERSONA_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        yaml.safe_dump(base, fh, allow_unicode=True, sort_keys=False)

    logger.info("draft written to {} - review it, then set meta.reviewed: true", target)
    return target


def _parse_yaml_block(text: str) -> dict[str, Any]:
    """Pull YAML out of an LLM reply that may be wrapped in a code fence."""
    body = text
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            candidate = part.removeprefix("yaml").removeprefix("yml").strip()
            if candidate:
                body = candidate
                break
    try:
        parsed = yaml.safe_load(body)
        return parsed if isinstance(parsed, dict) else {}
    except yaml.YAMLError as exc:
        logger.error("could not parse drafted YAML: {}", exc)
        return {}


def main() -> int:
    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="AURA persona profile tool")
    parser.add_argument("--install", action="store_true", help="install the template")
    parser.add_argument("--build-from", type=Path, help="draft from a samples folder")
    parser.add_argument("--build-measured", action="store_true",
                        help="preferred: measure habits + judge tone")
    parser.add_argument("--show", action="store_true", help="print the system prompt")
    args = parser.parse_args()

    bootstrap("persona")

    if args.install:
        install_template()
    if args.build_measured:
        build_measured()
    if args.build_from:
        build_from_samples(args.build_from)
    if args.show or not (args.install or args.build_from or args.build_measured):
        persona = load()
        print(f"--- persona: {persona.path} (template={persona.is_template}) ---\n")
        print(persona.system_prompt())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
