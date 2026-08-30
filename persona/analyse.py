"""
AURA — persona analysis from real writing samples.

Two stages, because they answer different kinds of question.

**Measured, not guessed.** How often she capitalises, whether she uses full
stops, which emoji she actually reaches for, how long her messages run, which
phrases recur, how she mixes English with romanised Telugu and Hindi - all of
that is countable. Asking a language model to estimate frequencies it cannot
count is how you get a confident, wrong profile.

**Judged, not counted.** Tone, humour, how she handles being annoyed, what she
does when someone needs something - those need reading, and that is what the
local model is for.

The second stage only ever sees a curated sample of her *substantive* messages.
Chat exports are dominated by one-word replies - here, 55% of lines are two words
or fewer - and a model fed mostly "ok" and "lol" will hallucinate a personality
to fill the gap. Length is a crude proxy for substance, but it is an honest one.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from aura import config
from aura.safety import redaction

SAMPLES_DIR = config.PERSONA_DIR / "samples"

_EMOJI = re.compile(
    "[\U0001f000-\U0001faff☀-➿⬀-⯿️❤]+"
)
_TELUGU = re.compile("[ఀ-౿]")
_DEVANAGARI = re.compile("[ऀ-ॿ]")
_ELONGATED = re.compile(r"\b\w*(\w)\1{2,}\w*\b")
_LAUGH = re.compile(r"(?i)\b(lol|lmao|lmfao|rofl|ha(?:ha)+|hehe+|xd)\b")

# Romanised Telugu and Hindi markers. Chosen to be words that rarely appear in
# English text, so their presence is a real signal rather than a coincidence.
_TELUGU_ROMAN = {
    "ra", "da", "kada", "ante", "em", "enti", "ledu", "avunu", "sari", "ela",
    "cheppu", "chey", "vaddu", "kavali", "unnav", "unna", "chala", "bagundi",
    "emaindi", "ipudu", "appudu", "nenu", "nuvvu", "meeru", "vachi", "vellu",
    "anna", "akka", "amma", "nanna", "baga", "koncham", "thopu", "asalu",
}
_HINDI_ROMAN = {
    "haan", "nahi", "nahin", "kya", "hai", "hain", "tha", "thi", "acha",
    "accha", "theek", "bas", "matlab", "yaar", "arey", "abey", "kuch",
    "kaise", "kaisa", "bohot", "bahut", "mera", "tera", "apna", "chal",
    "chalo", "karo", "karna", "pata", "sach", "bhai", "didi",
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "is", "are", "was", "were", "be", "been", "am", "i", "you",
    "it", "that", "this", "so", "my", "me", "we", "they", "he", "she", "as",
    "no", "not", "do", "did", "does", "have", "has", "had", "will", "would",
    "can", "could", "just", "what", "when", "how", "why", "there", "here",
}


@dataclass
class Stats:
    messages: int = 0
    words: int = 0
    mean_words: float = 0.0
    median_words: int = 0
    p90_words: int = 0
    longest: int = 0
    short_share: float = 0.0
    capitalised_share: float = 0.0
    full_stop_share: float = 0.0
    question_share: float = 0.0
    exclamation_share: float = 0.0
    emoji_share: float = 0.0
    top_emoji: list[tuple[str, int]] = field(default_factory=list)
    elongation_share: float = 0.0
    laughter_share: float = 0.0
    telugu_script_share: float = 0.0
    devanagari_share: float = 0.0
    telugu_roman_share: float = 0.0
    hindi_roman_share: float = 0.0
    top_words: list[tuple[str, int]] = field(default_factory=list)
    top_bigrams: list[tuple[str, int]] = field(default_factory=list)
    top_openers: list[tuple[str, int]] = field(default_factory=list)

    def summary(self) -> str:
        """A human- and model-readable digest of the measured habits."""
        lines = [
            f"Sample: {self.messages} messages, {self.words} words.",
            f"Length: mean {self.mean_words:.1f} words, median {self.median_words}, "
            f"90th percentile {self.p90_words}, longest {self.longest}.",
            f"{self.short_share:.0%} of messages are two words or fewer.",
            f"Capitalises the first letter {self.capitalised_share:.0%} of the time.",
            f"Ends with a full stop {self.full_stop_share:.0%} of the time; "
            f"question marks {self.question_share:.0%}; exclamation {self.exclamation_share:.0%}.",
            f"Uses emoji in {self.emoji_share:.0%} of messages.",
            f"Stretches letters (e.g. 'soooo') in {self.elongation_share:.0%}.",
            f"Laughs (lol/haha/etc) in {self.laughter_share:.0%}.",
        ]
        if self.top_emoji:
            lines.append("Most-used emoji: " + " ".join(e for e, _ in self.top_emoji[:8]))
        lang = []
        if self.telugu_roman_share > 0.005:
            lang.append(f"romanised Telugu in {self.telugu_roman_share:.0%}")
        if self.hindi_roman_share > 0.005:
            lang.append(f"romanised Hindi in {self.hindi_roman_share:.0%}")
        if self.telugu_script_share > 0.002:
            lang.append(f"Telugu script in {self.telugu_script_share:.1%}")
        if self.devanagari_share > 0.002:
            lang.append(f"Devanagari in {self.devanagari_share:.1%}")
        lines.append("Language mixing: " + (", ".join(lang) if lang else "almost entirely English"))
        if self.top_words:
            lines.append("Characteristic words: " + ", ".join(w for w, _ in self.top_words[:15]))
        if self.top_bigrams:
            lines.append("Recurring pairs: " + ", ".join(b for b, _ in self.top_bigrams[:10]))
        if self.top_openers:
            lines.append("Common openers: " + ", ".join(o for o, _ in self.top_openers[:8]))
        return "\n".join(lines)


def load_messages(directory: Path | None = None) -> list[str]:
    target = directory or SAMPLES_DIR
    if not target.exists():
        raise FileNotFoundError(f"no samples at {target}")

    messages: list[str] = []
    for path in sorted(target.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        messages.extend(line.strip() for line in text.splitlines() if line.strip())

    # Redact before anything else touches it.
    cleaned: list[str] = []
    redacted = 0
    for message in messages:
        result = redaction.redact(message)
        if not result.is_clean:
            redacted += 1
        cleaned.append(result.text)
    if redacted:
        logger.warning("redacted {} message(s) before analysis", redacted)
    return cleaned


def analyse(messages: list[str]) -> Stats:
    if not messages:
        return Stats()

    lengths = sorted(len(m.split()) for m in messages)
    total = len(messages)

    def share(predicate) -> float:  # noqa: ANN001
        return sum(1 for m in messages if predicate(m)) / total

    emoji_counter: Counter[str] = Counter()
    word_counter: Counter[str] = Counter()
    bigram_counter: Counter[str] = Counter()
    opener_counter: Counter[str] = Counter()
    telugu_roman = hindi_roman = 0

    for message in messages:
        for cluster in _EMOJI.findall(message):
            for ch in cluster:
                if ch.strip() and ch != "️":
                    emoji_counter[ch] += 1

        words = re.findall(r"[a-z']+", message.lower())
        if words:
            opener_counter[words[0]] += 1
        for word in words:
            if word not in _STOPWORDS and len(word) > 2:
                word_counter[word] += 1
        for a, b in zip(words, words[1:], strict=False):
            if a not in _STOPWORDS or b not in _STOPWORDS:
                bigram_counter[f"{a} {b}"] += 1

        lowered = set(words)
        if lowered & _TELUGU_ROMAN:
            telugu_roman += 1
        if lowered & _HINDI_ROMAN:
            hindi_roman += 1

    return Stats(
        messages=total,
        words=sum(lengths),
        mean_words=sum(lengths) / total,
        median_words=lengths[total // 2],
        p90_words=lengths[int(total * 0.9)],
        longest=lengths[-1],
        short_share=sum(1 for n in lengths if n <= 2) / total,
        capitalised_share=share(lambda m: m[:1].isupper()),
        full_stop_share=share(lambda m: m.rstrip().endswith(".")),
        question_share=share(lambda m: "?" in m),
        exclamation_share=share(lambda m: "!" in m),
        emoji_share=share(lambda m: bool(_EMOJI.search(m))),
        elongation_share=share(lambda m: bool(_ELONGATED.search(m))),
        laughter_share=share(lambda m: bool(_LAUGH.search(m))),
        telugu_script_share=share(lambda m: bool(_TELUGU.search(m))),
        devanagari_share=share(lambda m: bool(_DEVANAGARI.search(m))),
        telugu_roman_share=telugu_roman / total,
        hindi_roman_share=hindi_roman / total,
        top_emoji=emoji_counter.most_common(12),
        top_words=word_counter.most_common(25),
        top_bigrams=bigram_counter.most_common(15),
        top_openers=opener_counter.most_common(10),
    )


def substantive_sample(
    messages: list[str],
    budget_chars: int = 6000,
    min_words: int = 5,
    max_words: int = 60,
) -> list[str]:
    """A spread of her real sentences, for the qualitative pass.

    Chat exports skew hard to one-word replies - 57% of these are two words or
    fewer - so a model fed raw would see mostly "okie" and invent the rest.

    Two limits matter. `min_words` skips the acknowledgements. `max_words`
    excludes outliers: one 694-word message would otherwise consume the whole
    budget and leave the model reading a single atypical monologue as if it
    were her voice. Between those bounds, messages are taken evenly across the
    corpus rather than all from one conversation.
    """
    eligible = [m for m in messages if min_words <= len(m.split()) <= max_words]
    if not eligible:
        return []

    # Walk the corpus at an even stride so all chats and periods are represented.
    average = sum(len(m) for m in eligible) / len(eligible)
    wanted = max(1, int(budget_chars / max(average, 1)))
    stride = max(1, len(eligible) // wanted)

    chosen: list[str] = []
    used = 0
    for message in eligible[::stride]:
        if used + len(message) > budget_chars:
            break
        chosen.append(message)
        used += len(message)
    return chosen


def save_stats(stats: Stats, path: Path | None = None) -> Path:
    target = path or (config.PERSONA_DIR / "writing_stats.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        k: v for k, v in stats.__dict__.items() if not k.startswith("_")
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def main() -> int:
    import argparse

    from aura.runtime import bootstrap

    parser = argparse.ArgumentParser(description="Analyse writing samples")
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--sample", action="store_true", help="show the curated sample size")
    args = parser.parse_args()

    bootstrap("analyse")
    messages = load_messages(args.dir)
    stats = analyse(messages)

    print("=" * 72)
    print("  MEASURED WRITING HABITS")
    print("=" * 72)
    print(stats.summary())
    print("=" * 72)

    if args.sample:
        chosen = substantive_sample(messages)
        print(f"\ncurated sample: {len(chosen)} substantive messages, "
              f"{sum(len(c) for c in chosen)} chars")

    path = save_stats(stats)
    print(f"\nsaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
