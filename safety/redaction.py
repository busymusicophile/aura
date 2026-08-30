"""
AURA — hard-excluded data filter (design principle 4).

AURA must never read, store, transmit, log, or display: bank account or card
numbers, UPI PINs, Aadhaar numbers, PAN numbers, passwords, OTPs, or other
government ID numbers. This filter is the single chokepoint that enforces it and
runs on text before it reaches the LLM context, the memory store, the audit log,
or spoken output. It is global - it applies at every access tier, including when
Keerthana is alone, and there is no bypass flag.

Design notes
------------
A filter that fires constantly gets switched off, so precision matters as much as
recall. Two techniques keep false positives down:

* Checksums. Aadhaar carries a Verhoeff check digit and payment cards carry a
  Luhn check digit. A random 12- or 16-digit number almost never validates, so
  order numbers, timestamps and IDs pass through untouched.
* Context anchors. A bare 4-digit number is meaningless; "OTP is 4821" is not.
  Short secrets are only matched when an anchoring keyword sits next to them.

Findings never retain the matched value. Only the category, the position, and the
length survive - otherwise the audit log would itself become a place where
excluded data is stored, which is the exact thing this module exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    AADHAAR = "AADHAAR"
    PAN = "PAN"
    CARD = "CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    UPI_PIN = "UPI_PIN"
    OTP = "OTP"
    PASSWORD = "PASSWORD"
    PASSPORT = "PASSPORT"
    VOTER_ID = "VOTER_ID"
    DRIVING_LICENCE = "DRIVING_LICENCE"


@dataclass(frozen=True)
class Finding:
    """A detected secret. Deliberately does NOT retain the matched text."""

    category: Category
    start: int
    end: int
    length: int

    def __repr__(self) -> str:  # keeps accidental logging harmless
        return f"Finding({self.category.value}, span={self.start}:{self.end}, len={self.length})"


@dataclass
class RedactionResult:
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.findings

    @property
    def categories(self) -> set[Category]:
        return {f.category for f in self.findings}

    def summary(self) -> str:
        if self.is_clean:
            return "clean"
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.category.value] = counts.get(f.category.value, 0) + 1
        return ", ".join(f"{k}x{v}" for k, v in sorted(counts.items()))


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def verhoeff_valid(digits: str) -> bool:
    """Aadhaar's check-digit scheme."""
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def luhn_valid(digits: str) -> bool:
    """Payment-card check-digit scheme."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

# Digit runs that may carry separators, e.g. "1234 5678 9012" or "1234-5678".
_SEPARATED_DIGITS = re.compile(r"\b\d[\d\s\-]{9,24}\d\b")

_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_PASSPORT = re.compile(r"\b[A-PR-WY][0-9]{7}\b")
_VOTER_ID = re.compile(r"\b[A-Z]{3}[0-9]{7}\b")
_DL = re.compile(r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?(?:19|20)\d{2}[-\s]?\d{7}\b")

_ANCHORED = [
    (
        Category.OTP,
        re.compile(
            r"(?i)\b(?:otp|one[\s-]?time[\s-]?(?:password|code|pin)|verification\s+code|"
            r"security\s+code|auth(?:entication)?\s+code)\b[^\n\d]{0,24}(\d{4,8})\b"
        ),
    ),
    (
        Category.UPI_PIN,
        re.compile(
            r"(?i)\b(?:upi[\s-]?pin|m[\s-]?pin|atm[\s-]?pin|card[\s-]?pin|"
            r"(?<!zip[\s])\bpin(?:\s+(?:is|=|:))?)\b[^\n\d]{0,16}(\d{4,6})\b"
        ),
    ),
    (
        Category.PASSWORD,
        re.compile(
            # Separator class includes underscore: real code and config write
            # these as api_key / secret_key / access_token far more often than
            # with a space, and an underscore-only miss is a silent leak.
            r"(?i)\b(?:password|passwd|pwd|passphrase|api[\s\-_]?key|"
            r"secret[\s\-_]?key|access[\s\-_]?token|auth[\s\-_]?token|"
            r"client[\s\-_]?secret|bearer)\b\s*(?:is|=|:)?\s*[\"']?([^\s\"'\n,;]{4,128})"
        ),
    ),
    (
        Category.BANK_ACCOUNT,
        re.compile(
            r"(?i)\b(?:a/c|acc(?:oun)?t)\b[^\n\d]{0,24}(\d{9,18})\b"
        ),
    ),
]

# Words that make a nearby digit-run look like a bank account even without Luhn.
_ACCOUNT_ANCHOR = re.compile(
    r"(?i)\b(?:account|a/c|acct|ifsc|beneficiary|bank)\b"
)

_PLACEHOLDER = "[REDACTED:{}]"


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def _scan_numeric(text: str) -> list[Finding]:
    """Find Aadhaar / card / bank numbers using checksums plus context."""
    findings: list[Finding] = []

    for m in _SEPARATED_DIGITS.finditer(text):
        raw = m.group(0)
        digits = _digits_only(raw)
        start, end = m.start(), m.end()

        if len(digits) == 12 and digits[0] not in "01" and verhoeff_valid(digits):
            findings.append(Finding(Category.AADHAAR, start, end, len(raw)))
            continue

        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            findings.append(Finding(Category.CARD, start, end, len(raw)))
            continue

        # No checksum for Indian bank accounts, so require a nearby anchor word.
        if 9 <= len(digits) <= 18:
            window = text[max(0, start - 40) : min(len(text), end + 40)]
            if _ACCOUNT_ANCHOR.search(window):
                findings.append(Finding(Category.BANK_ACCOUNT, start, end, len(raw)))

    return findings


def _scan_patterns(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for pattern, category in (
        (_PAN, Category.PAN),
        (_DL, Category.DRIVING_LICENCE),
        (_VOTER_ID, Category.VOTER_ID),
        (_PASSPORT, Category.PASSPORT),
    ):
        for m in pattern.finditer(text):
            findings.append(Finding(category, m.start(), m.end(), len(m.group(0))))
    return findings


def _scan_anchored(text: str) -> list[Finding]:
    """Short secrets that are only identifiable from surrounding words."""
    findings: list[Finding] = []
    for category, pattern in _ANCHORED:
        for m in pattern.finditer(text):
            # Redact only the secret itself (group 1), never the anchor word,
            # so the sentence stays readable: "the OTP is [REDACTED:OTP]".
            findings.append(
                Finding(category, m.start(1), m.end(1), len(m.group(1)))
            )
    return findings


def _resolve_overlaps(findings: list[Finding]) -> list[Finding]:
    """Keep the longest match when spans overlap, so we never double-redact."""
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start)))
    kept: list[Finding] = []
    for f in ordered:
        if kept and f.start < kept[-1].end:
            if (f.end - f.start) > (kept[-1].end - kept[-1].start):
                kept[-1] = f
            continue
        kept.append(f)
    return kept


def scan(text: str) -> list[Finding]:
    """Detect excluded data without modifying the text."""
    if not text:
        return []
    found = _scan_anchored(text) + _scan_patterns(text) + _scan_numeric(text)
    return _resolve_overlaps(found)


def redact(text: str) -> RedactionResult:
    """Replace every excluded value with a category placeholder.

    This is the function every other module should call. Nothing bypasses it.
    """
    if not text:
        return RedactionResult(text="", findings=[])

    findings = scan(text)
    if not findings:
        return RedactionResult(text=text, findings=[])

    out: list[str] = []
    cursor = 0
    for f in findings:
        out.append(text[cursor : f.start])
        out.append(_PLACEHOLDER.format(f.category.value))
        cursor = f.end
    out.append(text[cursor:])

    return RedactionResult(text="".join(out), findings=findings)


class StreamRedactor:
    """Redacts a token stream without letting a split secret slip through.

    A streamed reply arrives in fragments, so a card number can be delivered as
    "4111 1111 " then "1111 1111". Scanning each fragment alone finds nothing and
    the whole number reaches the screen. This holds back a tail long enough to
    contain any single excluded value, emitting only the part that can no longer
    be extended into a match.

    Usage:
        r = StreamRedactor()
        for chunk in stream:
            out = r.feed(chunk)
            if out: print(out, end="")
        print(r.flush(), end="")
    """

    # Longer than the longest pattern we match, including separators and the
    # anchor words that precede short secrets.
    TAIL = 160

    def __init__(self) -> None:
        self._buffer = ""
        self.findings: list[Finding] = []

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        if len(self._buffer) <= self.TAIL:
            return ""

        # Everything before the tail is safe to scan and release.
        head, self._buffer = self._buffer[: -self.TAIL], self._buffer[-self.TAIL :]
        result = redact(head)
        self.findings.extend(result.findings)
        return result.text

    def flush(self) -> str:
        result = redact(self._buffer)
        self.findings.extend(result.findings)
        self._buffer = ""
        return result.text

    @property
    def is_clean(self) -> bool:
        return not self.findings


def is_safe(text: str) -> bool:
    """True when the text contains nothing in an excluded category."""
    return not scan(text)


def assert_safe(text: str, where: str = "unspecified") -> str:
    """Raise if excluded data is present. Use on paths that must never leak."""
    result = redact(text)
    if not result.is_clean:
        raise ValueError(
            f"excluded data blocked at {where}: {result.summary()}"
        )
    return text
