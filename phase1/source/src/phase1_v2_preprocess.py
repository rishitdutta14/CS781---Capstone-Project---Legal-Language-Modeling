"""Conservative, auditable cleaning for the official Phase 1 legal corpus."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass


LEADING_SCRAPE_COUNTER = re.compile(
    r"^\s*\[Cites\s+\d+,\s*Cited by\s+\d+\]\s*",
    flags=re.IGNORECASE,
)
SIGNATURE_MARKER = re.compile(r"(?im)^\s*Digitally signed by\s*:-\s*$")


@dataclass(frozen=True)
class CleaningResult:
    text: str | None
    reason: str | None
    replacement_characters: int
    removed_scrape_counter: bool
    removed_signature_footer: bool
    removed_control_characters: int


def clean_legal_text(source: str, text: str) -> CleaningResult:
    """Keep legal content; remove only verified extraction artifacts.

    Records below 10 words and records with more than 5% unknown replacement
    characters are excluded. Short legal orders above that floor are retained.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    replacement_count = text.count("\ufffd")
    if not text.strip():
        return CleaningResult(None, "empty", replacement_count, False, False, 0)
    if replacement_count > max(10, int(len(text) * 0.05)):
        return CleaningResult(
            None, "excessive_replacement_characters", replacement_count, False, False, 0
        )

    text = unicodedata.normalize("NFC", text)
    removed_scrape_counter = bool(LEADING_SCRAPE_COUNTER.match(text))
    if removed_scrape_counter:
        text = LEADING_SCRAPE_COUNTER.sub("", text, count=1)

    removed_signature_footer = False
    if source == "high_courts":
        for match in SIGNATURE_MARKER.finditer(text):
            tail = text[match.start():]
            if len(tail) <= 300 and match.start() >= len(text) * 0.7:
                text = text[:match.start()]
                removed_signature_footer = True
                break

    cleaned = []
    removed_controls = 0
    for character in text:
        if character == "\ufffd":
            cleaned.append(" ")
        elif character.isspace():
            cleaned.append(" ")
        elif unicodedata.category(character) in {"Cc", "Cf", "Cs"}:
            removed_controls += 1
        else:
            cleaned.append(character)
    canonical = " ".join("".join(cleaned).split())
    if len(canonical.split()) < 10:
        return CleaningResult(
            None, "fewer_than_10_words", replacement_count,
            removed_scrape_counter, removed_signature_footer, removed_controls,
        )
    return CleaningResult(
        canonical, None, replacement_count, removed_scrape_counter,
        removed_signature_footer, removed_controls,
    )


def content_hash(text: str) -> str:
    """Case and whitespace insensitive exact-document fingerprint."""

    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def script_letter_counts(text: str, limit: int = 20_000) -> Counter:
    """Count Unicode scripts for auditing; this is not a language classifier."""

    counts = Counter()
    scripts = (
        "DEVANAGARI", "BENGALI", "GUJARATI", "GURMUKHI", "TAMIL",
        "TELUGU", "KANNADA", "MALAYALAM", "ORIYA", "ARABIC",
    )
    for character in text[:limit]:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        script = next((value for value in scripts if value in name), None)
        if script is None:
            script = "LATIN" if "LATIN" in name else "OTHER"
        counts[script] += 1
    return counts
