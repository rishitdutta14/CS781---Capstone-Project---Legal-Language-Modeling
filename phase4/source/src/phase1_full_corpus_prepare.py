"""Versioned, full-coverage extraction for the new Phase 1 corpus.

This stage creates candidates, not a final training split. Global duplicate
decisions and split assignment must be audited before any BPE fitting.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path


SOURCES = ("acts", "high_courts", "supreme_court", "tribunals")
SCRAPE_COUNTER = re.compile(r"^\s*\[Cites\s+\d+,\s*Cited by\s+\d+\]\s*", re.I)
SIGNATURE = re.compile(r"(?im)^\s*Digitally signed by\s*:-\s*$")
SURROGATE = re.compile("[\ud800-\udfff]")
SCRIPTS = (
    "DEVANAGARI", "BENGALI", "GUJARATI", "GURMUKHI", "TAMIL", "TELUGU",
    "KANNADA", "MALAYALAM", "ORIYA", "ARABIC", "HEBREW", "CYRILLIC",
    "GREEK", "THAI", "LATIN",
)
# Whitespace is handled by A1's canonical split/join, not deleted as a control.
CONTROL_TRANSLATION = {
    value: None for value in (*range(32), *range(127, 160))
    if not chr(value).isspace()
}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_json_records(path: Path, digest=None, chunk_size=4 * 1024 * 1024):
    """Strict streaming JSON array reader, hashing the actual bytes consumed.

    Rejects missing/trailing commas, non-object rows, truncation, invalid UTF-8
    and trailing content. Unlike the older reader, it validates the file tail.
    """
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")("strict")
    with path.open("rb") as handle:
        buffer, position, eof = "", 0, False

        def refill():
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            block = handle.read(chunk_size)
            if digest is not None:
                digest.update(block)
            eof = not block
            buffer += utf8.decode(block, final=eof)

        def skip_space():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n":
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        refill()
        skip_space()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected top-level JSON array: {path}")
        position += 1
        allow_empty = True
        while True:
            skip_space()
            if position >= len(buffer):
                raise ValueError(f"Truncated JSON array: {path}")
            if buffer[position] == "]":
                if not allow_empty:
                    raise ValueError(f"Trailing comma: {path}")
                position += 1
                break
            while True:
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError(f"Invalid JSON: {path}: {error}") from error
                    refill()
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record: {path}")
            position = end
            yield record
            skip_space()
            if position >= len(buffer):
                raise ValueError(f"Truncated JSON array: {path}")
            delimiter = buffer[position]
            position += 1
            if delimiter == "]":
                break
            if delimiter != ",":
                raise ValueError(f"Missing comma: {path}")
            allow_empty = False
        # Drain the tail so input SHA includes the entire file and any trailing
        # invalid UTF-8 or non-whitespace content is caught.
        while True:
            if buffer[position:].strip(" \t\r\n"):
                raise ValueError(f"Trailing content: {path}")
            position = len(buffer)
            if eof:
                break
            refill()


def clean_text(source: str, text: str) -> tuple[str | None, dict]:
    if not isinstance(text, str):
        raise TypeError("Official record must have string text")
    flags = {
        "reason": None, "scrape_counter_removed": False,
        "signature_footer_removed": False, "nfc_changed": False,
        "control_characters_removed": 0,
        "replacement_characters": text.count("\ufffd"),
    }
    if SURROGATE.search(text):
        flags["reason"] = "invalid_unicode_surrogate"
        return None, flags
    if not text.strip():
        flags["reason"] = "empty"
        return None, flags
    if flags["replacement_characters"] > max(10, int(len(text) * .05)):
        flags["reason"] = "excessive_replacement_characters"
        return None, flags
    normalized = unicodedata.normalize("NFC", text)
    flags["nfc_changed"] = normalized != text
    text = normalized
    if SCRAPE_COUNTER.match(text):
        text = SCRAPE_COUNTER.sub("", text, count=1)
        flags["scrape_counter_removed"] = True
    if source == "high_courts":
        for match in SIGNATURE.finditer(text):
            if len(text) - match.start() <= 300 and match.start() >= len(text) * .7:
                text = text[:match.start()]
                flags["signature_footer_removed"] = True
                break
    without_controls = text.translate(CONTROL_TRANSLATION)
    flags["control_characters_removed"] = len(text) - len(without_controls)
    canonical = " ".join(without_controls.replace("\ufffd", " ").split())
    if not canonical:
        flags["reason"] = "empty_after_cleaning"
        return None, flags
    # No minimum word count. Legitimate short orders remain candidates.
    # All Cf characters, including ZWJ/ZWNJ and direction controls, survive.
    return canonical, flags


@lru_cache(maxsize=None)
def character_info(character: str):
    category = unicodedata.category(character)
    script = None
    if character.isalpha():
        name = unicodedata.name(character, "")
        script = next((label for label in SCRIPTS if label in name), "OTHER")
    return category, script


def text_statistics(text: str) -> tuple[Counter, Counter]:
    """Full-text counts; no prefix restriction or document sampling."""
    scripts, formats = Counter(), Counter()
    for character, count in Counter(text).items():
        category, script = character_info(character)
        if script is not None:
            scripts[script] += count
        if category == "Cf":
            formats[f"U+{ord(character):04X}"] += count
    return scripts, formats


def prepare_source(source: str, input_path: Path, output_dir: Path) -> dict:
    """Extract every source record once into immutable candidate/audit files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / "completed.json"
    if final.exists() or any(output_dir.glob("*.jsonl*")):
        raise FileExistsError(f"Existing source outputs must be preserved: {output_dir}")
    before = input_path.stat()
    digest = hashlib.sha256()
    counts = Counter()
    raw_scripts, clean_scripts, raw_formats, clean_formats = (Counter() for _ in range(4))
    started = time.monotonic()
    candidates = output_dir / "candidates.jsonl.partial"
    audit = output_dir / "record_audit.jsonl.partial"
    with candidates.open("w", encoding="utf-8") as data, audit.open("w", encoding="utf-8") as records:
        for index, row in enumerate(iter_json_records(input_path, digest)):
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{source}:{index} lacks string text")
            document_id = f"{source}:{index}"
            counts["input_records"] += 1
            counts["raw_characters"] += len(text)
            counts["raw_words"] += len(text.split())
            scripts, formats = text_statistics(text)
            raw_scripts.update(scripts)
            raw_formats.update(formats)
            cleaned, flags = clean_text(source, text)
            item = {"document_id": document_id, "source": source, "input_index": index, **flags}
            # Surrogates are accounted for as rejected, not silently repaired.
            item["raw_text_sha256"] = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
            for flag in ("scrape_counter_removed", "signature_footer_removed", "nfc_changed", "control_characters_removed", "replacement_characters"):
                counts[flag] += flags[flag]
            if cleaned is None:
                counts["rejected_records"] += 1
                counts[f"rejected_{flags['reason']}"] += 1
                item["stage_outcome"] = "rejected"
            else:
                counts["candidate_records"] += 1
                words = len(cleaned.split())
                counts["candidate_words"] += words
                counts["candidate_characters"] += len(cleaned)
                counts["candidate_utf8_bytes"] += len(cleaned.encode("utf-8"))
                counts["candidates_below_10_words"] += words < 10
                scripts, formats = text_statistics(cleaned)
                clean_scripts.update(scripts)
                clean_formats.update(formats)
                identity = hashlib.sha256(cleaned.casefold().encode("utf-8")).hexdigest()
                item.update(stage_outcome="candidate_pending_global_dedup_and_split", normalized_sha256=identity, words=words)
                document = {
                    "document_id": document_id, "source": source, "input_index": index,
                    "normalized_sha256": identity, "words": words,
                    "metadata": {key: value for key, value in row.items() if key != "text"},
                    "text": cleaned,
                }
                data.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
            records.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            if counts["input_records"] % 25000 == 0:
                data.flush()
                records.flush()
                progress = {"source": source, "status": "running", "counts": dict(counts), "elapsed_seconds": time.monotonic() - started}
                atomic_json(output_dir / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    after = input_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"Input changed during extraction: {input_path}")
    if counts["input_records"] != counts["candidate_records"] + counts["rejected_records"]:
        raise RuntimeError("Record accounting mismatch")
    candidates.replace(output_dir / "candidates.jsonl")
    audit.replace(output_dir / "record_audit.jsonl")
    result = {
        "status": "source_extraction_complete_not_final_split", "source": source,
        "input_file": str(input_path.resolve()), "input_bytes": before.st_size,
        "input_sha256": digest.hexdigest(), "counts": dict(counts),
        "complete_file_scan": True, "sampling": False, "truncation": False,
        "raw_script_letter_counts": dict(raw_scripts),
        "candidate_script_letter_counts": dict(clean_scripts),
        "raw_format_character_counts": dict(raw_formats),
        "candidate_format_character_counts": dict(clean_formats),
        "script_statistics_are_language_identification": False,
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {
            name: {"path": str((output_dir / name).resolve()), "bytes": (output_dir / name).stat().st_size,
                   "sha256": file_sha256(output_dir / name)}
            for name in ("candidates.jsonl", "record_audit.jsonl")
        },
        "global_deduplication_complete": False, "split_frozen": False,
    }
    atomic_json(final, result)
    atomic_json(output_dir / "progress.json", result)
    return result
