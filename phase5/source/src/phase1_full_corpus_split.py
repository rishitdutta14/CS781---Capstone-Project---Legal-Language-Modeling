"""Frozen document membership referencing immutable, complete cleaned texts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

from src.phase1_full_corpus_prepare import SOURCES, atomic_json, file_sha256, text_statistics
from src.phase1_full_corpus_dedup import split_for_hash


def read_database(path):
    connection = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-65536")
    return connection


def partition_jobs(extraction, output_dir, partition_bytes):
    jobs = []
    for source_index, source in enumerate(SOURCES):
        item = extraction["sources"][source]["outputs"]["candidates.jsonl"]
        path = Path(item["path"])
        for number, start in enumerate(range(0, item["bytes"], partition_bytes)):
            jobs.append({
                "name": f"{source}_{number:04d}", "source": source,
                "source_index": source_index, "input_file": str(path),
                "input_sha256": item["sha256"], "start": start,
                "end": min(start + partition_bytes, item["bytes"]),
                "output_dir": str(output_dir),
            })
    return jobs


def records_in_range(path, start, end):
    """Assign each whole JSONL record to the partition containing its start."""
    with Path(path).open("rb") as handle:
        if start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        while handle.tell() < end:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            yield offset, len(line), json.loads(line)


def decision_rows(connection, source_index, start=None, end=None):
    conditions = "d.source_index=?"
    values = [source_index]
    if start is not None:
        conditions += " AND d.byte_offset>=? AND d.byte_offset<?"
        values.extend((start, end))
    return connection.execute(f"""
        SELECT d.id,d.document_id,d.input_index,d.byte_offset,d.words,
               d.normalized_hash,d.disposition,d.matched_id,d.canonical_id,
               d.full_jaccard,d.word_length_ratio,
               r.document_id,r.normalized_hash,r.disposition,
               r.source_index,r.byte_offset,r.words,
               m.document_id,m.normalized_hash,m.canonical_id
        FROM documents d
        JOIN documents r ON r.id=d.canonical_id
        LEFT JOIN documents m ON m.id=d.matched_id
        WHERE {conditions} ORDER BY d.byte_offset
    """, values)


def verify_relationship(decision):
    (identifier, doc_id, _, _, _, digest, disposition, matched, canonical,
     score, ratio, root_id, root_hash, root_disposition, _, _, _,
     match_id, match_hash, match_root) = decision
    if root_disposition != "retained":
        raise ValueError(f"Non-retained canonical root: {doc_id}")
    if disposition == "retained":
        if canonical != identifier or matched is not None or digest != root_hash:
            raise ValueError(f"Invalid retained relationship: {doc_id}")
    elif disposition == "duplicate_exact":
        if matched is None or match_id is None or match_hash != digest or match_root != canonical:
            raise ValueError(f"Invalid exact relationship: {doc_id}")
    elif disposition == "duplicate_near":
        if matched != canonical or match_id != root_id or score is None or ratio is None or score < .98 or ratio < .98:
            raise ValueError(f"Invalid near relationship: {doc_id}")
    else:
        raise ValueError(f"Unknown disposition: {disposition}")


def export_partition(job, database, candidate_files):
    output = Path(job["output_dir"])
    name = job["name"]
    manifest = output / f"{name}.json"
    if manifest.exists() or any(output.glob(f"*_{name}.jsonl*")):
        raise FileExistsError(f"Existing partition must be preserved: {name}")
    train_path = output / f"train_{name}.jsonl"
    val_path = output / f"val_{name}.jsonl"
    connection = read_database(database)
    decisions = iter(decision_rows(connection, job["source_index"], job["start"], job["end"]))
    random_files = [Path(path).open("rb") for path in candidate_files]
    counts, scripts, formats, groups = Counter(), {s: Counter() for s in ("train", "val")}, {s: Counter() for s in ("train", "val")}, {s: Counter() for s in ("train", "val")}
    started = time.monotonic()

    @lru_cache(maxsize=4)
    def old_word_tuples(source_index, offset):
        handle = random_files[source_index]
        handle.seek(offset)
        text = json.loads(handle.readline())["text"]
        words = text.casefold().split()
        return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}

    try:
        with train_path.with_suffix(".jsonl.partial").open("w", encoding="utf-8") as train, val_path.with_suffix(".jsonl.partial").open("w", encoding="utf-8") as val:
            for offset, length, row in records_in_range(job["input_file"], job["start"], job["end"]):
                decision = next(decisions, None)
                if decision is None:
                    raise ValueError("Candidate has no decision")
                identifier, doc_id, input_index, expected_offset, words, digest, disposition = decision[:7]
                if (doc_id, input_index, offset) != (row["document_id"], row["input_index"], expected_offset):
                    raise ValueError("Record/decision location mismatch")
                text = row["text"]
                if text != " ".join(text.split()) or hashlib.sha256(text.casefold().encode()).digest() != digest or digest.hex() != row["normalized_sha256"] or len(text.split()) != words:
                    raise ValueError(f"Text/fingerprint/count mismatch: {doc_id}")
                verify_relationship(decision)
                counts["candidates"] += 1
                counts[disposition] += 1
                if disposition == "duplicate_near":
                    # Independent tuple-based full-document verification; no
                    # sampled grams or stored score alone can pass this gate.
                    current_words = text.casefold().split()
                    current = {tuple(current_words[i:i + 5]) for i in range(len(current_words) - 4)}
                    previous = old_word_tuples(decision[14], decision[15])
                    common = len(current.intersection(previous))
                    union = len(current) + len(previous) - common
                    root_words = decision[16]
                    ratio = min(words, root_words) / max(words, root_words)
                    if not union or common * 100 < union * 98 or min(words, root_words) * 100 < max(words, root_words) * 98:
                        raise ValueError(f"Full near similarity failed: {doc_id}")
                    if abs(common / union - decision[9]) > 1e-12 or abs(ratio - decision[10]) > 1e-12:
                        raise ValueError(f"Stored similarity mismatch: {doc_id}")
                    counts["near_pairs_independently_reverified"] += 1
                if disposition != "retained":
                    continue
                split = split_for_hash(digest)
                character_count = len(text)
                utf8_bytes = len(text.encode("utf-8"))
                letter_counts, format_counts = text_statistics(text)
                scripts[split].update(letter_counts)
                formats[split].update(format_counts)
                if letter_counts:
                    dominant, largest = max(letter_counts.items(), key=lambda pair: (pair[1], pair[0]))
                    group = dominant if largest / sum(letter_counts.values()) >= .8 else "MIXED"
                else:
                    group = "UNKNOWN"
                groups[split][group] += 1
                counts[f"{split}_documents"] += 1
                counts[f"{split}_words"] += words
                counts[f"{split}_characters"] += character_count
                counts[f"{split}_utf8_bytes"] += utf8_bytes
                reference = {
                    "document_id": doc_id, "source": job["source"], "input_index": input_index,
                    "decision_id": identifier, "split": split,
                    "candidate_file": job["input_file"], "byte_offset": offset, "byte_length": length,
                    "normalized_sha256": digest.hex(), "words": words,
                    "characters": character_count, "utf8_bytes": utf8_bytes,
                    "script_group": group,
                }
                target = train if split == "train" else val
                target.write(json.dumps(reference, ensure_ascii=False, separators=(",", ":")) + "\n")
            if next(decisions, None) is not None:
                raise ValueError("Decision has no candidate")
        train_path.with_suffix(".jsonl.partial").replace(train_path)
        val_path.with_suffix(".jsonl.partial").replace(val_path)
        result = {
            "partition": job, "counts": dict(counts),
            "split_script_letter_counts": {s: dict(v) for s, v in scripts.items()},
            "split_format_character_counts": {s: dict(v) for s, v in formats.items()},
            "split_script_group_documents": {s: dict(v) for s, v in groups.items()},
            "outputs": {s: {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)} for s, path in (("train", train_path), ("val", val_path))},
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(manifest, result)
        return result
    finally:
        connection.close()
        for handle in random_files:
            handle.close()


def export_record_outcomes(source_index, audit_path, database, output_dir):
    source = SOURCES[source_index]
    path = Path(output_dir) / f"record_outcomes_{source}.jsonl"
    if path.exists() or path.with_suffix(".jsonl.partial").exists():
        raise FileExistsError(path)
    connection = read_database(database)
    decisions = iter(decision_rows(connection, source_index))
    decision = next(decisions, None)
    counts = Counter()
    with Path(audit_path).open(encoding="utf-8") as original, path.with_suffix(".jsonl.partial").open("w", encoding="utf-8") as output:
        for index, line in enumerate(original):
            record = json.loads(line)
            if record["source"] != source or record["input_index"] != index or record["document_id"] != f"{source}:{index}":
                raise ValueError("Original audit is not complete source/index order")
            counts["input_records"] += 1
            if record["stage_outcome"] == "rejected":
                if not record["reason"] or (decision is not None and decision[2] <= index):
                    raise ValueError("Rejected record has a candidate decision")
                record.update(final_outcome="rejected", split=None)
                counts["rejected"] += 1
            else:
                if decision is None or decision[1:3] != (record["document_id"], index):
                    raise ValueError("Audit/decision mismatch")
                verify_relationship(decision)
                if decision[5].hex() != record["normalized_sha256"] or decision[4] != record["words"]:
                    raise ValueError("Audit/fingerprint/count mismatch")
                disposition = decision[6]
                split = split_for_hash(decision[12])
                record.update(
                    final_outcome="retained" if disposition == "retained" else disposition,
                    split=split, canonical_document_id=decision[11], matched_document_id=decision[17],
                    full_jaccard=decision[9], word_length_ratio=decision[10],
                )
                counts[disposition] += 1
                if disposition == "retained":
                    counts[f"{split}_documents"] += 1
                decision = next(decisions, None)
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    if decision is not None:
        raise ValueError("Unaccounted decision after raw audit EOF")
    connection.close()
    path.with_suffix(".jsonl.partial").replace(path)
    return {"source": source, "counts": dict(counts), "path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def audit_reference_partition(partition_result):
    """Second pass checks every retained reference against actual full text."""
    counts = Counter()
    for split in ("train", "val"):
        item = partition_result["outputs"][split]
        path = Path(item["path"])
        if file_sha256(path) != item["sha256"]:
            raise ValueError("Membership shard changed")
        candidate = Path(partition_result["partition"]["input_file"])
        with candidate.open("rb") as texts, path.open(encoding="utf-8") as references:
            previous_offset = -1
            for line in references:
                ref = json.loads(line)
                if ref["split"] != split or ref["candidate_file"] != str(candidate) or ref["byte_offset"] <= previous_offset:
                    raise ValueError("Reference location/order/split mismatch")
                previous_offset = ref["byte_offset"]
                texts.seek(ref["byte_offset"])
                raw = texts.readline()
                if len(raw) != ref["byte_length"]:
                    raise ValueError("Byte length mismatch")
                document = json.loads(raw)
                text = document["text"]
                digest = hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
                expected_split = "val" if int(digest[:16], 16) / 18446744073709551616 < .05 else "train"
                if digest != ref["normalized_sha256"] or expected_split != split:
                    raise ValueError("Independent split/hash rule failure")
                if (ref["document_id"], ref["source"], ref["input_index"]) != (document["document_id"], document["source"], document["input_index"]):
                    raise ValueError("Identity/provenance mismatch")
                if len(text.split()) != ref["words"] or len(text) != ref["characters"] or len(text.encode()) != ref["utf8_bytes"]:
                    raise ValueError("Independent text count mismatch")
                counts[f"{split}_documents"] += 1
                counts[f"{split}_words"] += ref["words"]
                counts[f"{split}_characters"] += ref["characters"]
                counts[f"{split}_utf8_bytes"] += ref["utf8_bytes"]
        for suffix in ("documents", "words", "characters", "utf8_bytes"):
            key = f"{split}_{suffix}"
            if counts[key] != partition_result["counts"].get(key, 0):
                raise ValueError(f"Partition statistics do not reconcile: {key}")
    return {"partition": partition_result["partition"]["name"], "passed": True, "counts": dict(counts)}
