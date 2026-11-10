"""Global, deterministic duplicate decisions with full-document verification.

LSH discovery is approximate. Removal uses full word-shingle sets, not the
sampled shingles used to discover candidates. Representatives are retained
greedily in fixed input order; removed near duplicates are never promoted to
new representatives, preventing transitive similarity drift.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from array import array
from collections import defaultdict

import xxhash

from src.phase1_v2_dedup import minhash_signature


def candidate_features(text: str, discover: bool = True):
    normalized = text.casefold()
    words = normalized.split()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    if not discover or len(words) < 300:
        return digest, len(words), None, 0
    if len(words) <= 512:
        segments = [words]
    else:
        mid = len(words) // 2
        segments = [words[:192], words[mid - 96:mid + 96], words[-192:]]
    hashes = {
        xxhash.xxh3_64_intdigest(" ".join(segment[start:start + 5]).encode("utf-8"))
        for segment in segments for start in range(len(segment) - 4)
    }
    if len(hashes) < 10:
        return digest, len(words), None, len(hashes)
    signature = minhash_signature(array("Q", sorted(hashes)))
    return digest, len(words), signature, len(hashes)


def full_shingles(text: str):
    words = text.casefold().split()
    # Actual strings, not probabilistic hash equality, decide removal.
    return {" ".join(words[start:start + 5]) for start in range(len(words) - 4)}


def split_for_hash(digest: bytes) -> str:
    return "val" if int.from_bytes(digest[:8], "big") / 2**64 < .05 else "train"


class ConservativeNearIndex:
    def __init__(self):
        self.buckets = defaultdict(list)
        self.word_counts = {}
        self.full_verifications = 0

    @staticmethod
    def keys(signature):
        return [(band, *signature[band * 4:(band + 1) * 4]) for band in range(8)]

    def add(self, document_id: int, word_count: int, signature):
        if signature is None:
            return
        self.word_counts[document_id] = word_count
        for key in self.keys(signature):
            self.buckets[key].append(document_id)

    def find(self, text: str, words: int, signature, old_shingles):
        if signature is None:
            return None
        candidates = set()
        for key in self.keys(signature):
            candidates.update(self.buckets.get(key, ()))
        current = None
        for identifier in sorted(candidates):
            previous_words = self.word_counts[identifier]
            if min(words, previous_words) * 100 < max(words, previous_words) * 98:
                continue
            if current is None:
                current = full_shingles(text)
            previous = old_shingles(identifier)
            shared = len(current & previous)
            union = len(current) + len(previous) - shared
            self.full_verifications += 1
            if union and shared * 100 >= union * 98:
                return identifier, shared / union, min(words, previous_words) / max(words, previous_words)
        return None


def open_database(path):
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA cache_size=-262144")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            document_id TEXT UNIQUE NOT NULL,
            source_index INTEGER NOT NULL,
            input_index INTEGER NOT NULL,
            byte_offset INTEGER NOT NULL,
            words INTEGER NOT NULL,
            normalized_hash BLOB NOT NULL,
            disposition TEXT NOT NULL,
            matched_id INTEGER,
            canonical_id INTEGER NOT NULL,
            full_jaccard REAL,
            word_length_ratio REAL,
            signature BLOB,
            sampled_distinct_shingles INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoint (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            state_json TEXT NOT NULL
        );
    """)
    return connection


class CorpusDeduplicator:
    def __init__(self, connection, read_text):
        self.connection = connection
        self.read_text = read_text
        self.exact = {}
        self.near = ConservativeNearIndex()
        # Rebuild only from committed rows, preserving global input order.
        for row in connection.execute("SELECT id, normalized_hash, canonical_id, disposition, words, signature FROM documents ORDER BY id"):
            identifier, digest, canonical, disposition, words, signature = row
            self.exact.setdefault(digest, (identifier, canonical))
            if disposition == "retained" and signature is not None:
                self.near.add(identifier, words, struct.unpack("<32Q", signature))

    def process(self, identifier, source_index, offset, row, features):
        digest, words, signature, distinct = features
        if digest.hex() != row["normalized_sha256"] or words != row["words"]:
            raise ValueError(f"Candidate text/hash/count mismatch: {row['document_id']}")
        matched, canonical, score, ratio = None, identifier, None, None
        if digest in self.exact:
            matched, canonical = self.exact[digest]
            disposition = "duplicate_exact"
        else:
            # The runner can supply a bounded cache for random-access old texts.
            match = self.near.find(row["text"], words, signature, self.read_text)
            if match is not None:
                matched, score, ratio = match
                canonical = matched
                disposition = "duplicate_near"
            else:
                disposition = "retained"
                self.near.add(identifier, words, signature)
            self.exact[digest] = (identifier, canonical)
        self.connection.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (identifier, row["document_id"], source_index, row["input_index"], offset,
             words, digest, disposition, matched, canonical, score, ratio,
             struct.pack("<32Q", *signature) if signature is not None else None, distinct),
        )
        return disposition
