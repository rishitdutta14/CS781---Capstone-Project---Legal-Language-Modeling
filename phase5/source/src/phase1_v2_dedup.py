"""Conservative near-duplicate detection for legal documents.

MinHash/LSH finds candidates. Exact sampled-shingle Jaccard and a word-length
check make the removal decision. Short orders are handled by exact hashing only
because formulaic wording can describe different cases.
"""

from __future__ import annotations

from array import array
from collections import defaultdict

import numpy as np
import xxhash


PERMUTATIONS = 32
BAND_SIZE = 4
_rng = np.random.default_rng(781)
_MULTIPLIERS = _rng.integers(1, 2**63, size=PERMUTATIONS, dtype=np.uint64) | np.uint64(1)
_INCREMENTS = _rng.integers(0, 2**63, size=PERMUTATIONS, dtype=np.uint64)


def sampled_shingles(text: str, width: int = 5) -> array:
    words = text.casefold().split()
    if len(words) <= 512:
        segments = [words]
    else:
        mid = len(words) // 2
        segments = [words[:192], words[mid - 96:mid + 96], words[-192:]]
    hashes = set()
    for segment in segments:
        for start in range(len(segment) - width + 1):
            shingle = " ".join(segment[start:start + width])
            hashes.add(xxhash.xxh3_64_intdigest(shingle.encode("utf-8")))
    return array("Q", sorted(hashes))


def minhash_signature(shingles: array) -> tuple[int, ...]:
    base = np.frombuffer(shingles, dtype=np.uint64)
    values = base[:, None] * _MULTIPLIERS[None, :] + _INCREMENTS[None, :]
    return tuple(int(value) for value in values.min(axis=0))


class NearDuplicateIndex:
    def __init__(self, threshold: float = 0.98, min_words: int = 300):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self.min_words = min_words
        self.buckets = defaultdict(list)
        self.representatives = {}
        self.checked_candidates = 0
        self.removed = 0

    def find_or_add(self, document_id: str, text: str) -> str | None:
        word_count = len(text.split())
        if word_count < self.min_words:
            return None
        shingles = sampled_shingles(text)
        if len(shingles) < 10:
            return None
        signature = minhash_signature(shingles)
        bucket_keys = [
            (band, signature[band * BAND_SIZE:(band + 1) * BAND_SIZE])
            for band in range(PERMUTATIONS // BAND_SIZE)
        ]
        candidates = set()
        for key in bucket_keys:
            candidates.update(self.buckets[key])
        for candidate_id in candidates:
            old_words, old_shingles = self.representatives[candidate_id]
            if min(word_count, old_words) / max(word_count, old_words) < 0.98:
                continue
            self.checked_candidates += 1
            old_set = set(old_shingles)
            shared = sum(value in old_set for value in shingles)
            union = len(shingles) + len(old_shingles) - shared
            if union and shared / union >= self.threshold:
                self.removed += 1
                return candidate_id
        self.representatives[document_id] = (word_count, shingles)
        for key in bucket_keys:
            self.buckets[key].append(document_id)
        return None
