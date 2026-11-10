"""A1 byte-BPE with exact precomputed frequencies and equivalent heap updates.

Encoding, decoding, merge application, vocabulary layout and file format are
inherited unchanged. The original A1 implementation remains untouched.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import numpy as np

from src.tokenization.a1_byte_bpe import ByteBPE

BYTE_HEX = tuple(f"{value:02x}" for value in range(256))


class FrequencyByteBPE(ByteBPE):
    @staticmethod
    def _word_bytes(word):
        # Equal strings, shared storage; no change to A1 byte symbols.
        return [BYTE_HEX[value] for value in (" " + word).encode("utf-8")]

    def fit_counts(self, word_counts, *, progress_callback=None, progress_every=250,
                   initialization_callback=None, replay_merges=()):
        if self.vocab_size != 260 or self.merges:
            raise ValueError("Start with fresh byte vocabulary; replay checkpoints explicitly")
        if progress_every < 1:
            raise ValueError("Progress interval must be positive")
        if any(type(count) is not int or count < 1 or count > 2**63 - 1 for count in word_counts.values()):
            raise ValueError("Frequencies must be positive exact int64-compatible integers")
        words = sorted(word_counts)
        frequencies = np.asarray([word_counts[word] for word in words], dtype=np.int64)
        if np.any(frequencies < 1):
            raise ValueError("Word frequencies must be exact positive integers")
        sequences = []
        for index, word in enumerate(words):
            sequences.append(self._word_bytes(word))
            if initialization_callback and (index + 1) % 100000 == 0:
                initialization_callback("word_sequences", index + 1, len(words))
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for word_id, sequence in enumerate(sequences):
            local = Counter(zip(sequence, sequence[1:]))
            for pair, occurrences in local.items():
                pair_counts[pair] += int(frequencies[word_id]) * occurrences
                pair_words[pair].add(word_id)
            if initialization_callback and (word_id + 1) % 100000 == 0:
                initialization_callback("pair_index", word_id + 1, len(words))
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
        while self.vocab_size < self.target_vocab_size and heap:
            negative_count, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if -negative_count != count:
                continue
            if count <= 1:
                break
            rank = len(self.merges)
            replaying = rank < len(replay_merges)
            if replaying and pair != tuple(replay_merges[rank]):
                raise ValueError(f"Checkpoint does not match exact A1 selection at merge {rank}")
            merged = pair[0] + pair[1]
            self.merges.append([pair[0], pair[1]])
            if merged not in self.stoi:
                self.stoi[merged] = len(self.itos)
                self.itos.append(merged)
            changed_pairs = set()
            for word_id in list(pair_words.get(pair, ())):
                old = sequences[word_id]
                old_local = Counter(zip(old, old[1:]))
                new = self._merge_sequence(old, pair, merged)
                new_local = Counter(zip(new, new[1:]))
                sequences[word_id] = new
                weight = int(frequencies[word_id])
                for changed in old_local.keys() | new_local.keys():
                    delta = (new_local[changed] - old_local[changed]) * weight
                    if delta:
                        pair_counts[changed] += delta
                        if new_local[changed]:
                            pair_words[changed].add(word_id)
                        else:
                            pair_words[changed].discard(word_id)
                        changed_pairs.add(changed)
            # No pair selection occurs inside the word-update loop. Pushing
            # each final count once gives exactly the same next max-frequency,
            # lexical-tie pair as A1, without millions of redundant heap entries.
            for changed in changed_pairs:
                if pair_counts[changed] > 0:
                    heapq.heappush(heap, (-pair_counts[changed], changed))
            if progress_callback and not replaying and (
                len(self.merges) % progress_every == 0 or self.vocab_size >= self.target_vocab_size
            ):
                progress_callback(self, count, len(self.merges))
        if len(self.merges) < len(replay_merges):
            raise ValueError("Checkpoint is longer than eligible merge chain")
        self._ranks = {tuple(pair): rank for rank, pair in enumerate(self.merges)}
        return self
