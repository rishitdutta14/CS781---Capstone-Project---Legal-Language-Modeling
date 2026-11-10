"""Assignment 1 byte-level BPE tokenizer.

The merge-learning and encoding behavior is preserved from the submitted
Assignment 1 notebook. Convenience methods for integer IDs and decoding are
added so the tokenizer can be reused by the Phase 1 language model.
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


class ByteBPE:
    """Deterministic byte-level BPE with word-initial space boundaries."""

    RESERVED = ["<pad>", "<unk>", "<s>", "</s>"]

    def __init__(self, target_vocab_size: int = 5000) -> None:
        if target_vocab_size < 260:
            raise ValueError("byte BPE needs four specials plus all 256 byte values")
        self.target_vocab_size = target_vocab_size
        self.merges: list[list[str]] = []
        self.itos = self.RESERVED + [f"{value:02x}" for value in range(256)]
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self._ranks: dict[tuple[str, str], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_token_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return self.stoi["<s>"]

    @property
    def eos_token_id(self) -> int:
        return self.stoi["</s>"]

    @staticmethod
    def normalize(text: str) -> str:
        """Apply the whitespace normalization used by the A1 implementation."""

        return " ".join(re.findall(r"\S+", str(text)))

    @staticmethod
    def _word_bytes(word: str) -> list[str]:
        return [f"{value:02x}" for value in (" " + word).encode("utf-8")]

    @staticmethod
    def _merge_sequence(
        sequence: Sequence[str], pair: tuple[str, str], merged: str
    ) -> list[str]:
        result: list[str] = []
        index = 0
        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and (sequence[index], sequence[index + 1]) == pair
            ):
                result.append(merged)
                index += 2
            else:
                result.append(sequence[index])
                index += 1
        return result

    def fit(
        self,
        texts: Iterable[str],
        *,
        progress_callback: Callable[["ByteBPE", int, int], None] | None = None,
        progress_every: int = 100,
    ) -> "ByteBPE":
        """Learn BPE merges using the original Assignment 1 algorithm.

        ``progress_callback`` is observational only: it runs after a completed
        merge and cannot affect pair counting, tie-breaking, or merge order.
        Its arguments are ``(tokenizer, selected_pair_count, merge_count)``.
        """

        if progress_every <= 0:
            raise ValueError("progress_every must be positive")

        word_counts = Counter(
            word for text in texts for word in re.findall(r"\S+", str(text))
        )
        words = sorted(word_counts)
        sequences = [self._word_bytes(word) for word in words]
        frequencies = np.asarray(
            [word_counts[word] for word in words], dtype=np.int64
        )
        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_words: dict[tuple[str, str], set[int]] = defaultdict(set)

        for word_id, sequence in enumerate(sequences):
            local = Counter(zip(sequence, sequence[1:]))
            for pair, occurrences in local.items():
                pair_counts[pair] += int(frequencies[word_id]) * occurrences
                pair_words[pair].add(word_id)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        while self.vocab_size < self.target_vocab_size and heap:
            negative_count, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if -negative_count != count:
                continue
            if count <= 1:
                break

            merged = pair[0] + pair[1]
            self.merges.append([pair[0], pair[1]])
            if merged not in self.stoi:
                self.stoi[merged] = len(self.itos)
                self.itos.append(merged)

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
                        heapq.heappush(heap, (-pair_counts[changed], changed))

            if progress_callback is not None and (
                len(self.merges) % progress_every == 0
                or self.vocab_size >= self.target_vocab_size
            ):
                progress_callback(self, count, len(self.merges))

        self._ranks = {
            tuple(pair): rank for rank, pair in enumerate(self.merges)
        }
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPE":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        model = cls(payload["vocab_size"])
        model.merges = payload["merges"]
        model.itos = payload["itos"]
        model.stoi = {token: index for index, token in enumerate(model.itos)}
        model._ranks = {
            tuple(pair): rank for rank, pair in enumerate(model.merges)
        }
        if model.vocab_size != payload["vocab_size"]:
            raise ValueError("Tokenizer vocabulary length does not match vocab_size")
        return model

    def save(self, path: str | Path) -> None:
        payload = {
            "merges": self.merges,
            "itos": self.itos,
            "vocab_size": self.vocab_size,
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    def encode_word(self, word: str) -> list[str]:
        sequence = self._word_bytes(word)
        while len(sequence) > 1:
            candidates = [
                (self._ranks[pair], pair)
                for pair in zip(sequence, sequence[1:])
                if pair in self._ranks
            ]
            if not candidates:
                break
            _, best = min(candidates)
            sequence = self._merge_sequence(
                sequence, best, best[0] + best[1]
            )
        return sequence

    def encode(self, text: str) -> list[str]:
        """Return the BPE token strings produced by the A1 tokenizer."""

        return [
            piece
            for word in re.findall(r"\S+", str(text))
            for piece in self.encode_word(word)
        ]

    def encode_ids(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Encode text as vocabulary IDs, optionally adding boundary tokens."""

        pieces = self.encode(text)
        ids = [self.stoi.get(piece, self.unk_token_id) for piece in pieces]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(
        self, ids: Sequence[int], *, skip_special_tokens: bool = True
    ) -> str:
        """Decode IDs to the A1 tokenizer's canonical whitespace form."""

        byte_stream = bytearray()
        rendered_specials: list[str] = []

        for token_id in ids:
            token_id = int(token_id)
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            token = self.itos[token_id]
            if token in self.RESERVED:
                if not skip_special_tokens:
                    rendered_specials.append(token)
                continue
            try:
                byte_stream.extend(bytes.fromhex(token))
            except ValueError as exc:
                raise ValueError(f"Non-hex BPE token in vocabulary: {token!r}") from exc

        text = byte_stream.decode("utf-8", errors="strict").lstrip(" ")
        if rendered_specials:
            return " ".join(rendered_specials + ([text] if text else []))
        return text
