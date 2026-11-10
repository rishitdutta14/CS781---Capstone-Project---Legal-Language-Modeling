"""Original-character offsets for unchanged Assignment 1 byte-BPE encoding.

Offsets use Python Unicode character indices, not UTF-8 byte indices. Multiple
byte tokens may cover the same character. Synthetic word-initial space tokens
have zero-width offsets when they contain no original character. Consequently,
offsets are not a promise that every gold boundary is a token boundary.
"""

import bisect
import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class OffsetEncoding:
    ids: list[int]
    offsets: list[tuple[int, int]]

    def covering_tokens(self, start: int, end: int) -> tuple[int, int]:
        """Smallest token interval covering characters overlapping [start,end).

        This can include characters outside the requested span. Callers must
        test exact endpoint alignment rather than silently treating it as exact.
        """
        if not 0 <= start < end:
            raise ValueError("require 0 <= start < end")
        left = bisect.bisect_right([b for _, b in self.offsets], start)
        right = bisect.bisect_left([a for a, _ in self.offsets], end)
        return left, max(left, right)


class A1OffsetTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._pieces = lru_cache(maxsize=100_000)(tokenizer.encode_word)

    def encode(self, text: str) -> OffsetEncoding:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids, offsets = [], []
        for word in re.finditer(r"\S+", text):
            # Byte 0 is the synthetic space inserted by A1, not a raw-text byte.
            boundaries = [1]
            for character in word.group():
                boundaries.append(boundaries[-1] + len(character.encode("utf-8")))
            cursor = 0
            pieces = self._pieces(word.group())
            for piece in pieces:
                end = cursor + len(bytes.fromhex(piece))
                begin_real, end_real = max(1, cursor), min(end, boundaries[-1])
                if end_real <= begin_real:
                    offset = (word.start(), word.start())
                else:
                    first = bisect.bisect_right(boundaries, begin_real) - 1
                    stop = bisect.bisect_left(boundaries, end_real)
                    offset = (word.start() + first, word.start() + stop)
                ids.append(self.tokenizer.stoi[piece])
                offsets.append(offset)
                cursor = end
            if cursor != boundaries[-1]:
                raise RuntimeError("BPE pieces did not account for every word byte")
        return OffsetEncoding(ids, offsets)
