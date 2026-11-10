"""Complete prefix/continuation windows; no quotas, dropped tails or packing."""

from bisect import bisect_right
from pathlib import Path

import numpy as np

SOURCE_LIMIT = 800
TARGET_LIMIT = 800
BOS, EOS, PAD, IGNORE = 2, 3, 0, -100


def windows(length, target_limit=TARGET_LIMIT, source_limit=SOURCE_LIMIT):
    if length < 2 or target_limit < 1 or source_limit < 1:
        raise ValueError("Need nonempty content plus EOS and positive limits")
    for start in range(0, length, target_limit):
        end = min(start + target_limit, length)
        yield max(0, start - source_limit), start, end


def example(tokens, start, end, source_limit=SOURCE_LIMIT):
    if not 0 <= start < end <= len(tokens) or end - start > TARGET_LIMIT:
        raise ValueError("Invalid target interval")
    source = tokens[max(0, start - source_limit):start].astype(np.int64)
    if not len(source):
        source = np.asarray([BOS], dtype=np.int64)
    labels = tokens[start:end].astype(np.int64)
    decoder = np.concatenate(([BOS], labels[:-1]))
    if len(source) + len(decoder) > SOURCE_LIMIT + TARGET_LIMIT:
        raise ValueError("Combined input budget exceeded")
    return source, decoder, labels


class CompletePassDataset:
    """Read-only mmap of the audited plan and original uint16 token files."""
    def __init__(self, manifest, split):
        self.parts = [part for part in manifest["parts"] if part["split"] == split]
        self.ends = np.cumsum([part["samples"] for part in self.parts]).tolist()
        self.arrays = {}

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        local = index - (self.ends[shard - 1] if shard else 0)
        part = self.parts[shard]
        if shard not in self.arrays:
            self.arrays[shard] = (
                np.load(part["sample_file"]["path"], mmap_mode="r"),
                np.load(part["document_file"]["path"], mmap_mode="r"),
                np.memmap(part["binary"]["path"], dtype="<u2", mode="r"))
        samples, documents, tokens = self.arrays[shard]
        document, start, length = map(int, samples[local])
        offset, doc_length, input_index, first_sample = map(int, documents[document])
        return part, tokens[offset:offset + doc_length], start, start + length, input_index

    def __getitem__(self, index):
        part, tokens, start, end, input_index = self.locate(index)
        source, decoder, labels = example(tokens, start, end)
        return {"source": source, "decoder": decoder, "labels": labels,
                "document_id": f"{part['source']}:{input_index}", "target_start": start}


def collate(examples):
    import torch
    if not examples:
        # A distributed tail rank still participates with zero valid labels.
        examples = [{"source": np.asarray([BOS]), "decoder": np.asarray([BOS]),
                     "labels": np.asarray([IGNORE])}]
    source = torch.full((len(examples), max(len(x["source"]) for x in examples)), PAD, dtype=torch.long)
    decoder = torch.full((len(examples), max(len(x["decoder"]) for x in examples)), PAD, dtype=torch.long)
    labels = torch.full_like(decoder, IGNORE)
    source_valid, target_valid = torch.zeros_like(source, dtype=torch.bool), torch.zeros_like(decoder, dtype=torch.bool)
    for i, item in enumerate(examples):
        s, t = len(item["source"]), len(item["decoder"])
        source[i, :s] = torch.as_tensor(item["source"])
        decoder[i, :t] = torch.as_tensor(item["decoder"])
        labels[i, :t] = torch.as_tensor(item["labels"])
        source_valid[i, :s], target_valid[i, :t] = True, True
    return source, decoder, labels, source_valid, target_valid
