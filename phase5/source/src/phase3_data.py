"""RR examples remain sentence-indexed; only preceding same-document text is context."""

import json
from pathlib import Path

import numpy as np
import torch

from src.phase3_submission import file_sha256, strict_loads


class RRData:
    def __init__(self, manifest_path, label_path):
        self.manifest = strict_loads(Path(manifest_path).read_bytes())
        for name, expected in self.manifest["artifact_bindings"].items():
            if file_sha256(name) != expected:
                raise ValueError(f"Changed encoded-input binding: {name}")
        vocab = strict_loads(Path(label_path).read_bytes())
        self.labels = sorted(vocab, key=vocab.get)
        self.documents = {split: [] for split in ("train", "dev", "test")}
        self.rows = {split: [] for split in self.documents}
        for entry in self.manifest["documents"]:
            if file_sha256(entry["cache"]) != entry["sha256"]:
                raise ValueError("Changed token cache")
            source = strict_loads(Path(entry["source_file"]).read_bytes())
            if entry["split"] == "test":
                sentences, gold = source[entry["document_id"]], None
            else:
                matches = [doc for doc in source if doc["id"] == entry["document_id"]]
                if len(matches) != 1:
                    raise ValueError("Document identity changed")
                sentences, gold = matches[0]["text"], matches[0]["labels"]
            with np.load(entry["cache"], allow_pickle=False) as saved:
                ids, offsets = saved["ids"].copy(), saved["offsets"].copy()
            if (len(offsets) != len(sentences) + 1 or offsets[0] != 0
                    or offsets[-1] != len(ids) or not np.all(np.diff(offsets) > 0)):
                raise ValueError("Invalid sentence offsets")
            doc = {**entry, "ids": ids, "offsets": offsets, "sentences": sentences}
            self.documents[entry["split"]].append(doc)
            for index in range(len(sentences)):
                self.rows[entry["split"]].append({"doc": doc, "sentence_index": index,
                    "key": (entry["document_id"], index),
                    "label": vocab[gold[index]] if gold is not None else None})
        if {split: len(rows) for split, rows in self.rows.items()} != {"train": 16845, "dev": 2142, "test": 2197}:
            raise ValueError("Sentence coverage differs from the official splits")


def sentence_chunks(row, context_tokens=256, capacity=1600):
    doc, index = row["doc"], row["sentence_index"]
    ids, offsets = doc["ids"], doc["offsets"]
    a, b = map(int, offsets[index:index + 2])
    target = ids[a:b].tolist()
    if not target or not 0 <= context_tokens or capacity < 2:
        raise ValueError("Invalid target or input policy")
    # Existing EOS token separates preceding sentences. Never cross documents.
    preceding = []
    previous = index - 1
    while previous >= 0 and len(preceding) < context_tokens:
        start, end = map(int, offsets[previous:previous + 2])
        preceding = ids[start:end].tolist() + [3] + preceding
        previous -= 1
    preceding = preceding[-context_tokens:] if context_tokens else []
    chunks = []
    for start in range(0, len(target), capacity - 1):
        piece = target[start:start + capacity - 1]
        # In an overlong sentence, earlier target tokens can supply context too.
        before = preceding + target[:start]
        room = min(context_tokens, capacity - 1 - len(piece))
        context = before[-room:] if room else []
        tokens = [2] + context + piece
        mask = [False] * (1 + len(context)) + [True] * len(piece)
        chunks.append((tokens, mask))
    if sum(sum(mask) for _, mask in chunks) != len(target):
        raise ValueError("Chunk policy lost or duplicated target tokens")
    return chunks


def collate(rows, context_tokens=256, capacity=1600, device="cpu"):
    chunks, owners = [], []
    for owner, row in enumerate(rows):
        for chunk in sentence_chunks(row, context_tokens, capacity):
            chunks.append(chunk)
            owners.append(owner)
    length = max(len(ids) for ids, _ in chunks)
    ids = torch.zeros(len(chunks), length, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, (tokens, targets) in enumerate(chunks):
        ids[index, :len(tokens)] = torch.tensor(tokens)
        mask[index, :len(tokens)] = torch.tensor(targets)
    return {"ids": ids.to(device), "target_mask": mask.to(device),
            "owners": torch.tensor(owners, dtype=torch.long, device=device),
            "sentence_count": len(rows)}


def epoch_batches(size, batch_size, seed, epoch):
    order = np.random.default_rng(seed + epoch).permutation(size).tolist()
    return [order[start:start + batch_size] for start in range(0, size, batch_size)]


def predictions_from_rows(rows, indices, labels):
    if len(rows) != len(indices):
        raise ValueError("Missing sentence predictions")
    result, next_index = {}, {}
    for row, prediction in zip(rows, indices):
        doc_id, index = row["key"]
        if index != next_index.get(doc_id, 0):
            raise ValueError("Prediction sentence order was changed or duplicated")
        result.setdefault(doc_id, []).append(labels[prediction])
        next_index[doc_id] = index + 1
    return result
