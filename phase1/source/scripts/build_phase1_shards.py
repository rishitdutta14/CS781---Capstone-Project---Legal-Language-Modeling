#!/usr/bin/env python3
"""Build balanced, reusable token shards from the four Phase 1 train files."""

import argparse
import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data/raw/Phase_1_Pretraining/Train"
SOURCES = {
    "acts": TRAIN_DIR / "train_acts.json",
    "high_courts": TRAIN_DIR / "train_high_courts.json",
    "supreme_court": TRAIN_DIR / "train_supreme_court.json",
    "tribunals": TRAIN_DIR / "train_tribunals.json",
}

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_capstone_bpe import iter_json_array
from src.tokenization.a1_byte_bpe import ByteBPE


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "tokenizers/capstone_legal_bpe_16k.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/phase1_16k_100m",
    )
    parser.add_argument("--train-tokens-per-source", type=int, default=25_000_000)
    parser.add_argument("--val-tokens-per-source", type=int, default=1_000_000)
    parser.add_argument("--validation-percent", type=int, default=5)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenWriter:
    def __init__(self, path, limit):
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.limit = limit
        self.count = 0
        self.handle = self.temporary.open("wb")

    @property
    def full(self):
        return self.count >= self.limit

    def write(self, token_ids):
        remaining = self.limit - self.count
        values = np.asarray(token_ids[:remaining], dtype=np.uint16)
        values.tofile(self.handle)
        self.count += len(values)
        return len(values)

    def close(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if self.count != self.limit:
            raise RuntimeError(
                f"{self.path.name}: wrote {self.count:,}, expected {self.limit:,} tokens"
            )
        os.replace(self.temporary, self.path)


def split_for(source, index, validation_percent):
    key = f"{source}:{index}".encode()
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
    return "val" if bucket < validation_percent else "train"


def main():
    args = arguments()
    if not 0 < args.validation_percent < 100:
        raise ValueError("validation-percent must be between 1 and 99")
    if args.train_tokens_per_source <= 0 or args.val_tokens_per_source <= 0:
        raise ValueError("token quotas must be positive")
    if args.output_dir.exists() and (args.output_dir / "manifest.json").is_file():
        print(f"Shards already complete: {args.output_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteBPE.load(args.tokenizer)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("uint16 shards cannot hold this vocabulary")

    @lru_cache(maxsize=500_000)
    def encode_word(word):
        return tuple(tokenizer.stoi[piece] for piece in tokenizer.encode_word(word))

    def encode(text):
        ids = []
        for word in re.findall(r"\S+", text):
            ids.extend(encode_word(word))
        ids.append(tokenizer.eos_token_id)
        return ids

    started = time.time()
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256(args.tokenizer),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "split_rule": (
            f"sha256(source:index) modulo 100 < {args.validation_percent} is validation"
        ),
        "source_mixing": "equal token quota from each of the four official sources",
        "test_data_used": False,
        "sources": {},
    }
    split_records = args.output_dir / "split_documents.jsonl.tmp"
    seen = {"train": set(), "val": set()}

    with split_records.open("w", encoding="utf-8") as record_file:
        for source, path in SOURCES.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            writers = {
                "train": TokenWriter(
                    args.output_dir / f"train_{source}.bin",
                    args.train_tokens_per_source,
                ),
                "val": TokenWriter(
                    args.output_dir / f"val_{source}.bin",
                    args.val_tokens_per_source,
                ),
            }
            stats = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "records_scanned": 0,
                "train_documents": 0,
                "val_documents": 0,
                "train_tokens": 0,
                "val_tokens": 0,
                "truncated_documents": 0,
            }
            source_started = time.time()
            for index, row in enumerate(iter_json_array(path)):
                stats["records_scanned"] = index + 1
                split = split_for(source, index, args.validation_percent)
                writer = writers[split]
                if writer.full:
                    if all(item.full for item in writers.values()):
                        break
                    continue
                text = row.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{path.name} record {index} has no string text")
                if not text.strip():
                    continue

                ids = encode(text)
                written = writer.write(ids)
                document_id = f"{source}:{index}"
                if document_id in seen["train"] or document_id in seen["val"]:
                    raise RuntimeError(f"duplicate document ID: {document_id}")
                seen[split].add(document_id)
                stats[f"{split}_documents"] += 1
                stats[f"{split}_tokens"] += written
                truncated = written < len(ids)
                stats["truncated_documents"] += int(truncated)
                record_file.write(
                    json.dumps(
                        {
                            "id": document_id,
                            "split": split,
                            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "tokens_written": written,
                            "truncated": truncated,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            for writer in writers.values():
                writer.close()
            stats["elapsed_seconds"] = time.time() - source_started
            stats["train_file"] = f"train_{source}.bin"
            stats["val_file"] = f"val_{source}.bin"
            manifest["sources"][source] = stats
            print(
                f"{source}: {stats['train_tokens']:,} train + "
                f"{stats['val_tokens']:,} val tokens from "
                f"{stats['records_scanned']:,} records",
                flush=True,
            )

    if seen["train"] & seen["val"]:
        raise RuntimeError("train/validation document leakage detected")
    os.replace(split_records, args.output_dir / "split_documents.jsonl")
    manifest["train_documents"] = len(seen["train"])
    manifest["val_documents"] = len(seen["val"])
    manifest["document_overlap"] = 0
    manifest["train_tokens"] = sum(
        item["train_tokens"] for item in manifest["sources"].values()
    )
    manifest["val_tokens"] = sum(
        item["val_tokens"] for item in manifest["sources"].values()
    )
    manifest["elapsed_seconds"] = time.time() - started
    manifest["cache"] = str(encode_word.cache_info())

    for split in ("train", "val"):
        for source in SOURCES:
            name = f"{split}_{source}.bin"
            manifest["sources"][source][f"{split}_sha256"] = sha256(
                args.output_dir / name
            )
    manifest["split_documents_sha256"] = sha256(
        args.output_dir / "split_documents.jsonl"
    )
    temporary = args.output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "manifest.json")
    print(
        f"Complete: {manifest['train_tokens']:,} train and "
        f"{manifest['val_tokens']:,} validation tokens in "
        f"{manifest['elapsed_seconds'] / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
