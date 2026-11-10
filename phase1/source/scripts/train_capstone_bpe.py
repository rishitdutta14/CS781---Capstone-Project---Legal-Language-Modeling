#!/usr/bin/env python3
"""Train the Assignment-1 ByteBPE on every Phase-1 training document.

The BPE algorithm lives in ``src.tokenization.a1_byte_bpe``. This script only
adds streaming JSON ingestion, progress/checkpoint files, and exports nested
vocabulary snapshots from one deterministic merge chain.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenization.a1_byte_bpe import ByteBPE


DEFAULT_TRAIN_FILES = (
    ROOT / "data/raw/Phase_1_Pretraining/Train/train_acts.json",
    ROOT / "data/raw/Phase_1_Pretraining/Train/train_high_courts.json",
    ROOT / "data/raw/Phase_1_Pretraining/Train/train_supreme_court.json",
    ROOT / "data/raw/Phase_1_Pretraining/Train/train_tribunals.json",
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def iter_json_array(path: Path, chunk_size: int = 16 * 1024 * 1024) -> Iterator[dict]:
    """Incrementally decode a top-level JSON array using only the stdlib."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        refill()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected a top-level JSON array in {path}")
        position += 1

        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise ValueError(f"Unexpected end of JSON array in {path}")
                refill()

            if buffer[position] == "]":
                return

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    if not isinstance(value, dict):
                        raise ValueError(f"Expected JSON objects in {path}")
                    yield value
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Invalid or truncated JSON in {path}: {exc}") from exc
                    refill()

            if position >= chunk_size:
                refill()


def iter_training_texts(paths: list[Path], stats: dict) -> Iterator[str]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_start = time.time()
        documents = 0
        nonempty = 0
        log(f"Scanning {path} ({path.stat().st_size / 2**30:.2f} GiB)")
        for record in iter_json_array(path):
            documents += 1
            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Record {documents} in {path} has no string text field")
            if text.strip():
                nonempty += 1
                yield text
            if documents % 100_000 == 0:
                log(f"{path.name}: streamed {documents:,} documents")
        elapsed = time.time() - source_start
        stats["sources"][path.name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "documents": documents,
            "nonempty_documents": nonempty,
            "scan_seconds": elapsed,
        }
        log(f"Finished {path.name}: {documents:,} documents in {elapsed / 60:.1f} min")


def atomic_save(tokenizer: ByteBPE, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    tokenizer.save(temporary)
    os.replace(temporary, path)


def snapshot_from_merge_chain(tokenizer: ByteBPE, target_size: int) -> ByteBPE:
    """Rebuild the exact prefix whose vocabulary first reaches target_size."""

    snapshot = ByteBPE(target_vocab_size=target_size)
    for left, right in tokenizer.merges:
        merged = left + right
        snapshot.merges.append([left, right])
        if merged not in snapshot.stoi:
            snapshot.stoi[merged] = len(snapshot.itos)
            snapshot.itos.append(merged)
        if snapshot.vocab_size >= target_size:
            break
    if snapshot.vocab_size != target_size:
        raise RuntimeError(
            f"Merge chain reached {snapshot.vocab_size}, not requested {target_size}"
        )
    snapshot._ranks = {
        tuple(pair): rank for rank, pair in enumerate(snapshot.merges)
    }
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-files", nargs="+", type=Path, default=list(DEFAULT_TRAIN_FILES)
    )
    parser.add_argument("--target-vocab-size", type=int, default=32_000)
    parser.add_argument("--snapshot-sizes", nargs="+", type=int, default=[5_000, 16_000, 32_000])
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=ROOT / "tokenizers"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/tokenizer_training/legal_bpe_32k",
    )
    parser.add_argument("--checkpoint-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if max(args.snapshot_sizes) != args.target_vocab_size:
        raise ValueError("Largest snapshot size must equal --target-vocab-size")
    if min(args.snapshot_sizes) < 260:
        raise ValueError("Every snapshot must include four specials and 256 bytes")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "checkpoint_latest.json"
    stats_path = args.run_dir / "training_metadata.json"
    stats = {
        "algorithm": "Assignment-1 ByteBPE",
        "target_vocab_size": args.target_vocab_size,
        "snapshot_sizes": sorted(set(args.snapshot_sizes)),
        "sources": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "test_data_used": False,
    }
    started = time.time()

    def checkpoint(model: ByteBPE, pair_count: int, merge_count: int) -> None:
        atomic_save(model, checkpoint_path)
        elapsed = time.time() - started
        log(
            f"checkpoint: vocab={model.vocab_size:,}, merges={merge_count:,}, "
            f"selected_pair_count={pair_count:,}, elapsed={elapsed / 3600:.2f} h"
        )

    tokenizer = ByteBPE(target_vocab_size=args.target_vocab_size)
    log("Starting full-corpus A1 ByteBPE training")
    tokenizer.fit(
        iter_training_texts(args.train_files, stats),
        progress_callback=checkpoint,
        progress_every=args.checkpoint_every,
    )
    if tokenizer.vocab_size != args.target_vocab_size:
        raise RuntimeError(
            f"Training stopped at vocab {tokenizer.vocab_size:,}; expected {args.target_vocab_size:,}"
        )

    exported = {}
    for size in sorted(set(args.snapshot_sizes)):
        snapshot = snapshot_from_merge_chain(tokenizer, size)
        destination = args.tokenizer_dir / f"capstone_legal_bpe_{size // 1000}k.json"
        atomic_save(snapshot, destination)
        exported[str(size)] = str(destination.resolve())
        log(f"Exported {size:,}-token vocabulary to {destination}")

    stats.update(
        {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": time.time() - started,
            "final_vocab_size": tokenizer.vocab_size,
            "merge_count": len(tokenizer.merges),
            "exports": exported,
        }
    )
    temporary_stats = stats_path.with_suffix(".json.tmp")
    temporary_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    os.replace(temporary_stats, stats_path)
    log(f"Completed in {stats['elapsed_seconds'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
