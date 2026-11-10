"""Pinned CJPE data, complete A1 tokenization, and strict submission checks."""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np

from src.phase1_full_corpus_encode import CachedDocumentEncoder
from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.tokenization.a1_byte_bpe import ByteBPE

REPO = "Exploration-Lab/CS-781-Capstone"
REVISION = "93b0059be756c6c461f3a225de25aabda7d5f84a"
FILES = {
    "train": ("Phase_4_CJPE/Train/train.json", "a115cc722e9de8b68ac5d41179c12d86c2d537c0de8ff16412c44c6c8d1cce72"),
    "dev": ("Phase_4_CJPE/Train/dev.json", "aa2d80ae77f5d532fbc3f30e4dea2776700b8e92e71c603737d8f41a9d30b514"),
    "test": ("Phase_4_CJPE/Test/test.json", "f3498fb832e90fe7264fcb6770918eb5261409e9a952d47a2979b6ed5b153b59"),
}
_CODEC = None


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=pairs, parse_constant=invalid)


def validate_records(records, split):
    if not isinstance(records, list) or not records:
        raise ValueError(f"{split}: expected a nonempty list")
    seen = set()
    for row in records:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"].strip() or (split == "test" and row["id"] in seen)
                or not isinstance(row.get("text"), str)):
            raise ValueError(f"{split}: invalid record or duplicate ID")
        seen.add(row["id"])
        if split != "test" and (type(row.get("label")) is not int or row["label"] not in (0, 1)):
            raise ValueError(f"{split}: expected integer label0 or1")
    return records


def raw_audit(splits):
    hashes, ids = defaultdict(list), defaultdict(list)
    summary = {}
    for split, records in splits.items():
        validate_records(records, split)
        labels = Counter(row["label"] for row in records) if split != "test" else None
        lengths = [len(row["text"]) for row in records]
        for index, row in enumerate(records):
            # Never read test labels, expert annotations, or other metadata.
            entry = {"split": split, "index": index, "id": row["id"]}
            ids[row["id"]].append(entry)
            hashes[hashlib.sha256(row["text"].encode()).hexdigest()].append(entry)
        summary[split] = {"documents": len(records), "labels": dict(labels) if labels else None,
            "unique_case_ids": len({row['id'] for row in records}),
            "repeated_ID_rows_preserved": len(records)-len({row['id'] for row in records}),
            "empty_texts": sum(not row["text"].strip() for row in records),
            "characters": int(sum(lengths)), "max_characters": max(lengths),
            "p95_characters": float(np.quantile(lengths, .95))}
    overlaps = {"cross_split_ids": [entries for entries in ids.values()
        if len({r["split"] for r in entries}) > 1],
        "cross_split_exact_texts": [entries for entries in hashes.values()
        if len({r["split"] for r in entries}) > 1]}
    return {"splits": summary, "overlaps": overlaps,
        "train_dev_or_test_resplit": False, "cleaning": False,
        "empty_text_policy": "Retain as a single BOS-only chunk; report counts",
        "duplicate_policy": "Retain every official train/dev row; stable internal keys use split and row index, not a dictionary keyed only by case ID",
        "test_labels_and_expert_fields_used": False}


def download(raw):
    from huggingface_hub import hf_hub_download
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    def one(item):
        split, (name, digest) = item
        path = raw / name
        if path.exists() and file_sha256(path) != digest:
            raise ValueError(f"Do not overwrite different existing data: {path}")
        path = Path(hf_hub_download(REPO, name, repo_type="dataset",
            revision=REVISION, local_dir=raw))
        if file_sha256(path) != digest:
            raise ValueError("Downloaded bytes differ from the pinned official input")
        return split, {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(pool.map(one, FILES.items()))


def initialize_encoder(tokenizer):
    global _CODEC
    _CODEC = CachedDocumentEncoder(tokenizer)


def encode_batch(rows):
    result = []
    for row in rows:
        ids = _CODEC.encode(row["text"])
        if ids and (min(ids) < 4 or max(ids) >= _CODEC.tokenizer.vocab_size):
            raise ValueError("Unknown or special content token")
        if _CODEC.decode(ids) != ByteBPE.normalize(row["text"]):
            raise ValueError("A1 canonical-whitespace round trip failed")
        result.append(np.asarray(ids, dtype="<u2"))
    return result


def chunk_slices(length, capacity=1600):
    if length < 0 or capacity < 2:
        raise ValueError("Invalid chunk configuration")
    # One BOS in every independent causal chunk; no overlap or token cap.
    if length == 0:
        return [(0, 0)]
    return [(start, min(length, start + capacity - 1))
            for start in range(0, length, capacity - 1)]


def prepare_encoded(raw_files, tokenizer, output, workers=12):
    output = Path(output)
    output.mkdir(exist_ok=False)
    splits = {name: validate_records(strict_json(row["path"]), name)
              for name, row in raw_files.items()}
    audit = raw_audit(splits)
    atomic_json(output / "raw_audit.json", audit)
    manifest = {"repo_id": REPO, "revision": REVISION, "raw_files": raw_files,
        "tokenizer": str(tokenizer), "tokenizer_sha256": file_sha256(Path(tokenizer)),
        "capacity": 1600, "content_tokens_per_chunk": 1599,
        "pooling": "last valid token per chunk, equal arithmetic mean across ALL chunks",
        "input_fields": ["text"], "cleaning": False,
        "tokenizer_whitespace_semantics": "Original A1 per-word leading space; canonical whitespace on decode; no raw file edits",
        "sampling": False, "truncation": False, "splits": {}}
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_encoder,
                             initargs=(tokenizer,)) as pool:
        for split, records in splits.items():
            binary = output / (split + ".bin")
            index, offset = [], 0
            batches = [records[start:start + 32] for start in range(0, len(records), 32)]
            with binary.open("xb") as stream:
                for number, encoded in enumerate(pool.map(encode_batch, batches, chunksize=1)):
                    for values in encoded:
                        row = records[len(index)]
                        values.tofile(stream)
                        entry = {"id": row["id"], "row_key": f"{split}:{len(index)}",
                            "offset": offset, "tokens": len(values),
                            "chunks": len(chunk_slices(len(values))),
                            "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()}
                        if split != "test":
                            entry["label"] = row["label"]
                        index.append(entry)
                        offset += len(values)
                    if number % 20 == 0:
                        atomic_json(output / "progress.json", {"status": "CPU_complete_A1_encoding",
                            "split": split, "documents_done": len(index), "documents_total": len(records),
                            "GPU_training_started": False})
                stream.flush()
                os.fsync(stream.fileno())
            if len(index) != len(records) or binary.stat().st_size != offset * 2:
                raise ValueError("Full input coverage or byte accounting failed")
            index_path = output / (split + "_index.json")
            atomic_json(index_path, index)
            lengths = [row["tokens"] for row in index]
            chunks = [row["chunks"] for row in index]
            manifest["splits"][split] = {"binary": str(binary), "binary_sha256": file_sha256(binary),
                "index": str(index_path), "index_sha256": file_sha256(index_path),
                "documents": len(index), "tokens": offset, "chunks": sum(chunks),
                "max_tokens": max(lengths), "max_chunks": max(chunks),
                "p50_tokens": float(np.quantile(lengths, .5)), "p95_tokens": float(np.quantile(lengths, .95)),
                "all_documents_round_trip_checked": True, "unknown_content_tokens": 0}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "completed.json", {"complete": True,
        "manifest_sha256": file_sha256(output / "manifest.json"), "training_started": False})
    return manifest, audit


class CaseData:
    def __init__(self, manifest_path, verify=True):
        self.manifest = strict_json(manifest_path)
        self.rows, self.tokens = {}, {}
        for split, entry in self.manifest["splits"].items():
            if verify and (file_sha256(Path(entry["binary"])) != entry["binary_sha256"]
                    or file_sha256(Path(entry["index"])) != entry["index_sha256"]):
                raise ValueError("Encoded input changed")
            self.rows[split] = strict_json(entry["index"])
            self.tokens[split] = np.memmap(entry["binary"], mode="r", dtype="<u2")

    def chunks(self, split, index):
        row = self.rows[split][index]
        offset, length = row["offset"], row["tokens"]
        values = self.tokens[split][offset:offset + length]
        return [[2] + values[a:b].astype(np.int64).tolist()
                for a, b in chunk_slices(length, self.manifest["capacity"])]


def validate_predictions(predictions, expected_ids):
    if not isinstance(predictions, list) or len(predictions) != len(expected_ids):
        raise ValueError("Expected exactly one record for each test ID")
    seen = set()
    for row in predictions:
        if (not isinstance(row, dict) or set(row) != {"id", "pred"}
                or not isinstance(row["id"], str) or row["id"] in seen
                or type(row["pred"]) is not int or row["pred"] not in (0, 1)):
            raise ValueError("Invalid ID or non-integer binary prediction")
        seen.add(row["id"])
    if seen != set(expected_ids) or len(seen) != len(expected_ids):
        raise ValueError("Prediction ID set differs from official test IDs")


def package_submission(path, predictions, expected_ids):
    validate_predictions(predictions, expected_ids)
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve existing submission ZIP")
    payload = json.dumps(predictions, ensure_ascii=False, allow_nan=False).encode()
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", payload)
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["prediction.json"] or archive.testzip() is not None:
            raise ValueError("Submission archive structure/CRC failed")
        validate_predictions(json.loads(archive.read("prediction.json")), expected_ids)
    return {"path": str(path), "sha256": file_sha256(path), "predictions": len(predictions)}
