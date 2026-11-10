"""Complete document encoding with frozen A1 semantics and independent replay."""

import hashlib
import json
import os
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.tokenization.a1_byte_bpe import ByteBPE

_ENCODER = None
_REFERENCE = None


class CachedDocumentEncoder:
    def __init__(self, tokenizer_path):
        self.tokenizer = ByteBPE.load(tokenizer_path)
        self.word_ids = lru_cache(maxsize=500000)(self._word_ids)
        self.token_bytes = tuple(b"" if i < 4 else bytes.fromhex(token) for i, token in enumerate(self.tokenizer.itos))

    def _word_ids(self, word):
        # Original A1 merge/encoding routine, memoized without data selection.
        return tuple(self.tokenizer.stoi[piece] for piece in self.tokenizer.encode_word(word))

    def encode(self, text):
        values = []
        for word in re.findall(r"\S+", str(text)):
            values.extend(self.word_ids(word))
        return values

    def decode(self, ids):
        return b"".join(self.token_bytes[value] for value in ids).decode("utf-8").lstrip(" ")


class CachedReferenceBPE(ByteBPE):
    @lru_cache(maxsize=500000)
    def encode_word(self, word):
        return tuple(super().encode_word(word))


def initialize_encoder(tokenizer_path):
    global _ENCODER
    _ENCODER = CachedDocumentEncoder(tokenizer_path)


def initialize_reference(tokenizer_path):
    global _REFERENCE
    # encode_ids/decode are the unchanged reference methods. Only the pure
    # per-word function is memoized; caches never affect vocabulary or coverage.
    _REFERENCE = CachedReferenceBPE.load(tokenizer_path)


def quantiles(histogram):
    count = sum(histogram.values())
    if not count:
        return {"documents": 0, "min": None, "max": None, "mean": None, "median": None, "p95": None, "p99": None}
    values = sorted(histogram)
    result = {"documents": count, "min": values[0], "max": values[-1], "mean": sum(value * histogram[value] for value in values) / count}
    for name, numerator in (("median", 50), ("p95", 95), ("p99", 99)):
        target = (count * numerator + 99) // 100
        cumulative = 0
        for value in values:
            cumulative += histogram[value]
            if cumulative >= target:
                result[name] = value
                break
    return result


def encode_membership_job(job, output_dir, bindings):
    if _ENCODER is None:
        raise RuntimeError("Worker encoder was not initialized")
    codec = _ENCODER
    tokenizer = codec.tokenizer
    split, part = job["split"], job["partition"]
    name = f"{split}_{part['partition']['name']}"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{name}.json"
    binary_path, index_path = output / f"{name}.bin", output / f"{name}.docs.jsonl"
    if manifest_path.exists() or any(output.glob(f"{name}.bin*")) or any(output.glob(f"{name}.docs.jsonl*")):
        raise FileExistsError(f"Existing encoded shard must be preserved: {name}")
    membership = part["outputs"][split]
    if file_sha256(Path(membership["path"])) != membership["sha256"]:
        raise ValueError("Frozen membership changed")
    counts, histogram, groups = Counter(), Counter(), {}
    started = time.monotonic()
    candidates = Path(part["partition"]["input_file"])
    with Path(membership["path"]).open(encoding="utf-8") as references, candidates.open("rb") as texts, \
         binary_path.with_suffix(".bin.partial").open("wb") as binary, index_path.with_suffix(".jsonl.partial").open("w", encoding="utf-8") as index:
        for line in references:
            ref = json.loads(line)
            if ref["split"] != split or ref["candidate_file"] != str(candidates):
                raise ValueError("Reference split/path mismatch")
            texts.seek(ref["byte_offset"])
            raw = texts.readline()
            if len(raw) != ref["byte_length"]:
                raise ValueError("Source record byte length mismatch")
            document = json.loads(raw)
            text = document["text"]
            if (document["document_id"], document["source"], document["input_index"]) != (ref["document_id"], ref["source"], ref["input_index"]):
                raise ValueError("Reference identity/source mismatch")
            if hashlib.sha256(text.casefold().encode()).hexdigest() != ref["normalized_sha256"] or text != ByteBPE.normalize(text):
                raise ValueError("Frozen text normalization/hash mismatch")
            ids = codec.encode(text)
            if not ids or min(ids) < 4 or max(ids) >= tokenizer.vocab_size:
                raise ValueError("Invalid or unknown content token")
            if codec.decode(ids) != text:
                raise ValueError(f"Full document round trip failed: {ref['document_id']}")
            content_length = len(ids)
            histogram[content_length] += 1
            ids.append(tokenizer.eos_token_id)
            np.asarray(ids, dtype="<u2").tofile(binary)
            record = {
                "id": ref["document_id"], "source": ref["source"], "input_index": ref["input_index"],
                "split": split, "offset": counts["stored_token_positions"], "length": len(ids),
                "corpus_length": content_length, "appended_eos": 1,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "normalized_sha256": ref["normalized_sha256"],
                "words": ref["words"], "characters": ref["characters"], "utf8_bytes": ref["utf8_bytes"],
                "script_group": ref["script_group"], "decision_id": ref["decision_id"],
                "candidate_file": ref["candidate_file"], "candidate_byte_offset": ref["byte_offset"],
            }
            index.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["documents"] += 1
            counts["corpus_token_positions"] += content_length
            counts["appended_eos_positions"] += 1
            counts["stored_token_positions"] += len(ids)
            counts["words"] += ref["words"]
            counts["characters"] += ref["characters"]
            counts["utf8_bytes"] += ref["utf8_bytes"]
            counts["documents_above_1600_content_tokens"] += content_length > 1600
            group = groups.setdefault(ref["script_group"], Counter())
            group.update({"documents": 1, "corpus_token_positions": content_length, "words": ref["words"], "characters": ref["characters"], "utf8_bytes": ref["utf8_bytes"]})
            if counts["documents"] % 10000 == 0:
                atomic_json(output / f"{name}.progress.json", {"status": "encoding", "counts": dict(counts), "elapsed_seconds": time.monotonic() - started})
        for suffix, metric in (("documents", "documents"), ("words", "words"), ("characters", "characters"), ("utf8_bytes", "utf8_bytes")):
            if counts[metric] != part["counts"].get(f"{split}_{suffix}", 0):
                raise ValueError(f"Complete shard coverage mismatch: {name}/{suffix}")
        binary.flush()
        os.fsync(binary.fileno())
        index.flush()
        os.fsync(index.fileno())
    binary_path.with_suffix(".bin.partial").replace(binary_path)
    index_path.with_suffix(".jsonl.partial").replace(index_path)
    if binary_path.stat().st_size != counts["stored_token_positions"] * 2:
        raise ValueError("Binary token count/size mismatch")
    result = {
        "status": "encoded_pending_independent_reference_replay", "name": name,
        "split": split, "source": part["partition"]["source"], "membership": membership,
        "counts": dict(counts), "token_length_histogram": dict(histogram),
        "token_length_statistics": quantiles(histogram), "script_group_counts": {key: dict(value) for key, value in groups.items()},
        "binary": {"path": str(binary_path), "bytes": binary_path.stat().st_size, "sha256": file_sha256(binary_path)},
        "index": {"path": str(index_path), "bytes": index_path.stat().st_size, "sha256": file_sha256(index_path)},
        "bindings": bindings, "dtype": "little-endian uint16", "unknown_content_ids": 0,
        "truncation": False, "sampling": False, "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(manifest_path, result)
    atomic_json(output / f"{name}.progress.json", result)
    return result


def audit_encoded_job(result):
    if _REFERENCE is None:
        raise RuntimeError("Reference worker was not initialized")
    tokenizer = _REFERENCE
    for item in (result["binary"], result["index"], result["membership"]):
        if file_sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Encoded/index/reference artifact changed")
    counts = Counter()
    current_candidate, texts = None, None
    try:
        with Path(result["membership"]["path"]).open(encoding="utf-8") as references, \
             Path(result["index"]["path"]).open(encoding="utf-8") as index, \
             Path(result["binary"]["path"]).open("rb") as binary:
            for line in references:
                ref = json.loads(line)
                encoded_line = index.readline()
                if not encoded_line:
                    raise ValueError("Missing document index")
                record = json.loads(encoded_line)
                if any(type(record[key]) is not int for key in ("offset", "length", "corpus_length", "appended_eos")):
                    raise ValueError("Document interval metadata must use integers")
                if record["offset"] != counts["stored_token_positions"] or record["length"] != record["corpus_length"] + 1 or record["appended_eos"] != 1:
                    raise ValueError("Document boundaries are not exact contiguous intervals")
                if (record["id"], record["source"], record["input_index"], record["split"], record["normalized_sha256"], record["decision_id"]) != (ref["document_id"], ref["source"], ref["input_index"], ref["split"], ref["normalized_sha256"], ref["decision_id"]):
                    raise ValueError("Encoded membership/identity mismatch")
                if (record["candidate_file"], record["candidate_byte_offset"], record["words"], record["characters"], record["utf8_bytes"], record["script_group"]) != (ref["candidate_file"], ref["byte_offset"], ref["words"], ref["characters"], ref["utf8_bytes"], ref["script_group"]):
                    raise ValueError("Encoded provenance/text metadata mismatch")
                if ref["candidate_file"] != current_candidate:
                    if texts is not None:
                        texts.close()
                    current_candidate = ref["candidate_file"]
                    texts = Path(current_candidate).open("rb")
                texts.seek(ref["byte_offset"])
                document = json.loads(texts.readline())
                text = document["text"]
                if document["document_id"] != ref["document_id"] or hashlib.sha256(text.encode()).hexdigest() != record["text_sha256"]:
                    raise ValueError("Full original text identity/hash mismatch")
                raw = binary.read(record["length"] * 2)
                if len(raw) != record["length"] * 2:
                    raise ValueError("Truncated token interval")
                values = np.frombuffer(raw, dtype="<u2")
                if len(values) < 2 or values[-1] != tokenizer.eos_token_id or int(values[:-1].min()) < 4 or int(values[:-1].max()) >= tokenizer.vocab_size:
                    raise ValueError("Invalid special/unknown/out-of-range content ID")
                expected = tokenizer.encode_ids(text, add_eos=True)
                if values.tolist() != expected:
                    raise ValueError("Stored IDs differ from original A1 encode_ids")
                if tokenizer.decode(values) != text:
                    raise ValueError("Original A1 decode fails exact complete-document round trip")
                if len(re.findall(r"\S+", text)) != ref["words"] or len(text) != ref["characters"] or len(text.encode()) != ref["utf8_bytes"]:
                    raise ValueError("Independent raw text counts differ")
                counts.update({"documents": 1, "corpus_token_positions": len(values) - 1,
                               "appended_eos_positions": 1, "stored_token_positions": len(values),
                               "words": ref["words"], "characters": ref["characters"], "utf8_bytes": ref["utf8_bytes"]})
            if index.readline() or binary.read(1):
                raise ValueError("Unindexed trailing tokens or extra document")
        for key, value in counts.items():
            if value != result["counts"].get(key, 0):
                raise ValueError(f"Independent shard count mismatch: {key}")
        if counts["documents"] != result["counts"].get("documents", 0):
            raise ValueError("Document count mismatch")
        return {"name": result["name"], "passed": True, "counts": dict(counts),
                "every_document_reencoded_with_original_a1": True,
                "every_document_decoded_with_original_a1": True, "unknown_content_ids": 0}
    finally:
        if texts is not None:
            texts.close()
