"""Exact A1 word-frequency counting from frozen training references only."""

import hashlib
import json
import os
import pickle
import re
import resource
import time
from collections import Counter
from pathlib import Path

from src.phase1_full_corpus_prepare import atomic_json, file_sha256


def count_training_partition(partition, output_dir, bindings):
    name = partition["partition"]["name"]
    output = Path(output_dir)
    final = output / f"{name}.json"
    counts_path = output / f"{name}.pkl"
    if final.exists() or counts_path.exists() or counts_path.with_suffix(".pkl.partial").exists():
        raise FileExistsError(f"Existing count shard must be preserved: {name}")
    output.mkdir(parents=True, exist_ok=True)
    membership = partition["outputs"]["train"]
    reference_path = Path(membership["path"])
    if file_sha256(reference_path) != membership["sha256"]:
        raise ValueError("Frozen training membership changed")
    candidate_path = Path(partition["partition"]["input_file"])
    counts = Counter()
    documents, words_seen = 0, 0
    started = time.monotonic()
    with reference_path.open(encoding="utf-8") as references, candidate_path.open("rb") as texts:
        for line in references:
            ref = json.loads(line)
            if ref["split"] != "train" or ref["candidate_file"] != str(candidate_path):
                raise ValueError("Non-training or unexpected file reference")
            texts.seek(ref["byte_offset"])
            raw = texts.readline()
            if len(raw) != ref["byte_length"]:
                raise ValueError("Training record byte length mismatch")
            document = json.loads(raw)
            text = document["text"]
            digest = hashlib.sha256(text.casefold().encode()).hexdigest()
            if digest != ref["normalized_sha256"] or int(digest[:16], 16) / 2**64 < .05:
                raise ValueError("Validation membership or changed training text")
            if (document["document_id"], document["source"], document["input_index"]) != (ref["document_id"], ref["source"], ref["input_index"]):
                raise ValueError("Training identity/source mismatch")
            # Exactly the original A1 word extraction. No lowercasing, word
            # selection, frequency approximation or per-document truncation.
            words = re.findall(r"\S+", str(text))
            if len(words) != ref["words"]:
                raise ValueError("Training word count mismatch")
            counts.update(words)
            documents += 1
            words_seen += len(words)
            if documents % 10000 == 0:
                atomic_json(output / f"{name}.progress.json", {
                    "status": "counting", "partition": name, "train_documents": documents,
                    "word_occurrences": words_seen, "unique_words": len(counts),
                    "elapsed_seconds": time.monotonic() - started,
                })
    if documents != partition["counts"].get("train_documents", 0) or words_seen != partition["counts"].get("train_words", 0) or sum(counts.values()) != words_seen:
        raise ValueError("Incomplete training-frequency coverage")
    temporary = counts_path.with_suffix(".pkl.partial")
    with temporary.open("wb") as handle:
        pickle.dump(counts, handle, protocol=5)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(counts_path)
    result = {
        "status": "training_partition_counted", "partition": name,
        "source": partition["partition"]["source"], "train_documents": documents,
        "word_occurrences": words_seen, "unique_words": len(counts),
        "counts_file": str(counts_path), "counts_sha256": file_sha256(counts_path),
        "counts_bytes": counts_path.stat().st_size, "membership_sha256": membership["sha256"],
        "bindings": bindings, "elapsed_seconds": time.monotonic() - started,
        "worker_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "validation_documents_counted": 0, "sampling": False, "truncation": False,
        "word_extraction": "re.findall(r'\\S+', str(text)) as in A1",
    }
    atomic_json(final, result)
    atomic_json(output / f"{name}.progress.json", result)
    return result
