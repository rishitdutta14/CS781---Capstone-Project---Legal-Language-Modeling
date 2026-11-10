"""Ancillary CPU audit/cache of existing tokenizer IDs; no model/settings changes."""

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.setup_phase3_data import documents, strict_json
from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.tokenization.a1_byte_bpe import ByteBPE


def cached_encoder(tokenizer):
    # A1 merges are word-local. Cache exact original per-word IDs, not a new BPE.
    @lru_cache(maxsize=None)
    def word_ids(word):
        return tuple(tokenizer.stoi[piece] for piece in tokenizer.encode_word(word))
    def encode(text):
        return [identity for word in re.findall(r"\S+", text) for identity in word_ids(word)]
    return encode


def encode_document(sentences, tokenizer, encode):
    offsets, pieces, lengths = [0], [], []
    for sentence in sentences:
        ids = encode(sentence)
        if not ids or tokenizer.unk_token_id in ids or any(not 4 <= i < tokenizer.vocab_size for i in ids):
            raise ValueError("Empty encoding, unknown byte or unexpected special token")
        if tokenizer.decode(ids) != tokenizer.normalize(sentence):
            raise ValueError("A1 canonical-whitespace roundtrip failed")
        pieces.extend(ids)
        lengths.append(len(ids))
        offsets.append(len(pieces))
    return np.asarray(pieces, dtype=np.uint16), np.asarray(offsets, dtype=np.int64), lengths


def run(args):
    stage, report = args.stage_dir.resolve(), args.report_dir.resolve()
    stage.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    inspected = strict_json(args.input_preflight)
    bindings = inspected["data_files"].copy()
    tokenizer_record = inspected["backbones"]["mha"]
    if len({r["tokenizer_sha256"] for r in inspected["backbones"].values()}) != 1:
        raise ValueError("This cache cannot silently mix different tokenizer IDs")
    path = Path(tokenizer_record["tokenizer"])
    bindings[str(path)] = tokenizer_record["tokenizer_sha256"]
    bindings[str(args.input_preflight.resolve())] = file_sha256(args.input_preflight)
    for name, expected in bindings.items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Bound input changed: {name}")
    tokenizer = ByteBPE.load(path)
    encode = cached_encoder(tokenizer)
    labels = strict_json(ROOT/"data/raw/Phase_3_RR/Train/label_vocab.json")
    source_files = [Path(p) for p in inspected["data_files"] if Path(p).name != "label_vocab.json"]
    summary, index, all_lengths = {}, [], {}
    csv_path = stage/"sentence_lengths.csv"
    with csv_path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["domain", "split", "document_id", "sentence_index", "tokens", "characters", "whitespace_words"])
        for source in sorted(source_files):
            domain, split = source.stem.split("_")
            records = list(documents(strict_json(source), split != "test", set(labels)))
            collected = []
            for doc in records:
                ids, offsets, lengths = encode_document(doc["text"], tokenizer, encode)
                # Direct original encode_ids, not the cache, on every document's endpoints.
                for sentence_index in sorted({0, len(doc["text"])-1}):
                    a, b = offsets[sentence_index:sentence_index+2]
                    if tokenizer.encode_ids(doc["text"][sentence_index]) != ids[a:b].tolist():
                        raise ValueError("Cached and unmodified A1 encoding disagree")
                output = stage/f"document_{len(index):04d}.npz"
                np.savez_compressed(output, ids=ids, offsets=offsets)
                with np.load(output, allow_pickle=False) as restored:
                    if not np.array_equal(restored["ids"], ids) or not np.array_equal(restored["offsets"], offsets):
                        raise ValueError("Token cache roundtrip differs")
                index.append({"domain": domain, "split": split, "document_id": doc["id"],
                    "source_file": str(source), "sentence_count": len(lengths), "tokens": len(ids),
                    "cache": str(output), "sha256": file_sha256(output)})
                for number, (sentence, length) in enumerate(zip(doc["text"], lengths)):
                    writer.writerow([domain, split, doc["id"], number, length, len(sentence), len(sentence.split())])
                collected.extend(lengths)
                atomic_json(stage/"progress.json", {"status": "CPU_token_length_audit", "documents_done": len(index),
                    "last_document": doc["id"], "optimizer_updates": 0})
            all_lengths.setdefault(split, []).extend(collected)
            summary[f"{domain}_{split}"] = {"documents": len(records), "sentences": len(collected),
                "tokens": sum(collected), "maximum_sentence_tokens": max(collected),
                "p95_sentence_tokens": float(np.quantile(collected, .95)),
                "above1600_raw_tokens": sum(n > 1600 for n in collected),
                "above1599_tokens_if_one_BOS_reserved": sum(n > 1599 for n in collected)}
    expected = {"train": 16845, "dev": 2142, "test": 2197}
    if {split: len(values) for split, values in all_lengths.items()} != expected or len(index) != 100:
        raise ValueError("Token audit dropped or duplicated a document/sentence")
    result = {"passed": True, "CPU_only": True, "optimizer_updates": 0,
        "raw_text_edited": False, "tokenizer_refit_or_changed": False, "new_special_tokens": [],
        "roundtrip_scope": "All sentences decode to original A1 canonical whitespace, NOT byte-exact raw whitespace; raw files remain unchanged.",
        "unmodified_reference_encoding_endpoint_checks": 2*len(index),
        "files": summary, "sentence_totals": expected, "documents": index,
        "artifact_bindings": bindings, "sentence_length_csv": str(csv_path), "sentence_length_csv_sha256": file_sha256(csv_path),
        "setup_code_sha256": file_sha256(Path(__file__)),
        "tokenizer_code_sha256": file_sha256(ROOT/"src/tokenization/a1_byte_bpe.py"),
        "elapsed_seconds": time.perf_counter()-started,
        "scope": "Input encoding/length audit only; no context/training settings or Phase3 model/adapter implementation chosen."}
    atomic_json(stage/"completed.json", result)
    atomic_json(stage/"progress.json", {"status": "CPU_token_length_audit_complete", "passed": True,
        "documents": len(index), "sentences": sum(expected.values()), "optimizer_updates": 0})
    atomic_json(report/"token_length_audit.json", result)
    print(json.dumps({"passed": True, "files": summary, "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-preflight", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    run(parser.parse_args())
