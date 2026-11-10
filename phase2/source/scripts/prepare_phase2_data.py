#!/usr/bin/env python3
"""Detached-safe Phase 2 data preflight. Never loads or trains a model."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase2_data import (choose_split, make_entity_records, make_windows,
                             restore_entity, shared_entity_tokens)
from src.tokenization.a1_byte_bpe import ByteBPE
from src.tokenization.a1_offsets import A1OffsetTokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False,
                               indent=None if compact else 2) + "\n", encoding="utf-8")


def prepare(args):
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("output directory already exists; use a new path, never overwrite a run")
    output.mkdir(parents=True)
    status_path = output / "preflight.json"
    status = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "training_launched": False, "model_head_implemented": False,
              "gpu_used": False, "output_dir": str(output)}
    save(status_path, status)
    try:
        run_checks(output)
        raw = ROOT / "data/raw/Phase_2_LNER"
        inventory = json.loads((ROOT / "reports/phase2/data_inventory.json").read_text())
        inputs = {}
        for relative, expected in inventory["files"].items():
            path = raw / relative
            digest = sha(path)
            if digest != expected["sha256"]:
                raise ValueError(f"official data hash changed: {relative}")
            inputs[str(path)] = digest
        tokenizer_path = args.tokenizer.resolve()
        tokenizer_hash = sha(tokenizer_path)
        if tokenizer_path == (ROOT / "tokenizers/phase1_v2/capstone_legal_bpe_16k.json").resolve():
            old_audit = json.loads((ROOT / "reports/phase2/length_audit/length_audit.json").read_text())
            if tokenizer_hash != old_audit["inputs"]["tokenizer"]["sha256"]:
                raise ValueError("tokenizer changed since the length audit")
        elif tokenizer_path == (ROOT / "tokenizers/capstone_legal_bpe_16k.json").resolve():
            hashes = json.loads((ROOT / "outputs/phase1/mha/hashes.json").read_text())
            if tokenizer_hash != hashes["tokenizer"]["sha256"]:
                raise ValueError("original tokenizer changed since Phase 1")
        else:
            raise ValueError("only the two existing checkpoint tokenizer families are allowed")
        inputs[str(tokenizer_path)] = tokenizer_hash
        tokenizer = ByteBPE.load(tokenizer_path)
        wrapper = A1OffsetTokenizer(tokenizer)
        allowed = [line.strip() for line in (raw / "Train/labels.txt").read_text().splitlines() if line.strip()]
        if len(allowed) != 12 or len(set(allowed)) != 12:
            raise ValueError("expected 12 unique labels")
        texts, gold, fold_ids, origins = {}, {}, {}, {}
        for fold in ("fold1", "fold2", "fold3"):
            folder = "Test" if fold == "fold3" else "Train"
            part = json.loads((raw / folder / f"{fold}-text.json").read_text())
            if len(part) != 35 or set(part) & set(texts):
                raise ValueError("wrong document count or repeated document ID")
            texts.update(part)
            origins.update({key: fold for key in part})
            if folder == "Train":
                annotations = json.loads((raw / folder / f"{fold}-gold.json").read_text())
                if set(annotations) != set(part):
                    raise ValueError("text/gold document IDs differ")
                gold.update(annotations)
                fold_ids[fold] = list(part)
        split = choose_split(fold_ids, args.seed)
        split["test"] = sorted(key for key in texts if key not in gold)
        reference = json.loads(args.split_reference.read_text())["document_ids"]
        if split != reference:
            raise ValueError("split differs from the frozen Phase 2 document split")
        if [len(split[name]) for name in ("train", "validation", "test")] != [60, 10, 35]:
            raise ValueError("expected a 60/10/35 document split")
        assignments = {key: name for name, keys in split.items() for key in keys}
        # Detect duplicate complete texts across partitions, without examining
        # test labels (none are available) or modifying any text.
        hashes = {}
        for key, text in texts.items():
            if not isinstance(text, str):
                raise ValueError("document text must be a string")
            if text:
                digest = hashlib.sha256(text.encode()).hexdigest()
                previous = hashes.get(digest)
                if previous and assignments[previous] != assignments[key]:
                    raise ValueError(f"exact duplicate text crosses partitions: {previous}/{key}")
                hashes[digest] = key
        config = {"context_limit": 1600, "content_token_capacity": 1600,
                  "overlap_tokens": args.overlap, "added_special_tokens": [],
                  "special_token_policy": "No BOS/EOS inserted in these NER data artifacts; if later added, regenerate with reserved capacity.",
                  "split_seed": args.seed, "split_method": "SHA256(seed:fold:document_id), first five per labelled fold held out",
                  "raw_text_modified": False, "tokenizer_retrained": False,
                  "tokenizer_sha256": tokenizer_hash, "tokenizer_path": str(tokenizer_path),
                  "entity_targets": "Original gold spans plus signed token-relative character displacements, including omitted whitespace; not BIO labels or a model-head choice.",
                  "partial_entity_policy": "Listed explicitly as ignored supervision; a future training adapter must implement corresponding masks.",
                  "overlap_supervision_weighting": "Not chosen; complete gold entities are listed in every containing window.",
                  "conflicting_prediction_policy": "Not chosen; only exact-duplicate removal is tested using gold reconstructions."}
        code_paths = [Path(__file__), ROOT / "src/phase2_data.py", ROOT / "src/tokenization/a1_offsets.py",
                      ROOT / "src/tokenization/a1_byte_bpe.py", ROOT / "tests/test_phase2_data.py",
                      ROOT / "tests/test_a1_offsets.py", ROOT / "official/evaluation_scripts/evaluate_phase2.py"]
        save(output / "manifest.json", {"config": config, "document_ids": split,
                                       "inputs_sha256": inputs,
                                       "code_sha256": {str(p): sha(p) for p in code_paths},
                                       "labels": allowed, "dataset_revision": inventory["dataset_revision"]})
        counts = {name: Counter() for name in split}
        label_counts = {name: Counter() for name in ("train", "validation")}
        boundary_counts = Counter({"documents_with_shared_entity_tokens": 0,
                                   "tokens_shared_by_multiple_entities": 0,
                                   "entities_with_boundaries_in_omitted_whitespace": 0})
        shared_examples, inside_word_examples = [], []
        reconstructed = {}
        empty_ids = []
        for key in sorted(texts):
            text = texts[key]
            name = assignments[key]
            encoding = wrapper.encode(text)
            if encoding.ids != tokenizer.encode_ids(text):
                raise ValueError(f"tokenizer IDs changed for {key}")
            if tokenizer.unk_token_id in encoding.ids:
                raise ValueError(f"unknown token in {key}")
            if len(encoding.ids) != len(encoding.offsets) or any(
                    not 0 <= a <= b <= len(text) for a, b in encoding.offsets):
                raise ValueError(f"invalid token offsets for {key}")
            entities = make_entity_records(text, encoding.offsets, gold.get(key, []), set(allowed))
            windows = make_windows(encoding.offsets, entities, 1600, args.overlap)
            shares = shared_entity_tokens(entities)
            if shares:
                boundary_counts["documents_with_shared_entity_tokens"] += 1
                boundary_counts["tokens_shared_by_multiple_entities"] += len(shares)
                if len(shared_examples) < 10:
                    example = shares[0]
                    a, b = encoding.offsets[example["token"]]
                    shared_examples.append({"document_id": key, "token_characters": text[a:b],
                                            **example, "entities": [entities[i] for i in example["entity_numbers"]]})
            if name != "test":
                restored, duplicated = {}, 0
                for w in windows:
                    for number in w["complete_entity_numbers"]:
                        entity = entities[number]
                        # Convert window-local token indices back to document
                        # indices explicitly, then restore original characters.
                        local = dict(entity)
                        local["token_start"] -= w["token_start"]
                        local["token_end"] -= w["token_start"]
                        value = restore_entity(local, encoding.offsets[w["token_start"]:w["token_end"]], text)
                        span_key = tuple(value)
                        duplicated += span_key in restored
                        restored[span_key] = value
                expected = {tuple(span[:3]) for span in gold[key]}
                if set(restored) != expected:
                    raise ValueError(f"gold window round trip lost or changed spans: {key}")
                reconstructed[key] = sorted(restored.values(), key=lambda s: (s[0], s[1], s[2]))
                counts[name]["duplicate_gold_window_predictions_removed"] += duplicated
                label_counts[name].update(e["label"] for e in entities)
            for e in entities:
                boundary_counts["gold_entities"] += 1
                boundary_counts["nonexact_token_boundaries"] += e["nonexact_token_boundary"]
                boundary_counts["entities_with_boundaries_in_omitted_whitespace"] += e["boundary_in_omitted_whitespace"]
                boundary_counts["boundaries_inside_whitespace_delimited_words"] += e["boundary_inside_word"]
                if e["boundary_inside_word"] and len(inside_word_examples) < 10:
                    inside_word_examples.append({"document_id": key, "entity_text": text[e["start"]:e["end"]], **e})
            if not encoding.ids:
                empty_ids.append(key)
            counts[name].update(documents=1, tokens=len(encoding.ids), windows=len(windows), entities=len(entities),
                                partial_entity_window_occurrences=sum(len(w["partial_entity_numbers_ignore_supervision"]) for w in windows))
            payload = {"document_id": key, "fold": origins[key], "split": name,
                       "text": text, "token_ids": encoding.ids, "token_offsets": encoding.offsets,
                       "entities": entities, "windows": windows}
            # IDs come from official data; reject path-like IDs before writing.
            if not key.isdigit():
                raise ValueError("expected numeric official document IDs")
            save(output / "documents" / name / f"{key}.json", payload, compact=True)
            print(f"Prepared {name} {key}: {len(encoding.ids)} tokens, {len(windows)} windows", flush=True)
        if boundary_counts["gold_entities"] != inventory["gold_entity_annotations"]:
            raise ValueError("not all gold entities were processed")
        save(output / "gold_roundtrip" / "truth.json", gold)
        save(output / "gold_roundtrip" / "reconstructed_gold_NOT_MODEL_PREDICTIONS.json", reconstructed)
        result = subprocess.run([sys.executable, str(ROOT / "official/evaluation_scripts/evaluate_phase2.py"),
                                 "--truth", str(output / "gold_roundtrip/truth.json"),
                                 "--prediction", str(output / "gold_roundtrip/reconstructed_gold_NOT_MODEL_PREDICTIONS.json"),
                                 "--labels", str(raw / "Train/labels.txt")], text=True, capture_output=True, check=True)
        (output / "gold_roundtrip/evaluator.txt").write_text(result.stdout, encoding="utf-8")
        if "Strict Macro-F1:        1.000000" not in result.stdout:
            raise ValueError("gold reconstruction failed official strict scoring")
        if any(sha(Path(path)) != digest for path, digest in inputs.items()):
            raise ValueError("an input file changed during preparation")
        status.update(status="passed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=round(time.monotonic()-started, 2), split_counts=counts,
                      label_counts=label_counts, train_labels_missing=sorted(set(allowed)-set(label_counts["train"])),
                      validation_labels_missing=sorted(set(allowed)-set(label_counts["validation"])),
                      boundary_audit=dict(boundary_counts), shared_entity_token_examples=shared_examples,
                      inside_word_boundary_examples=inside_word_examples, empty_document_ids=empty_ids,
                      all_original_tokenizer_ids_unchanged=True, all_source_hashes_unchanged=True,
                      gold_window_roundtrip_exact=True, gold_roundtrip_macro_f1=1.0,
                      gold_roundtrip_warning="This is a bookkeeping test using gold annotations, NOT model performance and NOT a submission.",
                      training_ready=False,
                      unresolved_before_training=["Choose an exact-span output head after inspecting inside-token, shared-token, and omitted-whitespace boundaries.",
                                                  "Implement and test partial-entity supervision masks and overlap loss weighting for that head.",
                                                  "Choose a rule for conflicting window predictions; do not use test labels.",
                                                  "Clarify Phase 2 model-code LLM permission before assistant-led model implementation.",
                                                  "Check GPU memory and storage before any training launch."])
        if status["train_labels_missing"]:
            raise ValueError("training partition is missing entity labels")
        save(status_path, status)
        print(json.dumps({k: status[k] for k in ("status", "elapsed_seconds", "split_counts", "boundary_audit", "training_ready")}, indent=2), flush=True)
    except Exception as exc:
        status.update(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic()-started, 2))
        save(status_path, status)
        raise


def run_checks(output):
    for filename in ("test_a1_offsets.py", "test_phase2_data.py"):
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"),
                                 "-p", filename, "-v"], text=True, capture_output=True)
        (output / f"{filename}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"unit tests failed: {filename}; see output log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers/phase1_v2/capstone_legal_bpe_16k.json")
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--split-reference", type=Path, default=ROOT / "data/processed/phase2_preflight_20261011_r2/manifest.json")
    prepare(parser.parse_args())
