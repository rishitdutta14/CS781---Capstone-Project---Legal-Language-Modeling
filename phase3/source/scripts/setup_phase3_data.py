"""Download the pinned official RR files unchanged and audit their structure."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from official.evaluation_scripts.evaluate_phase3 import load_vocab, normalize_id
from src.phase1_full_corpus_prepare import atomic_json, file_sha256

REPO = "Exploration-Lab/CS-781-Capstone"
FILES = [f"Phase_3_RR/Train/{name}.json" for name in
         ("CL_train", "CL_dev", "IT_train", "IT_dev", "label_vocab")]
FILES += [f"Phase_3_RR/Test/{name}.json" for name in ("CL_test", "IT_test")]


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json(path):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_pairs,
                      parse_constant=invalid)


def documents(data, labelled, allowed):
    if labelled:
        if not isinstance(data, list):
            raise ValueError("Official train/dev data must be a list of document records")
        records = data
    else:
        if not isinstance(data, dict):
            raise ValueError("Official test data must map IDs to ordered sentence lists")
        records = [{"id": key, "text": sentences} for key, sentences in data.items()]
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid document record")
        key = normalize_id(record["id"])
        if key in seen:
            raise ValueError("Duplicate document ID after official .txt normalization")
        seen.add(key)
        sentences = record["text"]
        if not isinstance(sentences, list) or any(not isinstance(s, str) for s in sentences):
            raise ValueError("Each document must contain an ordered list of sentence strings")
        if labelled:
            labels = record.get("labels")
            if not isinstance(labels, list) or len(labels) != len(sentences):
                raise ValueError("Sentence/label count mismatch")
            if any(not isinstance(label, str) or label not in allowed for label in labels):
                raise ValueError("Unknown or non-string rhetorical-role label")
        yield record


def content_hash(sentences):
    # Hash the exact sentence list: do not normalize whitespace or boundaries.
    raw = json.dumps(sentences, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_remote_bytes(raw, remote):
    if len(raw) != remote.size:
        raise ValueError(f"Downloaded size differs for {remote.rfilename}")
    lfs = remote.lfs
    if lfs:
        expected = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
        actual = hashlib.sha256(raw).hexdigest()
    else:
        expected = remote.blob_id
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"Downloaded bytes differ from pinned remote object: {remote.rfilename}")


def length_summary(values):
    import numpy as np
    if not values:
        return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "max": 0}
    return {"count": len(values), "mean": float(np.mean(values)),
            "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)), "max": max(values)}


def audit(raw_dir, report_dir):
    raw = raw_dir / "Phase_3_RR"
    vocab = strict_json(raw / "Train/label_vocab.json")
    labels = load_vocab(vocab)  # Reuse the unchanged official evaluator's vocabulary checks.
    allowed = set(labels)
    if not isinstance(vocab, dict) or sorted(vocab.values()) != list(range(13)):
        raise ValueError("Official label-to-ID mapping is not a unique contiguous 0..12 mapping")
    stats, index, identities, texts = {}, {}, defaultdict(list), defaultdict(list)
    test_ids = set()
    for domain in ("CL", "IT"):
        for split in ("train", "dev", "test"):
            path = raw / ("Test" if split == "test" else "Train") / f"{domain}_{split}.json"
            data = strict_json(path)
            records = list(documents(data, split != "test", allowed))
            counts = Counter()
            chars, words, per_doc = [], [], []
            empty_sentences = 0
            for record in records:
                doc_id = record["id"]
                normalized = normalize_id(doc_id)
                sentences = record["text"]
                location = {"domain": domain, "split": split, "id": doc_id}
                identities[normalized].append(location)
                texts[content_hash(sentences)].append(location)
                per_doc.append(len(sentences))
                chars.extend(map(len, sentences))
                words.extend(len(s.split()) for s in sentences)
                empty_sentences += sum(not s.strip() for s in sentences)
                if split == "test":
                    if normalized in test_ids:
                        raise ValueError("CL/IT test IDs collide after evaluator normalization")
                    test_ids.add(normalized)
                    index[doc_id] = {"domain": domain, "sentences": len(sentences), "evaluator_id": normalized}
                else:
                    counts.update(record["labels"])
            stats[f"{domain}_{split}"] = {"file": str(path), "sha256": file_sha256(path),
                "documents": len(records), "sentences": sum(per_doc), "sentences_per_document": length_summary(per_doc),
                "sentence_characters": length_summary(chars), "sentence_whitespace_words": length_summary(words),
                "empty_sentences_preserved": empty_sentences,
                "empty_documents_preserved": per_doc.count(0),
                "label_counts": {label: counts[label] for label in labels} if split != "test" else None,
                "labels_absent_in_this_file": [label for label in labels if not counts[label]] if split != "test" else None}
    id_overlap = {key: entries for key, entries in identities.items() if len(entries) > 1}
    exact_overlap = {key: entries for key, entries in texts.items()
                     if len({e["split"] for e in entries}) > 1}
    warnings = []
    if id_overlap:
        warnings.append("Document IDs overlap between official files; inspect split_overlap.json before training")
    if exact_overlap:
        warnings.append("Exact sentence-list duplicates cross official splits; no records removed or resplit")
    if any(row["empty_sentences_preserved"] or row["empty_documents_preserved"] for row in stats.values()):
        warnings.append("Empty sentences/documents preserved; decide explicit handling before training")
    atomic_json(report_dir / "split_overlap.json", {"duplicate_normalized_ids": id_overlap,
        "exact_document_text_across_splits": exact_overlap,
        "scope": "ID and exact ordered-sentence-list checks; not a fuzzy/semantic duplicate audit"})
    atomic_json(report_dir / "test_index.json", index)
    result = {"status": "data_setup_complete_with_warnings" if warnings else "data_setup_complete",
        "structural_audit_passed": True, "labels": vocab, "files": stats, "warnings": warnings,
        "combined_test_documents": len(index), "combined_test_sentences": sum(r["sentences"] for r in index.values()),
        "raw_text_and_sentence_order_unchanged": True, "official_splits_preserved": True,
        "tokenization_cleaning_or_model_training_performed": False,
        "label_semantics": "The label named None is a string class, not JSON null",
        "test_output_contract": "One prediction.json mapping original document IDs to label lists of EXACT matching sentence count/order; both domains together",
        "metric": "Official evaluator macro-F1 over all 13 labels; CL+IT sentence predictions concatenated, not an average of domain F1 scores"}
    atomic_json(report_dir / "data_audit.json", result)
    return result


def run(args):
    from huggingface_hub import HfApi, hf_hub_download
    report = args.report_dir.resolve()
    report.mkdir(parents=True, exist_ok=False)
    try:
        info = HfApi().dataset_info(REPO, revision=args.revision, files_metadata=True)
        revision = info.sha
        metadata = {r.rfilename: r for r in info.siblings}
        if any(name not in metadata for name in FILES):
            raise ValueError("A required official Phase3 file is absent at the pinned revision")
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        required = sum(metadata[name].size for name in FILES)
        if shutil.disk_usage(args.raw_dir).free < required*3 + 20*2**20:
            raise RuntimeError("Insufficient storage; do not remove old artifacts")
        # Inspect ALL existing destinations before asking the SDK to download.
        for name in FILES:
            existing = args.raw_dir / name
            if existing.exists():
                verify_remote_bytes(existing.read_bytes(), metadata[name])
        def download(name):
            path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=revision,
                                      local_dir=str(args.raw_dir)))
            verify_remote_bytes(path.read_bytes(), metadata[name])
            return {"repository_path": name, "local_path": str(path.resolve()),
                    "bytes": path.stat().st_size, "sha256": file_sha256(path),
                    "remote_blob_id": metadata[name].blob_id, "remote_content_integrity_verified": True}
        with ThreadPoolExecutor(max_workers=4) as pool:
            downloaded = list(pool.map(download, FILES))
        manifest = {"repo_id": REPO, "revision": revision, "files": downloaded,
            "downloaded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "existing_unrelated_files_modified": False, "official_files_edited": False,
            "setup_script_sha256": file_sha256(Path(__file__)),
            "official_evaluator_sha256": file_sha256(ROOT/"official/evaluation_scripts/evaluate_phase3.py")}
        atomic_json(report / "download_manifest.json", manifest)
        result = audit(args.raw_dir, report)
        atomic_json(report / "COMPLETED.json", {"data_setup_complete": True,
            "download_manifest_sha256": file_sha256(report/"download_manifest.json"),
            "data_audit_sha256": file_sha256(report/"data_audit.json"),
            "warnings": result["warnings"], "training_started": False})
        print(json.dumps(result, indent=2))
    except BaseException as error:
        atomic_json(report / "FAILED.json", {"error": f"{type(error).__name__}: {error}", "training_started": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"data/raw")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    run(parser.parse_args())
