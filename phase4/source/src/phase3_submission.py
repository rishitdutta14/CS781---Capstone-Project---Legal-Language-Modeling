"""Read-only checks for RR prediction files and submission archives.

These checks prove the output structure, not prediction quality or sentence
alignment inside the model. The inference path must separately prove that it
emits labels in the original sentence order.
"""

import hashlib
import json
from pathlib import Path
import stat
import zipfile


DATASET_REVISION = "89edba6a44d6bb9593fd13b6895b5cd3b06eb82e"
MAX_PREDICTION_BYTES = 16 * 1024 * 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_loads(payload):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")

    return json.loads(payload.decode("utf-8-sig"),
                      object_pairs_hook=unique_pairs,
                      parse_constant=invalid_constant)


def normalized_id(doc_id):
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("Document IDs must be nonempty strings")
    return doc_id[:-4] if doc_id.endswith(".txt") else doc_id


def load_test_contract(raw_dir, manifest_path):
    """Derive expected counts from the pinned raw files, not a hand-built index."""
    raw_dir, manifest_path = Path(raw_dir), Path(manifest_path)
    manifest = strict_loads(manifest_path.read_bytes())
    if (manifest.get("repo_id") != "Exploration-Lab/CS-781-Capstone"
            or manifest.get("revision") != DATASET_REVISION):
        raise ValueError("Unexpected dataset repository or revision")
    entries = {}
    for entry in manifest["files"]:
        name = entry["repository_path"]
        if name in entries:
            raise ValueError("Duplicate repository path in download manifest")
        entries[name] = entry

    inputs = {}

    def read_verified(relative):
        path = raw_dir / relative
        entry = entries[f"Phase_3_RR/{relative}"]
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != entry["sha256"] or len(payload) != entry["bytes"]:
            raise ValueError(f"Raw input differs from pinned manifest: {relative}")
        inputs[str(path.resolve())] = actual
        return strict_loads(payload)

    vocab = read_verified("Train/label_vocab.json")
    if (not isinstance(vocab, dict) or len(vocab) != 13
            or any(not isinstance(label, str) or not label.strip() for label in vocab)
            or any(type(index) is not int for index in vocab.values())
            or sorted(vocab.values()) != list(range(13))):
        raise ValueError("Expected the official 13-label mapping with IDs 0..12")
    labels = sorted(vocab, key=vocab.get)
    counts, domains, seen = {}, {}, set()
    for domain in ("CL", "IT"):
        records = read_verified(f"Test/{domain}_test.json")
        if not isinstance(records, dict):
            raise ValueError("Test data must map IDs to sentence lists")
        domain_sentences = 0
        for doc_id, sentences in records.items():
            key = normalized_id(doc_id)
            if key in seen:
                raise ValueError("Test IDs collide after evaluator normalization")
            seen.add(key)
            if (not isinstance(sentences, list) or not sentences
                    or any(not isinstance(text, str) for text in sentences)):
                raise ValueError("Expected a nonempty ordered list of sentence strings")
            counts[doc_id] = len(sentences)
            domain_sentences += len(sentences)
        domains[domain] = {"documents": len(records), "sentences": domain_sentences}
    if len(counts) != 10 or sum(counts.values()) != 2197:
        raise ValueError("Pinned test data no longer has 10 documents / 2197 sentences")
    return {"sentence_counts": counts, "labels": labels, "domains": domains,
            "input_sha256": inputs, "download_manifest_sha256": file_sha256(manifest_path),
            "dataset_revision": DATASET_REVISION}


def validate_predictions(predictions, contract):
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must map original document IDs to label lists")
    seen = set()
    for doc_id in predictions:
        key = normalized_id(doc_id)
        if key in seen:
            raise ValueError("Prediction IDs collide after evaluator normalization")
        seen.add(key)
    expected = contract["sentence_counts"]
    if set(predictions) != set(expected):
        missing = sorted(set(expected) - set(predictions))
        extra = sorted(set(predictions) - set(expected))
        raise ValueError(f"Original ID mismatch: missing={missing}, extra={extra}")
    allowed = set(contract["labels"])
    for doc_id, count in expected.items():
        labels = predictions[doc_id]
        if not isinstance(labels, list) or len(labels) != count:
            raise ValueError(f"{doc_id}: expected a list of exactly {count} labels")
        if any(not isinstance(label, str) or label not in allowed for label in labels):
            raise ValueError(f"{doc_id}: unknown or non-string label")
    return {"structure_passed": True, "documents": len(expected),
            "sentences": sum(expected.values()), "sentence_counts": dict(expected),
            "scope": "Output structure only; not model quality, training completion or semantic sentence-order proof"}


def archive_payload(archive_path):
    """Read, never extract. Reject extra members and pathological payload sizes."""
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != "prediction.json":
            raise ValueError("ZIP must contain exactly one root-level prediction.json")
        member = members[0]
        mode = member.external_attr >> 16
        if member.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise ValueError("prediction.json must be a regular file, not a link or directory")
        if member.flag_bits & 1:
            raise ValueError("Encrypted prediction files are not supported")
        if member.file_size > MAX_PREDICTION_BYTES:
            raise ValueError("prediction.json exceeds the 16 MiB validation safety limit")
        with archive.open(member) as stream:
            payload = stream.read(MAX_PREDICTION_BYTES + 1)
        if len(payload) > MAX_PREDICTION_BYTES:
            raise ValueError("Decompressed prediction exceeds the validation safety limit")
        if len(payload) != member.file_size or archive.testzip() is not None:
            raise ValueError("ZIP size or CRC verification failed")
    return payload


def validate_submission(prediction_path, archive_path, contract):
    prediction_path, archive_path = Path(prediction_path), Path(archive_path)
    if prediction_path.stat().st_size > MAX_PREDICTION_BYTES:
        raise ValueError("prediction.json exceeds the 16 MiB validation safety limit")
    payload = prediction_path.read_bytes()
    result = validate_predictions(strict_loads(payload), contract)
    zipped = archive_payload(archive_path)
    validate_predictions(strict_loads(zipped), contract)
    if zipped != payload:
        raise ValueError("ZIP payload differs from the standalone prediction.json bytes")
    return {**result, "zip_integrity_passed": True, "zip_matches_prediction_bytes": True,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": hashlib.sha256(payload).hexdigest(),
            "zip_path": str(archive_path.resolve()), "zip_sha256": file_sha256(archive_path),
            "input_contract": contract}
