"""Strict validation of generated Phase 2 artifacts against original test text."""

import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("official_phase2_evaluator", ROOT / "official/evaluation_scripts/evaluate_phase2.py")
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def validate_predictions(predictions, texts, labels):
    if not isinstance(predictions, dict) or set(predictions) != set(texts):
        raise ValueError("submission must contain exactly the official test IDs")
    total = 0
    for key, text in texts.items():
        row = predictions[key]
        if not isinstance(row, list):
            raise ValueError("document predictions must be lists, never null")
        entities = evaluator.spans(row, set(labels), key)
        for a, b, _ in entities:
            if b > len(text):
                raise ValueError("entity exceeds original Unicode character length")
        if not text and row != []:
            raise ValueError("empty text needs empty predictions")
        if any(len(span) == 4 and span[3] != text[span[0]:span[1]] for span in row):
            raise ValueError("optional entity text does not match raw characters")
        total += len(entities)
    return {"passed": True, "document_ids": len(predictions), "entities": total,
            "empty_documents": [key for key, text in texts.items() if not text],
            "overlap_policy": "Official evaluator imposes no non-overlap restriction; exact duplicates are rejected."}


def validate_archive(path, labels):
    texts = evaluator.load_json(ROOT / "data/raw/Phase_2_LNER/Test/fold3-text.json")
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["prediction.json"] or archive.testzip() is not None:
            raise ValueError("archive must contain exactly prediction.json at root with valid CRC")
        predictions = json.loads(archive.read("prediction.json"),
                                 object_pairs_hook=evaluator.pairs_no_duplicates,
                                 parse_constant=evaluator.reject_constant)
    return validate_predictions(predictions, texts, labels)
