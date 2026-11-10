"""Prepared LSI cases, validation-only thresholds, and strict ZIP validation."""

import json
from pathlib import Path
import zipfile

import numpy as np

from src.phase1_full_corpus_prepare import file_sha256
from src.phase5_prepare import binary_target, chunk_spans


class StatuteData:
    def __init__(self, path, verify=True):
        self.manifest = json.loads(Path(path).read_text())
        self.rows, self.tokens = {}, {}
        for split, item in self.manifest['splits'].items():
            if verify:
                for name in ('binary', 'index'):
                    if file_sha256(Path(item[name])) != item[name + '_sha256']:
                        raise ValueError('Prepared LSI input checksum changed')
            self.rows[split] = json.loads(Path(item['index']).read_text())
            self.tokens[split] = np.memmap(item['binary'], mode='r', dtype='<u2')
            if len(self.rows[split]) != item['documents'] or len(self.tokens[split]) != item['tokens']:
                raise ValueError('Prepared input counts changed')

    def chunks(self, split, index):
        row = self.rows[split][index]
        values = self.tokens[split][row['offset']:row['offset'] + row['tokens']]
        if len(values) != row['tokens']:
            raise ValueError('Truncated document tokens')
        return [[2] + values[start:end].astype(np.int64).tolist()
                for start, end in chunk_spans(row['tokens'], self.manifest['capacity'])]


def metrics(gold, predicted):
    gold = np.asarray(gold, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if gold.shape != predicted.shape or gold.ndim != 2 or gold.shape[1] != 100 or not len(gold):
        raise ValueError('Metrics require nonempty matching Nx100 indicator matrices')
    tp = (gold & predicted).sum(0)
    fp = (~gold & predicted).sum(0)
    fn = (gold & ~predicted).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(100), where=denom > 0)
    total = int(denom.sum())
    return {'macro_f1': float(f1.mean()),
            'micro_f1': float(2 * tp.sum() / total) if total else 0.,
            'subset_accuracy': float(np.all(gold == predicted, axis=1).mean()),
            'per_label_F1': f1.tolist(), 'positive_support': gold.sum(0).astype(int).tolist(),
            'predicted_positive_counts': predicted.sum(0).astype(int).tolist(), 'documents': len(gold)}


def choose_threshold(probabilities, gold):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != np.asarray(gold).shape or not np.isfinite(probabilities).all():
        raise ValueError('Invalid development probabilities')
    trials = []
    for integer in range(1, 20):
        threshold = integer / 20
        result = metrics(gold, probabilities >= threshold)
        trials.append({'threshold': threshold, 'macro_f1': result['macro_f1'],
                       'micro_f1': result['micro_f1'], 'subset_accuracy': result['subset_accuracy']})
    # Prefer the closest value to0.5 on ties, then the larger threshold.
    best = max(trials, key=lambda row: (row['macro_f1'], -abs(row['threshold'] - .5), row['threshold']))
    return best['threshold'], trials


def predictions(rows, probabilities, threshold):
    values = np.asarray(probabilities)
    if values.shape != (len(rows), 100) or not np.isfinite(values).all() or not 0 < threshold < 1:
        raise ValueError('Invalid multi-label probabilities or threshold')
    return [{'id': row['id'], 'pred': np.flatnonzero(value >= threshold).astype(int).tolist()}
            for row, value in zip(rows, values)]


def validate_predictions(records, expected_ids):
    if not isinstance(records, list) or len(records) != len(expected_ids):
        raise ValueError('Require exactly one prediction for every test case')
    seen = set()
    for record in records:
        if (not isinstance(record, dict) or set(record) != {'id', 'pred'}
                or not isinstance(record['id'], str) or record['id'] in seen
                or not isinstance(record['pred'], list)):
            raise ValueError('Invalid/duplicate test ID or prediction structure')
        seen.add(record['id'])
        for label in record['pred']:
            if type(label) is not int or not 0 <= label < 100:
                raise ValueError('Invalid statute ID')
        if len(set(record['pred'])) != len(record['pred']):
            raise ValueError('Duplicate statute ID in prediction')
    if seen != set(expected_ids) or len(seen) != len(expected_ids):
        raise ValueError('Official test ID set mismatch')


def package_submission(path, records, expected_ids):
    validate_predictions(records, expected_ids)
    path = Path(path)
    payload = json.dumps(records, ensure_ascii=False, allow_nan=False).encode()
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('prediction.json', payload)
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['prediction.json'] or archive.testzip() is not None:
            raise ValueError('Submission ZIP structure/CRC failed')
        validate_predictions(json.loads(archive.read('prediction.json')), expected_ids)
    return {'path': str(path), 'sha256': file_sha256(path), 'predictions': len(records)}


def stopping_update(progress, score, threshold, epoch, minimum_epochs=2, patience=2):
    if not np.isfinite(score):
        raise ValueError('Nonfinite validation selection score')
    improved = progress['best_score'] is None or score > progress['best_score']
    if improved:
        progress.update(best_score=score, best_threshold=threshold, best_epoch=epoch, bad_epochs=0)
    else:
        progress['bad_epochs'] += 1
    stop = epoch >= minimum_epochs and progress['bad_epochs'] >= patience
    return improved, stop
