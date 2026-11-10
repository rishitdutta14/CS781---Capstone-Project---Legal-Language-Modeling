"""Explicit source selection and full-reference continuation windows."""

import json
from pathlib import Path
import numpy as np


def select_source(ids, budget=1024):
    """Keep contiguous opening/middle/closing spans, in document order."""
    if budget < 8:
        raise ValueError('Source budget is too small')
    if len(ids) <= budget:
        return [int(token) for token in ids], [(0, len(ids))]
    first = budget * 3 // 8
    middle = budget // 4
    last = budget - first - middle
    spans = [(0, first), ((len(ids) - middle) // 2, (len(ids) - middle) // 2 + middle),
             (len(ids) - last, len(ids))]
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('Selected source spans overlap')
    return [int(token) for start, end in spans for token in ids[start:end]], spans


def windows(length, block=512):
    if length < 0 or block < 1:
        raise ValueError('Invalid reference length/window')
    # The EOS target is supervised once, including for an exact block multiple.
    return [(start, min(length + 1, start + block)) for start in range(0, length + 1, block)]


class SummaryData:
    def __init__(self, manifest, source_budget=1024):
        self.manifest = json.loads(Path(manifest).read_text())
        self.rows, self.sources, self.targets, self.selected = {}, {}, {}, {}
        self.source_budget = source_budget
        for split, files in self.manifest['splits'].items():
            self.rows[split] = json.loads(Path(files['index']).read_text())
            self.sources[split] = np.memmap(files['source'], mode='r', dtype='<u2')
            self.targets[split] = (np.memmap(files['target'], mode='r', dtype='<u2')
                                   if Path(files['target']).stat().st_size else np.empty(0, dtype='<u2'))
            self.selected[split] = []
            for row in self.rows[split]:
                start = row['source_offset']
                tokens = self.sources[split][start:start + row['source_tokens']]
                selected, spans = select_source(tokens, source_budget)
                self.selected[split].append((selected, spans))

    def reference(self, split, index):
        row = self.rows[split][index]
        start = row['target_offset']
        return self.targets[split][start:start + row['target_tokens']].astype(np.int64).tolist()

    def jobs(self, split, block=512, alternate=None):
        result = []
        for i, row in enumerate(self.rows[split]):
            for start, end in windows(row['target_tokens'], block):
                result.append((i, start, end, 0))
            if alternate is not None:
                for start, end in windows(len(alternate[i]), block):
                    result.append((i, start, end, 1))
        return result


def example(source, reference, start, end, encoder_decoder=False, source_budget=1024, history=60):
    if start < 0 or end <= start or end > len(reference) + 1:
        raise ValueError('Invalid supervised target interval')
    selected = select_source(source, source_budget)[0]
    padded_source = selected + [0] * (source_budget - len(selected))
    previous = reference[max(0, start - history):start]
    target = (reference + [3])[start:end]
    if encoder_decoder:
        context = [2] + previous
        source_ids = [2] + padded_source
    else:
        context = [2] + padded_source + [3, 2] + previous
        source_ids = None
    ids = context + target[:-1]
    labels = [-100] * (len(context) - 1) + target
    if len(ids) != len(labels) or len(ids) + (len(source_ids) if source_ids else 0) > 1600:
        raise ValueError('Context accounting exceeded the Phase1 capacity')
    return source_ids, ids, labels


def validate_predictions(records, expected):
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError('Exactly one summary is required for each official test ID')
    seen = set()
    for row in records:
        if (set(row) != {'id', 'summary'} or not isinstance(row['id'], str)
                or row['id'] in seen or not isinstance(row['summary'], str) or not row['summary'].strip()):
            raise ValueError('Invalid/duplicate ID or empty generated summary')
        seen.add(row['id'])
    if seen != set(expected):
        raise ValueError('Official test IDs differ')
