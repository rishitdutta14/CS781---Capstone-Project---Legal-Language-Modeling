"""Full LSI encoding with frozen A1 tokenizers; no model or CUDA imports."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from src.phase1_full_corpus_encode import CachedDocumentEncoder
from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.tokenization.a1_byte_bpe import ByteBPE

_ENCODER = None


def binary_target(labels, dimension=100):
    """Repeated annotations express the same positive, never extra weight."""
    if type(dimension) is not int or dimension < 1 or not isinstance(labels, list):
        raise ValueError("Expected a label list and positive integer dimension")
    result = np.zeros(dimension, dtype=np.float32)
    for label in labels:
        if type(label) is not int or not 0 <= label < dimension:
            raise ValueError("Statute ID outside the training label vocabulary")
        result[label] = 1.
    return result


def chunk_spans(length, capacity=1600):
    if type(length) is not int or length < 0 or type(capacity) is not int or capacity < 2:
        raise ValueError("Invalid token length or context capacity")
    return ([(0, 0)] if length == 0 else
            [(start, min(length, start + capacity - 1))
             for start in range(0, length, capacity - 1)])


def initialize_encoder(tokenizer):
    global _ENCODER
    _ENCODER = CachedDocumentEncoder(tokenizer)
    if _ENCODER.tokenizer.vocab_size > 65536:
        raise ValueError("uint16 cache cannot represent this vocabulary")


def encode_rows(rows):
    result = []
    for row in rows:
        ids = _ENCODER.encode(row['text'])
        if ids and (min(ids) < 4 or max(ids) >= _ENCODER.tokenizer.vocab_size):
            raise ValueError("Unknown, special, or out-of-range content ID")
        if _ENCODER.decode(ids) != ByteBPE.normalize(row['text']):
            raise ValueError("Inherited A1 canonical-whitespace round trip failed")
        result.append((row, np.asarray(ids, dtype='<u2')))
    return result


def input_batches(path, split, batch_size=16):
    batch = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream):
            row = json.loads(line)
            fields = {'id', 'row_key', 'text'} | (set() if split == 'test' else {'labels'})
            if set(row) != fields or row['row_key'] != f'{split}:{number}':
                raise ValueError("Unsafe fields or altered official row order")
            if not isinstance(row['id'], str) or not isinstance(row['text'], str):
                raise ValueError("Invalid input ID or text")
            if split != 'test':
                binary_target(row['labels'])
            batch.append(row)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def ordered_encoding(pool, batches, pending_limit):
    """Bound queued text memory, preserving all records in official order."""
    pending = deque()
    for batch in batches:
        pending.append(pool.submit(encode_rows, batch))
        if len(pending) >= pending_limit:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def encode_dataset(inputs, tokenizer, output, workers=12, capacity=1600):
    if workers < 1:
        raise ValueError("At least one CPU encoder worker is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        'tokenizer': str(tokenizer), 'tokenizer_sha256': file_sha256(Path(tokenizer)),
        'dtype': 'little-endian uint16', 'capacity': capacity,
        'content_tokens_per_chunk': capacity - 1, 'BOS_per_chunk': 1,
        'sampling': False, 'truncation': False, 'deduplication': False,
        'raw_text_modified': False, 'official_splits_unchanged': True,
        'tokenizer_retrained': False, 'test_labels_or_metadata_used': False,
        'A1_whitespace_semantics': 'Existing per-word leading space and canonical whitespace on decode; raw text is never rewritten',
        'chunking': 'Contiguous non-overlapping complete-document chunks; independent BOS; no cap',
        'splits': {},
    }
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_encoder,
                             initargs=(str(tokenizer),)) as pool:
        for split in ('train', 'dev', 'test'):
            source = inputs[split]
            if file_sha256(Path(source['path'])) != source['sha256']:
                raise ValueError("Safe input view changed after the raw audit")
            binary = output / f'{split}.bin'
            index_path = output / f'{split}_index.json'
            index, offset = [], 0
            with binary.open('xb') as stream:
                for batch in ordered_encoding(pool, input_batches(source['path'], split), workers * 2):
                    for row, ids in batch:
                        entry = {key: row[key] for key in ('id', 'row_key')}
                        entry.update(offset=offset, tokens=len(ids),
                                     chunks=len(chunk_spans(len(ids), capacity)),
                                     text_sha256=hashlib.sha256(row['text'].encode()).hexdigest())
                        if split != 'test':
                            entry['labels'] = row['labels']
                        ids.tofile(stream)
                        index.append(entry)
                        offset += len(ids)
                    if len(index) % 512 == 0:
                        atomic_json(output / 'progress.json', {
                            'split': split, 'documents_done': len(index), 'tokens_done': offset,
                            'elapsed_seconds': time.monotonic() - started, 'GPU_training_started': False})
                stream.flush()
                os.fsync(stream.fileno())
            if binary.stat().st_size != 2 * offset or not index:
                raise ValueError("Token byte accounting or split nonemptiness failed")
            atomic_json(index_path, index)
            lengths = np.asarray([row['tokens'] for row in index])
            chunks = [row['chunks'] for row in index]
            manifest['splits'][split] = {
                'input': source, 'binary': str(binary), 'binary_sha256': file_sha256(binary),
                'index': str(index_path), 'index_sha256': file_sha256(index_path),
                'documents': len(index), 'tokens': offset, 'chunks': sum(chunks),
                'mean_chunks': sum(chunks) / len(chunks), 'max_chunks': max(chunks),
                'min_tokens': int(lengths.min()), 'max_tokens': int(lengths.max()),
                'p50_tokens': float(np.quantile(lengths, .5)),
                'p95_tokens': float(np.quantile(lengths, .95)),
                'p99_tokens': float(np.quantile(lengths, .99)),
                'documents_needing_multiple_chunks': int((lengths > capacity - 1).sum()),
                'every_document_decoded_and_round_trip_checked': True, 'unknown_content_IDs': 0,
            }
    manifest['elapsed_seconds'] = time.monotonic() - started
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'completed.json', {
        'manifest_sha256': file_sha256(output / 'manifest.json'), 'GPU_training_started': False})
    return manifest


def verify_alignment(manifests, audited_inputs):
    """All tokenizers must retain exactly the same official rows and annotations."""
    for split in ('train', 'dev', 'test'):
        expected = [row for batch in input_batches(audited_inputs[split]['path'], split) for row in batch]
        for manifest in manifests:
            rows = json.loads(Path(manifest['splits'][split]['index']).read_text())
            if len(rows) != len(expected):
                raise ValueError("Encoding omitted or added official rows")
            for raw, encoded in zip(expected, rows):
                if any(raw[key] != encoded[key] for key in ('id', 'row_key')):
                    raise ValueError("Cross-tokenizer case identity/order mismatch")
                if hashlib.sha256(raw['text'].encode()).hexdigest() != encoded['text_sha256']:
                    raise ValueError("Encoded original text identity mismatch")
                if split != 'test' and raw['labels'] != encoded['labels']:
                    raise ValueError("Original raw label list changed")
                if split == 'test' and set(encoded) != {'id', 'row_key', 'offset', 'tokens', 'chunks', 'text_sha256'}:
                    raise ValueError("Test cache includes non-input information")
    return {'all_rows_text_hashes_and_raw_labels_match': True,
            'test_caches_label_free': True, 'official_splits_unchanged': True}
