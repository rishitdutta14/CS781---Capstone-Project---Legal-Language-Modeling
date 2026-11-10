"""Train-only, raw-character-aligned entity-type knowledge distillation.

BIO beginnings depend on tokenization. We therefore distill O plus entity
type mass (B + I), while ordinary BIO and exact-boundary losses supervise
the student's own boundaries. Teacher probabilities never supply gold spans.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


def temperature_value(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("KD temperature must be finite and positive")
    return float(value)


def type_log_probabilities(logits, temperature=2.):
    if logits.shape[-1] < 3 or logits.shape[-1] % 2 != 1:
        raise ValueError("expected O followed by paired B/I tags")
    logp = F.log_softmax(logits.float()/temperature_value(temperature), dim=-1)
    paired = logp[..., 1:].reshape(*logp.shape[:-1], -1, 2).logsumexp(-1)
    return torch.cat((logp[..., :1], paired), dim=-1)


def label_order(tags):
    if not tags or tags[0] != 'O' or len(tags) % 2 != 1:
        raise ValueError("invalid BIO label order")
    labels = []
    for begin, inside in zip(tags[1::2], tags[2::2]):
        if not begin.startswith('B-') or inside != 'I-' + begin[2:]:
            raise ValueError("BIO tags must be ordered in matching B/I pairs")
        labels.append(begin[2:])
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate entity labels")
    return ['O'] + labels


def identity_sha256(value):
    encoded = value.encode('utf-8') if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _offset_tensor(offsets, length):
    if any(len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
           or not 0 <= pair[0] <= pair[1] <= length for pair in offsets):
        raise ValueError("invalid original Unicode character offsets")
    return torch.tensor(offsets, dtype=torch.long).reshape(-1, 2)


def align_type_probabilities(text, teacher_offsets, teacher_probabilities, student_offsets):
    """Average overlapping byte pieces per character, then pool student spans.

    Each Unicode character has equal weight regardless of its UTF-8 byte count.
    Uncovered non-whitespace is an error. Uncovered whitespace contributes no
    pseudo-target; tokens with no covered characters have KD disabled, but their
    ordinary supervised BIO loss is untouched. No text normalization occurs.
    """
    teacher = _offset_tensor(teacher_offsets, len(text))
    student = _offset_tensor(student_offsets, len(text))
    probabilities = torch.as_tensor(teacher_probabilities).detach().double().cpu()
    if probabilities.ndim != 2 or probabilities.shape[0] != len(teacher):
        raise ValueError("teacher probabilities and offsets differ")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or not torch.allclose(
            probabilities.sum(-1), torch.ones(len(teacher), dtype=torch.double), atol=2e-3, rtol=0):
        raise ValueError("teacher probabilities are not finite normalized distributions")
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    # Difference arrays avoid a Python operation for every byte-level token.
    delta = torch.zeros(len(text)+1, probabilities.shape[1], dtype=torch.double)
    coverage_delta = torch.zeros(len(text)+1, dtype=torch.long)
    if len(teacher):
        delta.index_add_(0, teacher[:, 0], probabilities)
        delta.index_add_(0, teacher[:, 1], -probabilities)
        coverage_delta.index_add_(0, teacher[:, 0], torch.ones(len(teacher), dtype=torch.long))
        coverage_delta.index_add_(0, teacher[:, 1], -torch.ones(len(teacher), dtype=torch.long))
    coverage = coverage_delta.cumsum(0)[:-1]
    covered = coverage > 0
    uncovered_nonwhite = [i for i, character in enumerate(text) if not character.isspace() and not covered[i]]
    if uncovered_nonwhite:
        raise ValueError(f"teacher omitted {len(uncovered_nonwhite)} non-whitespace characters")
    character_probs = delta.cumsum(0)[:-1].clamp_min(0) / coverage.clamp_min(1)[:, None]
    character_probs[~covered] = 0
    prefix = torch.cat((torch.zeros(1, probabilities.shape[1], dtype=torch.double), character_probs.cumsum(0)))
    counts = torch.cat((torch.zeros(1, dtype=torch.long), covered.long().cumsum(0)))
    pooled = prefix[student[:, 1]] - prefix[student[:, 0]]
    mass = counts[student[:, 1]] - counts[student[:, 0]]
    mask = mass > 0
    pooled = pooled.clamp_min(0)
    pooled[mask] /= pooled[mask].sum(-1, keepdim=True)
    pooled[~mask] = 0
    return pooled.float(), mask, {"raw_characters": len(text),
        "teacher_covered_characters": int(covered.sum()),
        "uncovered_whitespace_characters": int((~covered).sum()),
        "overlapping_byte_piece_characters": int((coverage > 1).sum()),
        "student_tokens": len(student), "student_kd_tokens": int(mask.sum())}


def distillation_loss_sum(logits, targets, mask, temperature=2.):
    """KL numerator includes T²; caller globally normalizes valid owned tokens."""
    logp = type_log_probabilities(logits, temperature)
    targets = torch.as_tensor(targets, dtype=torch.float32, device=logits.device).detach()
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if targets.shape != logp.shape or mask.shape != logp.shape[:-1]:
        raise ValueError("KD target/mask shapes differ from student predictions")
    real = targets[mask]
    if not torch.isfinite(real).all() or (real < 0).any() or not torch.allclose(
            real.sum(-1), torch.ones(len(real), device=real.device), atol=2e-3, rtol=0):
        raise ValueError("invalid cached KD probabilities")
    # Cache storage may be FP16; normalize its quantization error before KL.
    real = real / real.sum(-1, keepdim=True)
    numerator = F.kl_div(logp[mask], real, reduction='sum') * temperature_value(temperature)**2
    return numerator, mask.sum()


def attach_training_cache(directory, manifest, tags, documents, rows, temperature):
    """Validate provenance and attach targets to owned training positions only."""
    directory = Path(directory)
    metadata = json.loads((directory/'manifest.json').read_text())
    if metadata.get('status') != 'completed' or metadata.get('partition') != 'train':
        raise ValueError("KD cache must be a completed train-only cache")
    if metadata.get('teacher_frozen') is not True or not metadata.get('source_bindings'):
        raise ValueError("KD cache needs a frozen teacher and explicit source bindings")
    if metadata['student_tokenizer_sha256'] != manifest['config']['tokenizer_sha256']:
        raise ValueError("KD student tokenizer differs")
    if metadata['type_labels'] != label_order(tags) or metadata['temperature'] != temperature_value(temperature):
        raise ValueError("KD labels or temperature differ")
    if sorted(metadata['document_ids']) != sorted(manifest['document_ids']['train']):
        raise ValueError("KD cache IDs differ from the frozen training split")
    if {doc['document_id'] for doc in documents} != set(metadata['document_ids']):
        raise ValueError("KD attachment must receive exactly the training documents")
    if metadata['teacher_parameters'] <= metadata['student_parameters']:
        raise ValueError("KD requires a larger teacher and smaller student")
    for binding in metadata['source_bindings']:
        if file_sha256(binding['path']) != binding['sha256']:
            raise ValueError("KD source binding changed")
    cache_path = directory/'targets.pt'
    if file_sha256(cache_path) != metadata['targets_sha256']:
        raise ValueError("KD target cache hash differs")
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    if set(cache) != set(metadata['document_ids']):
        raise ValueError("KD cache contains missing or extra document IDs")
    for document in documents:
        key = document['document_id']
        target = cache[key]
        identity = {name: identity_sha256(document[field]) for name, field in (
            ('raw_text_sha256', 'text'), ('token_ids_sha256', 'token_ids'), ('offsets_sha256', 'token_offsets'))}
        if any(target.get(name) != value for name, value in identity.items()):
            raise ValueError("KD cache does not match the exact raw document and student encoding")
        probabilities, mask = target['probabilities'].float(), target['mask']
        if probabilities.shape != (len(document['token_ids']), len(metadata['type_labels'])) or (
                mask.shape != (len(document['token_ids']),) or mask.dtype != torch.bool):
            raise ValueError("invalid KD cache dimensions")
        for enabled, (left, right) in zip(mask.tolist(), document['token_offsets']):
            if enabled and left == right or not enabled and any(not c.isspace() for c in document['text'][left:right]):
                raise ValueError("KD validity mask omitted real non-whitespace or enabled a zero-width token")
        # Validate all valid targets, not only whichever window is first visited.
        valid = probabilities[mask]
        if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or not torch.allclose(
                valid.sum(-1), torch.ones(len(valid)), atol=2e-3, rtol=0):
            raise ValueError("invalid KD cache probability values")
    by_key = {doc['document_id']: doc for doc in documents}
    count = 0
    for row in rows:
        doc = by_key[row['document_id']]
        window = doc['windows'][row['window_number']]
        a, b = window['token_start'], window['token_end']
        target = cache[row['document_id']]
        row['kd_probabilities'] = target['probabilities'][a:b].float()
        row['kd_mask'] = target['mask'][a:b] & (torch.tensor(row['targets']) != -100)
        row['kd_normalizer'] = int(row['kd_mask'].sum())
        count += row['kd_normalizer']
    expected = sum(int(item['mask'].sum()) for item in cache.values())
    if count != expected:
        raise ValueError("KD supervision dropped or duplicated owned real tokens")
    return {"cache_directory": str(directory.resolve()), "cache_manifest_sha256": file_sha256(directory/'manifest.json'),
            "targets_sha256": metadata['targets_sha256'], "temperature": metadata['temperature'],
            "type_labels": metadata['type_labels'], "teacher_parameters": metadata['teacher_parameters'],
            "student_parameters": metadata['student_parameters'], "kd_tokens_per_epoch": count,
            "partition": 'train', "teacher_frozen": True,
            "alignment": 'teacher token types -> equal-weight raw Unicode characters -> student token types'}
