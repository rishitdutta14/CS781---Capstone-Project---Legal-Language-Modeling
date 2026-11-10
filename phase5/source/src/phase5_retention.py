"""Change artifact retention without changing a saved training trajectory."""

import copy
import json
import os
from pathlib import Path

import torch

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase5_extend import checkpoint_content_digest


def verify_retention_change(original, revised):
    changed = {key for key in set(original) | set(revised) if original.get(key) != revised.get(key)}
    allowed = {'bindings', 'export_each_validation_improvement', 'resume_recovery_check', 'retention_revision'}
    if not changed <= allowed or revised['export_each_validation_improvement'] is not False:
        raise ValueError('Only artifact-retention and recovery provenance may change')
    return sorted(changed)


def rebind_checkpoint(path, old_digest, new_digest):
    """Rebind only config identity; preserve model, optimizer, RNG and progress."""
    path = Path(path)
    metadata_path = path.with_suffix('.metadata.json')
    metadata = json.loads(metadata_path.read_text())
    if metadata['config_sha256'] != old_digest or file_sha256(path) != metadata['sha256']:
        raise ValueError('Original checkpoint identity/checksum mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['config_sha256'] != old_digest or saved['progress'] != metadata['progress']:
        raise ValueError('Original checkpoint payload/metadata mismatch')
    progress = copy.deepcopy(saved['progress'])
    content = checkpoint_content_digest(saved)
    saved['config_sha256'] = new_digest
    partial = path.with_name(path.name + '.retention.partial')
    backup = path.with_name(path.name + '.retention.backup')
    if partial.exists() or backup.exists():
        raise FileExistsError('Preserve interrupted retention migration for inspection')
    with partial.open('xb') as stream:
        torch.save(saved, stream)
        stream.flush()
        os.fsync(stream.fileno())
    restored = torch.load(partial, map_location='cpu', weights_only=False)
    if checkpoint_content_digest(restored) != content or restored['progress'] != progress:
        raise ValueError('Retention migration changed training state')
    checksum = file_sha256(partial)
    os.link(path, backup)
    partial.replace(path)
    atomic_json(metadata_path, {**metadata, 'sha256': checksum, 'config_sha256': new_digest,
        'retention_source_sha256': metadata['sha256']})
    if file_sha256(path) != checksum:
        raise ValueError('Migrated checkpoint checksum failed; retain backup')
    backup.unlink()
    return {'old_sha256': metadata['sha256'], 'new_sha256': checksum,
            'training_state_sha256': content, 'progress': progress,
            'only_checkpoint_payload_change': 'config_sha256'}
