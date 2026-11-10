"""Audited continuation into new folders; never rewrites the five-epoch stage."""

import copy
import json
import os
from pathlib import Path
import shutil
import time

import torch

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase1_full_training import state_digest
from src.phase4_training import load, schedule_settings, verify_bindings
from src.phase5_checks import recovery_gate
from src.phase5_tasks import StatuteData
from src.phase5_training import build, make_optimizer, read_json, teacher_targets


ALLOWED_CHANGES = {'max_epochs', 'patience', 'output', 'campaign', 'bindings',
                   'continuation_parent_config', 'continuation_parent_config_sha256',
                   'continuation_policy_sha256'}


def verify_policy_change(original, extended):
    changed = {key for key in set(original) | set(extended) if original.get(key) != extended.get(key)}
    if not changed <= ALLOWED_CHANGES:
        raise ValueError('Extension changes something other than budget, patience or artifact/provenance paths')
    if extended['max_epochs'] != 20 or extended['patience'] != 3:
        raise ValueError('User authorized a20-epoch cap and patience3')
    if original['lr_schedule'] != 'warmup_constant':
        raise ValueError('This continuation requires an endpoint-independent schedule')
    if original['teacher_manifest'] != extended['teacher_manifest']:
        raise ValueError('Never change a running student\'s frozen teacher')
    if original['output'] == extended['output']:
        raise ValueError('Preserve the original five-epoch stage in a separate folder')
    return sorted(changed)


def rearm_progress(progress, minimum_epochs=2, patience=3):
    result = copy.deepcopy(progress)
    # Keep the existing count: two bad epochs under the old policy are still two
    # bad epochs, not a fresh allowance of three more.
    result['stopped_early'] = result['epoch'] >= minimum_epochs and result['bad_epochs'] >= patience
    return result


def checkpoint_content_digest(saved):
    progress = {key: value for key, value in saved['progress'].items() if key != 'stopped_early'}
    numpy_state = saved['rng']['numpy']
    rng = {**saved['rng'], 'numpy': (numpy_state[0], torch.as_tensor(numpy_state[1]), *numpy_state[2:])}
    return state_digest({'model': saved['model'], 'optimizer': saved['optimizer'],
                         'rng': rng, 'progress': progress})


def migrate_checkpoint(source, destination, original_digest, new_digest, schedule,
                       minimum_epochs=2, patience=3):
    source, destination = Path(source), Path(destination)
    metadata = read_json(source.with_suffix('.metadata.json'))
    if file_sha256(source) != metadata['sha256'] or metadata['config_sha256'] != original_digest:
        raise ValueError('Source checkpoint checksum/configuration mismatch')
    saved = torch.load(source, map_location='cpu', weights_only=False)
    if saved['config_sha256'] != original_digest or saved['progress'] != metadata['progress']:
        raise ValueError('Source checkpoint payload and metadata disagree')
    if saved['progress']['schedule'] != schedule:
        raise ValueError('Migration would change the existing learning-rate schedule')
    if saved['progress']['epoch'] > 20:
        raise ValueError('Cannot migrate a checkpoint already beyond epoch20')
    protected = checkpoint_content_digest(saved)
    original_stopped = saved['progress']['stopped_early']
    saved['progress'] = rearm_progress(saved['progress'], minimum_epochs, patience)
    saved['config_sha256'] = new_digest
    if checkpoint_content_digest(saved) != protected:
        raise ValueError('Migration changed training state, weights, RNG or optimizer')
    if destination.exists() or destination.with_suffix('.partial.pt').exists():
        raise FileExistsError('Preserve prior migration artifacts')
    temporary = destination.with_suffix('.partial.pt')
    with temporary.open('xb') as stream:
        torch.save(saved, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    checksum = file_sha256(destination)
    atomic_json(destination.with_suffix('.metadata.json'), {
        'sha256': checksum, 'progress': saved['progress'], 'config_sha256': new_digest,
        'migrated_from': str(source), 'source_sha256': metadata['sha256']})
    return {'source': str(source), 'source_sha256': metadata['sha256'],
            'destination': str(destination), 'destination_sha256': checksum,
            'model_optimizer_RNG_and_progress_content_sha256': protected,
            'epoch': saved['progress']['epoch'], 'next_batch': saved['progress']['next_batch'],
            'updates': saved['progress']['updates'], 'bad_epochs_carried_over': saved['progress']['bad_epochs'],
            'previous_stopped_early': original_stopped, 'new_stopped_early': saved['progress']['stopped_early'],
            'only_payload_changes': ['config_sha256', 'progress.stopped_early'],
            'learning_rate_schedule_unchanged': True}


def migrate_candidate(config_path):
    config_path = Path(config_path)
    config = read_json(config_path)
    parent_path = Path(config['continuation_parent_config'])
    if file_sha256(parent_path) != config['continuation_parent_config_sha256']:
        raise ValueError('Parent configuration changed')
    original = read_json(parent_path)
    changed = verify_policy_change(original, config)
    verify_bindings(config)
    source = Path(original['output']) / 'production'
    completed = read_json(source / 'completed.json')
    if completed['status'] != 'completed' or completed['config_sha256'] != file_sha256(parent_path):
        raise ValueError('Wait for the original stage to finish cleanly before migrating')
    if file_sha256(source / 'best.pt') != completed['checkpoint_sha256']:
        raise ValueError('Original completed best checkpoint changed')
    output = Path(config['output']) / 'production'
    output.mkdir(exist_ok=False)
    data = read_json(config['data_manifest'])
    schedule = schedule_settings(config, data['splits']['train']['documents'])
    digest = file_sha256(config_path)
    records = {name: migrate_checkpoint(source / f'{name}.pt', output / f'{name}.pt',
                                        file_sha256(parent_path), digest, schedule,
                                        config['min_epochs'], config['patience'])
               for name in ('latest', 'best')}
    for filename in ('epochs.jsonl',):
        if (source / filename).exists():
            shutil.copy2(source / filename, output / filename)
    result = {'status': 'migrated_not_yet_resumed', 'changed_configuration_fields': changed,
              'source_completed_record': str(source / 'completed.json'),
              'source_completed_sha256': file_sha256(source / 'completed.json'),
              'checkpoints': records, 'original_artifacts_modified': False,
              'frozen_teacher_manifest_unchanged': config['teacher_manifest'],
              'policy_sha256': config['continuation_policy_sha256']}
    atomic_json(Path(config['output']) / 'migration.json', result)
    latest = read_json(output / 'latest.metadata.json')['progress']
    atomic_json(output / 'progress.json', {**latest, 'stage': 'migrated_not_yet_resumed'})
    return result


def recovery(config_path, device):
    config = read_json(config_path)
    verify_bindings(config)
    model = build(config, device)
    optimizer = make_optimizer(model, config)
    output = Path(config['output']) / 'production'
    digest = file_sha256(Path(config_path))
    progress = load(output / 'latest.pt', model, optimizer, digest)
    data = StatuteData(config['data_manifest'])
    teacher = teacher_targets(config, data)
    result = recovery_gate(output, model, optimizer, digest,
                           [data.chunks('train', 0)[0][:64]], data.rows['train'][0]['labels'])
    result.update(epoch_resumed_from=progress['epoch'], next_batch=progress['next_batch'],
                  updates_preserved=progress['updates'], bad_epochs_preserved=progress['bad_epochs'],
                  learning_rate_schedule=progress['schedule'],
                  teacher_cache_verified=teacher is not None,
                  note='Diagnostic updates are rolled back; production reloads the exact migrated latest checkpoint')
    atomic_json(Path(config['output']) / f'extension_recovery_{time.time_ns()}.json', result)
    if not result['passed']:
        raise ValueError('Migrated exact next-update recovery failed; no continued production')
    return result
