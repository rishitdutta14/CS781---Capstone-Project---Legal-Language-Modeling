"""Resume the completed 20-cap stage to50, preserving patience history."""

from pathlib import Path
import shutil

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase4_training import schedule_settings, verify_bindings
from src.phase5_extend import ALLOWED_CHANGES, migrate_checkpoint
from src.phase5_training import read_json


def verify_policy_change(original, extended):
    changed = {key for key in set(original) | set(extended) if original.get(key) != extended.get(key)}
    if not changed <= ALLOWED_CHANGES:
        raise ValueError('Only stopping settings and continuation artifact/provenance paths may change')
    if original['max_epochs'] != 20 or extended['max_epochs'] != 50 or extended['patience'] != 5:
        raise ValueError('This policy continues the20-cap stage to50 with patience5')
    if original['lr_schedule'] != 'warmup_constant':
        raise ValueError('Never alter the learning-rate horizon during continuation')
    if original['teacher_manifest'] != extended['teacher_manifest']:
        raise ValueError('The student must retain the same frozen teacher')
    if original['output'] == extended['output']:
        raise ValueError('Preserve the20-cap artifacts in their original folder')
    return sorted(changed)


def migrate_candidate(config_path):
    config_path = Path(config_path)
    config = read_json(config_path)
    source_config = Path(config['continuation_parent_config'])
    if file_sha256(source_config) != config['continuation_parent_config_sha256']:
        raise ValueError('Source20-cap configuration changed')
    original = read_json(source_config)
    changed = verify_policy_change(original, config)
    verify_bindings(config)
    source = Path(original['output']) / 'production'
    completed = read_json(source / 'completed.json')
    old_digest = file_sha256(source_config)
    if completed['status'] != 'completed' or completed['config_sha256'] != old_digest:
        raise ValueError('The source20-cap stage has not completed cleanly')
    if file_sha256(source / 'best.pt') != completed['checkpoint_sha256']:
        raise ValueError('The source validation-selected checkpoint changed')
    output = Path(config['output']) / 'production'
    output.mkdir(exist_ok=False)
    manifest = read_json(config['data_manifest'])
    schedule = schedule_settings(config, manifest['splits']['train']['documents'])
    digest = file_sha256(config_path)
    # The existing audited migration accepts source epochs<=20, preserves every
    # nonimprovement count, and only rearms stopped_early under the new limit.
    records = {name: migrate_checkpoint(source / f'{name}.pt', output / f'{name}.pt',
                                        old_digest, digest, schedule, config['min_epochs'], 5)
               for name in ('latest', 'best')}
    if (source / 'epochs.jsonl').exists():
        shutil.copy2(source / 'epochs.jsonl', output / 'epochs.jsonl')
    result = {'status': 'migrated_not_yet_resumed', 'max_epochs_total': 50, 'patience': 5,
              'bad_epochs_reset': False, 'changed_configuration_fields': changed,
              'source_completed_record': str(source / 'completed.json'),
              'source_completed_sha256': file_sha256(source / 'completed.json'),
              'checkpoints': records, 'original_artifacts_modified': False,
              'frozen_teacher_manifest_unchanged': config['teacher_manifest'],
              'policy_sha256': config['continuation_policy_sha256']}
    atomic_json(Path(config['output']) / 'migration.json', result)
    latest = read_json(output / 'latest.metadata.json')['progress']
    atomic_json(output / 'progress.json', {**latest, 'stage': 'migrated_not_yet_resumed'})
    return result
