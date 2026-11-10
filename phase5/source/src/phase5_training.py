"""Whole-document LSI training, validation-only calibration, and resumable state."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase2_adaptation import frozen_parameter_sha256
from src.phase3_model import load_backbone
from src.phase3_rebuilt_model import load_rebuilt_backbone
from src.phase4_model import trainability
from src.phase4_training import (append, load, save, schedule_factor, schedule_settings,
                                 verify_bindings)
from src.phase5_model import StatuteClassifier, backward_document, predict_document
from src.phase5_prepare import binary_target
from src.phase5_tasks import (StatuteData, choose_threshold, metrics, package_submission,
                              predictions, stopping_update)

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def build(config, device):
    if config['architecture'] in ('dense', 'encoder_decoder'):
        backbone = load_rebuilt_backbone(config['checkpoint'], config['model_config'], config['checkpoint_sha256'])
    else:
        if file_sha256(Path(config['checkpoint'])) != config['checkpoint_sha256']:
            raise ValueError('Original Phase1 checkpoint hash changed')
        backbone = load_backbone(config['checkpoint'], config['model_config'])
    model = StatuteClassifier(backbone, config['architecture'], config['method'], config['compute_precision'],
                              config['partial_layers'], config['rank'], config['alpha'], config['head_dropout'])
    return model.to(device)


def make_optimizer(model, config):
    groups = [
        {'params': [p for name, p in model.named_parameters() if p.requires_grad and name.startswith('backbone.')],
         'lr': config['learning_rate'], 'peak': config['learning_rate']},
        {'params': list(model.classifier.parameters()),
         'lr': config['head_learning_rate'], 'peak': config['head_learning_rate']},
    ]
    return torch.optim.AdamW(groups, weight_decay=config['weight_decay'])


@torch.inference_mode()
def infer(model, data, split, config, progress_path=None):
    model.eval()
    values = []
    for index, row in enumerate(data.rows[split]):
        logits = predict_document(model, data.chunks(split, index), config['chunk_microbatch'])
        values.append(logits.float().cpu().numpy())
        if progress_path is not None and index % 256 == 0:
            atomic_json(progress_path, {'stage': 'complete_document_inference', 'split': split,
                                       'documents_done': index + 1, 'documents_total': len(data.rows[split])})
    return np.stack(values)


def sigmoid(logits):
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


def write_array(path, array):
    with Path(path).open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def evaluate(model, data, config, output, epoch):
    folder = Path(output) / f'dev_epoch{epoch:03d}_{time.time_ns()}'
    folder.mkdir(exist_ok=False)
    logits = infer(model, data, 'dev', config, folder / 'progress.json')
    probabilities = sigmoid(logits)
    gold = np.stack([binary_target(row['labels']) for row in data.rows['dev']]).astype(bool)
    threshold, grid = choose_threshold(probabilities, gold)
    predicted = probabilities >= threshold
    result = metrics(gold, predicted)
    result.update(epoch=epoch, threshold=threshold, threshold_search=grid,
                  fixed_threshold_05=metrics(gold, probabilities >= .5),
                  all_cases_all_chunks=True, mean_supervised_BCE=float(
                      torch.nn.functional.binary_cross_entropy_with_logits(
                          torch.from_numpy(logits), torch.from_numpy(gold.astype(np.float32)))) )
    train_hashes = {row['text_sha256'] for row in data.rows['train']}
    unseen = np.asarray([row['text_sha256'] not in train_hashes for row in data.rows['dev']])
    result['unseen_training_text_dev'] = metrics(gold[unseen], predicted[unseen]) if unseen.any() else None
    result['unseen_diagnostic_note'] = 'Same threshold selected on complete official dev; this subset is diagnostic, not an independent calibration split'
    write_array(folder / 'logits.npy', logits)
    atomic_json(folder / 'prediction.json', predictions(data.rows['dev'], probabilities, threshold))
    atomic_json(folder / 'truth.json', [{'id': row['id'], 'gold': np.flatnonzero(target).astype(int).tolist()}
                                      for row, target in zip(data.rows['dev'], gold)])
    result['truth_export_note'] = 'Unique statute IDs in evaluator truth, matching binary target semantics; audited raw label lists are unchanged'
    command = [sys.executable, str(ROOT / 'official/evaluation_scripts/evaluate_phase5.py'),
               '--truth', str(folder / 'truth.json'), '--prediction', str(folder / 'prediction.json')]
    official = subprocess.run(command, capture_output=True, text=True,
                              env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
    with (folder / 'official_evaluator.txt').open('x') as stream:
        stream.write(official.stdout + official.stderr)
    if official.returncode:
        raise ValueError('Unmodified official Phase5 evaluator failed')
    score = float(next(line.split(':')[1] for line in official.stdout.splitlines() if line.startswith('Macro-F1:')))
    if abs(score - result['macro_f1']) > 5.1e-7:
        raise ValueError('Local and official Macro-F1 disagree')
    atomic_json(folder / 'metrics.json', result)
    result['metrics_path'] = str(folder / 'metrics.json')
    return result


def teacher_targets(config, data):
    if config['method'] != 'knowledge_distillation':
        return None
    manifest = read_json(config['teacher_manifest'])
    for path, digest in manifest['bindings'].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError('Bound teacher input/cache/checkpoint changed')
    if (manifest['partition'] != 'train' or not manifest['teacher_frozen']
            or manifest['data_manifest_sha256'] != file_sha256(Path(config['data_manifest']))
            or manifest['temperature'] != config['temperature']
            or manifest['train_row_keys'] != [row['row_key'] for row in data.rows['train']]
            or manifest['train_ids'] != [row['id'] for row in data.rows['train']]
            or manifest['teacher_backbone_parameters'] <= manifest['student_backbone_parameters']):
        raise ValueError('Teacher/student identity, size or row alignment mismatch')
    values = np.load(manifest['probabilities'], allow_pickle=False)
    if values.shape != (len(data.rows['train']), 100) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Invalid independent sigmoid teacher probabilities')
    return values


def check_teacher_gradient(model, chunks, labels, soft, config):
    model.eval()
    device = next(model.parameters()).device
    from src.phase4_model import batches
    from src.phase5_model import document_loss
    ids, lengths = next(batches([chunks[0][:64]], 1, device))
    logits = model.classifier(model.chunk_vectors(ids, lengths)[0])
    loss = document_loss(logits, labels, soft, config['temperature'], 1.)[0]
    gradient = torch.autograd.grad(loss, model.classifier[-1].weight)[0]
    record = {'teacher_signal_loss': float(loss.detach()), 'head_gradient_norm': float(gradient.norm()),
              'actual_cached_teacher_used': True, 'training_rows_only': True}
    if not math.isfinite(record['teacher_signal_loss']) or not math.isfinite(record['head_gradient_norm']) or record['head_gradient_norm'] <= 0:
        raise ValueError('Actual teacher signal is not finite/nonzero')
    return record


def train(config_path, device, resume=False):
    config = read_json(config_path)
    verify_bindings(config)
    output = Path(config['output']) / 'production'
    approval = read_json(Path(config['output']) / 'pilot/completed.json')
    if not approval['passed'] or approval['config_sha256'] != file_sha256(Path(config_path)):
        raise ValueError('The matching numerical/recovery pilot must pass before production')
    output.mkdir(exist_ok=resume)
    data = StatuteData(config['data_manifest'])
    model = build(config, device)
    atomic_json(output / 'trainability.json', trainability(model))
    frozen = frozen_parameter_sha256(model)
    optimizer = make_optimizer(model, config)
    schedule = schedule_settings(config, len(data.rows['train']))
    digest = file_sha256(Path(config_path))
    progress = {'epoch': 0, 'next_batch': 0, 'updates': 0, 'best_score': None,
                'best_threshold': None, 'best_epoch': None, 'bad_epochs': 0,
                'epoch_loss_sum': 0., 'epoch_cases': 0, 'schedule': schedule, 'stopped_early': False}
    if resume:
        progress = load(output / 'latest.pt', model, optimizer, digest)
        if progress['schedule'] != schedule:
            raise ValueError('Saved schedule differs; never silently restart warmup')
    teacher = teacher_targets(config, data)
    if teacher is not None:
        record = check_teacher_gradient(model, data.chunks('train', 0), data.rows['train'][0]['labels'], teacher[0], config)
        atomic_json(output / f'KD_signal_{time.time_ns()}.json', record)
    atomic_json(output / f'train_request_{time.time_ns()}.json', {
        'resume': resume, 'max_epochs': config['max_epochs'], 'patience': config['patience'],
        'config_sha256': digest, 'schedule': schedule, 'starts_from': 'latest' if resume else 'own_Phase1'})
    log = output / (f'updates_resume_{time.time_ns()}.jsonl' if resume else 'updates.jsonl')
    started = time.perf_counter()
    if not progress['stopped_early']:
        for epoch in range(progress['epoch'], config['max_epochs']):
            order = np.random.default_rng(config['seed'] + epoch).permutation(len(data.rows['train'])).tolist()
            case_batches = [order[start:start + config['cases_per_update']]
                            for start in range(0, len(order), config['cases_per_update'])]
            for number in range(progress['next_batch'], len(case_batches)):
                selected = case_batches[number]
                step = progress['updates'] + 1
                factor = schedule_factor(schedule, step)
                for group in optimizer.param_groups:
                    group['lr'] = group['peak'] * factor
                optimizer.zero_grad(set_to_none=True)
                model.train()
                tick = time.perf_counter()
                losses = [backward_document(model, data.chunks('train', index), data.rows['train'][index]['labels'],
                                            len(selected), config['chunk_microbatch'],
                                            None if teacher is None else teacher[index],
                                            config['temperature'], config['kd_weight']) for index in selected]
                norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                            config['clip_norm'], error_if_nonfinite=True))
                if norm > config['maximum_gradient_norm']:
                    raise ValueError('Finite gradient exceeds the announced safety limit')
                optimizer.step()
                total_loss = sum(row['loss'] for row in losses)
                progress.update(epoch=epoch, next_batch=number + 1, updates=step,
                                epoch_loss_sum=progress['epoch_loss_sum'] + total_loss,
                                epoch_cases=progress['epoch_cases'] + len(selected))
                append(log, {'epoch': epoch + 1, 'batch': number + 1, 'updates': step,
                             'cases': len(selected), 'chunks': sum(row['chunks'] for row in losses),
                             'mean_document_loss': total_loss / len(selected),
                             'mean_supervised_BCE': sum(row['supervised_BCE'] for row in losses) / len(selected),
                             'mean_distillation_BCE_T2': sum(row['distillation_BCE_T2'] for row in losses) / len(selected),
                             'gradient_norm': norm, 'backbone_learning_rate': optimizer.param_groups[0]['lr'],
                             'head_learning_rate': optimizer.param_groups[1]['lr'], 'seconds': time.perf_counter() - tick})
                if step % config['checkpoint_every_updates'] == 0:
                    save(output, model, optimizer, progress, digest, False)
                    atomic_json(output / 'progress.json', {**progress, 'stage': 'training',
                                                          'elapsed_seconds': time.perf_counter() - started})
            if progress['epoch_cases'] != len(data.rows['train']):
                raise ValueError('Epoch failed full official training-row coverage')
            atomic_json(output / 'progress.json', {**progress, 'stage': 'official_dev_evaluation'})
            result = evaluate(model, data, config, output, epoch + 1)
            improved, stop = stopping_update(progress, result['macro_f1'], result['threshold'], epoch + 1,
                                             config['min_epochs'], config['patience'])
            append(output / 'epochs.jsonl', {'epoch': epoch + 1,
                                            'mean_train_document_loss': progress['epoch_loss_sum'] / progress['epoch_cases'],
                                            'dev_macro_f1': result['macro_f1'], 'dev_micro_f1': result['micro_f1'],
                                            'threshold': result['threshold'], 'improved': improved,
                                            'metrics_path': result['metrics_path']})
            progress.update(epoch=epoch + 1, next_batch=0, epoch_loss_sum=0., epoch_cases=0, stopped_early=stop)
            if frozen_parameter_sha256(model) != frozen:
                raise ValueError('Declared frozen pretrained parameters changed')
            save(output, model, optimizer, progress, digest, improved)
            atomic_json(output / 'progress.json', {**progress, 'stage': 'epoch_complete', 'latest_dev': result})
            if stop:
                break
    if progress['best_epoch'] is None:
        raise ValueError('No validation-selected checkpoint exists')
    atomic_json(output / 'progress.json', {**progress, 'stage': 'best_checkpoint_test_inference'})
    best = load(output / 'best.pt', model, None, digest, restore=False)
    if best['best_epoch'] != progress['best_epoch'] or best['best_threshold'] != progress['best_threshold']:
        raise ValueError('Best checkpoint and threshold selection state disagree')
    logits = infer(model, data, 'test', config, output / 'inference_progress.json')
    records = predictions(data.rows['test'], sigmoid(logits), best['best_threshold'])
    export = output / 'exports' / f'epoch{progress["epoch"]:03d}_{time.time_ns()}'
    export.mkdir(parents=True, exist_ok=False)
    atomic_json(export / 'prediction.json', records)
    archive = package_submission(export / 'submission.zip', records, [row['id'] for row in data.rows['test']])
    write_array(export / 'test_logits.npy', logits)
    verify_bindings(config)
    result = {'status': 'completed', 'name': config['name'], 'architecture': config['architecture'],
              'method': config['method'], 'epochs': progress['epoch'], 'stopped_early': progress['stopped_early'],
              'best_epoch': progress['best_epoch'], 'best_dev_macro_f1': progress['best_score'],
              'threshold': best['best_threshold'], 'updates': progress['updates'],
              'checkpoint': str(output / 'best.pt'), 'checkpoint_sha256': file_sha256(output / 'best.pt'),
              'latest_checkpoint_sha256': file_sha256(output / 'latest.pt'), 'submission': archive,
              'elapsed_seconds': time.perf_counter() - started, 'all_cases_all_chunks': True,
              'raw_splits_unchanged': True, 'test_labels_or_metadata_used': False, 'uploaded': False,
              'frozen_parameters_unchanged': True, 'config_sha256': digest}
    atomic_json(output / f'completed_history_{time.time_ns()}.json', result)
    atomic_json(output / 'completed.json', result)
    return result


def cache_teacher(campaign_path, device):
    campaign_path = Path(campaign_path)
    campaign = read_json(campaign_path / 'campaign.json')
    teacher_path = Path(campaign['configs']['mha_full']['path'])
    student = read_json(campaign['configs']['mqa_kd']['path'])
    config = read_json(teacher_path)
    verify_bindings(config)
    completed = read_json(Path(config['output']) / 'production/completed.json')
    checkpoint = Path(config['output']) / 'production/best.pt'
    if file_sha256(checkpoint) != completed['checkpoint_sha256']:
        raise ValueError('Selected teacher checkpoint changed')
    model = build(config, device)
    load(checkpoint, model, None, file_sha256(teacher_path), restore=False)
    model.eval().requires_grad_(False)
    data = StatuteData(config['data_manifest'])
    folder = campaign_path / 'teacher'
    folder.mkdir(exist_ok=False)
    logits = infer(model, data, 'train', config, folder / 'progress.json')
    values = sigmoid(logits / student['temperature']).astype(np.float32)
    path = folder / 'probabilities.npy'
    write_array(path, values)
    teacher_size = campaign['backbones']['mha']['parameters']
    student_size = campaign['backbones']['mqa']['parameters']
    if teacher_size <= student_size:
        raise ValueError('Teacher must be larger than the student')
    record = {'probabilities': str(path), 'temperature': student['temperature'],
              'teacher_backbone_parameters': teacher_size, 'student_backbone_parameters': student_size,
              'train_ids': [row['id'] for row in data.rows['train']],
              'train_row_keys': [row['row_key'] for row in data.rows['train']],
              'partition': 'train', 'teacher_frozen': True,
              'data_manifest_sha256': file_sha256(Path(config['data_manifest'])),
              'teacher_best_epoch': completed['best_epoch'], 'teacher_dev_macro_f1': completed['best_dev_macro_f1'],
              'probability_semantics': '100 independent sigmoid(logit/temperature) targets; no softmax',
              'dev_or_test_teacher_targets_generated': False,
              'bindings': {str(path): file_sha256(path), str(teacher_path): file_sha256(teacher_path),
                           str(checkpoint): completed['checkpoint_sha256']}}
    atomic_json(folder / 'manifest.json', record)
    # Validate the actual finished cache using the student's exact input manifest.
    teacher_targets(student, data)
    return {'status': 'teacher_cache_complete', 'training_rows': len(values),
            'labels': 100, 'teacher_best_epoch': completed['best_epoch']}
