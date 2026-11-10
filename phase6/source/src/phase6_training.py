"""Summary-target-only training, scoped RNG recovery and one final test export."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase1_full_corpus_encode import CachedDocumentEncoder
from src.phase2_adaptation import frozen_parameter_sha256, parameter_summary
from src.phase6_data import SummaryData, example, validate_predictions
from src.phase6_model import build


def append(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()


def verify(config):
    for path, digest in config['bindings'].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError('Bound Phase6 source/input changed: ' + path)


def rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(0) if torch.cuda.is_available() else None}


def restore(saved):
    random.setstate(saved['python']); np.random.set_state(saved['numpy']); torch.set_rng_state(saved['cpu'])
    if saved['cuda'] is not None: torch.cuda.set_rng_state(saved['cuda'], 0)


def save(output, model, optimizer, progress, digest, improved):
    output = Path(output)
    partial = output / 'checkpoint.partial.pt'
    if partial.exists(): raise FileExistsError('Preserve interrupted checkpoint write')
    with partial.open('xb') as stream:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'progress': progress,
                    'rng': rng(), 'config_sha256': digest}, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(partial, output / 'latest.pt')
    metadata = {'sha256': file_sha256(output / 'latest.pt'), 'progress': progress, 'config_sha256': digest}
    atomic_json(output / 'latest.metadata.json', metadata)
    if improved:
        link = output / '.best.link'
        os.link(output / 'latest.pt', link); os.replace(link, output / 'best.pt')
        atomic_json(output / 'best.metadata.json', metadata)


def load(path, model, optimizer, digest, restore_state=True):
    path = Path(path); metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    if metadata['sha256'] != file_sha256(path) or metadata['config_sha256'] != digest:
        raise ValueError('Checkpoint SHA/config mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if saved['config_sha256'] != digest or saved['progress'] != metadata['progress']:
        raise ValueError('Checkpoint metadata/payload mismatch')
    model.load_state_dict(saved['model'], strict=True)
    if optimizer is not None: optimizer.load_state_dict(saved['optimizer'])
    if restore_state: restore(saved['rng'])
    return dict(saved['progress'])


def batch(data, split, jobs, config, device, alternate=None):
    rows = []
    for index, start, end, kind in jobs:
        target = alternate[index] if kind else data.reference(split, index)
        source = data.selected[split][index][0]
        rows.append(example(source, target, start, end, config['architecture'] == 'encdec251',
                            config['source_budget'], config['history']))
    length = max(len(row[1]) for row in rows)
    ids = torch.zeros(len(rows), length, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    source = None
    if rows[0][0] is not None:
        source = torch.tensor([row[0] for row in rows], dtype=torch.long, device=device)
    for i, (_, values, truth) in enumerate(rows):
        ids[i, :len(values)] = torch.tensor(values, device=device)
        labels[i, :len(truth)] = torch.tensor(truth, device=device)
    valid_positions = (labels != -100).nonzero()
    first = int(valid_positions[:, 1].min())
    return source, ids, labels[:, first:], first


def loss(model, data, split, jobs, config, device, alternate=None, teacher=None):
    source, ids, truth, first = batch(data, split, jobs, config, device, alternate)
    logits = model(ids, source=source, valid=ids != 0, output_start=first)[0]
    count = int((truth != -100).sum())
    total = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), truth.reshape(-1),
                            ignore_index=-100, reduction='sum')
    if teacher is not None:
        kd = total.new_zeros(())
        for i, (index, start, end, kind) in enumerate(jobs):
            if kind: raise ValueError('Token KD uses the original reference continuation only')
            offset = data.rows[split][index]['target_offset'] + index + start
            selected = truth[i] != -100
            kd = kd + partitioned_kl(logits[i][selected].float(),
                torch.tensor(np.asarray(teacher['ids'][offset:offset + end - start]).astype(np.int64), device=device),
                torch.tensor(np.asarray(teacher['probabilities'][offset:offset + end - start]).astype(np.float32), device=device),
                torch.tensor(np.asarray(teacher['tail'][offset:offset + end - start]), device=device),
                config['kd_temperature'])
        total = (1 - config['kd_weight']) * total + config['kd_weight'] * kd
    if not bool(torch.isfinite(total)) or count < 1:
        raise ValueError('Invalid summary-target loss')
    return total, count


@torch.inference_mode()
def evaluate(model, data, config, device):
    model.eval(); total = count = 0
    jobs = data.jobs('dev', config['target_block'])
    for start in range(0, len(jobs), config['microbatch']):
        value, n = loss(model, data, 'dev', jobs[start:start + config['microbatch']], config, device)
        total += float(value); count += n
    return {'summary_token_cross_entropy': total / count, 'valid_targets': count,
            'documents': len(data.rows['dev']), 'all_reference_tokens_and_EOS_covered': True}


@torch.inference_mode()
def generate(model, sources, config, device):
    """Batched cached decoding, with the same continuation refresh as training."""
    model.eval()
    b = len(sources); budget = config['source_budget']
    selected = [s[:budget] + [0] * (budget - len(s[:budget])) for s in sources]
    source = torch.tensor([[2] + s for s in selected], device=device)
    memory = model.encode(source) if model.encoder_decoder else None
    results = [[] for _ in sources]; done = [False] * b
    bans = [{} for _ in sources]
    caches = None; valid = None; position = 0; current = None
    for step in range(config['max_new_tokens']):
        if step % config['target_block'] == 0:
            previous = [r[-config['history']:] for r in results]
            longest = max(map(len, previous), default=0)
            # Active rows have equal generated lengths. Pad only inactive rows.
            prefixes = [[2] + p + [0] * (longest - len(p)) if model.encoder_decoder else
                        [2] + s + [3, 2] + p + [0] * (longest - len(p))
                        for s, p in zip(selected, previous)]
            current = torch.tensor(prefixes, device=device)
            valid = current != 0; caches = None; position = 0
        else:
            valid = torch.cat((valid, current != 0), dim=1)
        logits, caches = model(current, source=source if model.encoder_decoder else None,
                               memory=memory, caches=caches, start=position, valid=valid, last=True)
        position += current.shape[1]
        scores = logits[:, -1].float()
        scores[:, :3] = -torch.inf
        if step < config['min_new_tokens']: scores[:, 3] = -torch.inf
        for i, tokens in enumerate(results):
            if len(tokens) >= 3:
                blocked = bans[i].get(tuple(tokens[-3:]), ())
                if blocked: scores[i, list(blocked)] = -torch.inf
        chosen = scores.argmax(-1).tolist()
        for i, token in enumerate(chosen):
            if done[i]: continue
            if token == 3:
                done[i] = True
            else:
                results[i].append(token)
                if len(results[i]) >= 4:
                    prefix = tuple(results[i][-4:-1]); bans[i].setdefault(prefix, set()).add(token)
        if all(done): break
        current = torch.tensor([[token if not done[i] else 3] for i, token in enumerate(chosen)], device=device)
    return results, {'reached_length_limit': sum(not value for value in done), 'tokens': list(map(len, results))}


def decode(codec, tokens):
    payload = b''.join(codec.token_bytes[token] for token in tokens if token >= 4)
    # Generated byte sequences need not be valid UTF8; preserve invalid-byte
    # evidence rather than silently replacing a prediction with source text.
    text = ' '.join(payload.decode('utf-8', errors='replace').split())
    if not text: raise ValueError('Model generated an empty summary')
    return text


def export(model, data, config, device, output, split='test'):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    codec = CachedDocumentEncoder(config['tokenizer'])
    records = []; diagnostics = []; size = config['generation_batch']
    for start in range(0, len(data.rows[split]), size):
        chosen = list(range(start, min(start + size, len(data.rows[split]))))
        sources = [data.selected[split][i][0] for i in chosen]
        generated, stats = generate(model, sources, config, device)
        records.extend({'id': data.rows[split][i]['id'], 'summary': decode(codec, ids)}
                       for i, ids in zip(chosen, generated))
        diagnostics.append(stats)
        atomic_json(output / 'progress.json', {'stage': 'generating_summaries', 'split': split,
                    'documents': len(records), 'total': len(data.rows[split])})
    validate_predictions(records, [r['id'] for r in data.rows[split]])
    atomic_json(output / 'prediction.json', records)
    atomic_json(output / 'generation_diagnostics.json', {'batches': diagnostics,
        'UTF8_replacement_characters': sum(r['summary'].count('\ufffd') for r in records),
        'all_IDs_covered': True, 'source_selection': config['source_selection']})
    archive = output / 'submission.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr('prediction.json', json.dumps(records, ensure_ascii=False, allow_nan=False).encode())
    with zipfile.ZipFile(archive) as zipped:
        if zipped.namelist() != ['prediction.json'] or zipped.testzip() is not None:
            raise ValueError('Submission structure/CRC failed')
        validate_predictions(json.loads(zipped.read('prediction.json')), [r['id'] for r in data.rows[split]])
    result = {'path': str(archive), 'sha256': file_sha256(archive), 'predictions': len(records), 'split': split}
    atomic_json(output / 'completed.json', result)
    return result


def partitioned_kl(student, indices, probabilities, tail, temperature=2.):
    """KL over teacher top32 singleton bins and one aggregated remaining bin.

    This is explicitly a partitioned distribution, not a claim to full-vocab KL.
    Both probabilities and logits use temperature T; T squared preserves scale.
    """
    teacher = torch.cat((probabilities.float(), tail.float()[:, None]), dim=1)
    teacher = teacher / teacher.sum(-1, keepdim=True)
    log_student = F.log_softmax(student / temperature, dim=-1)
    chosen = log_student.gather(-1, indices)
    # A complement in log space prevents cancellation around high top-k mass.
    mass = chosen.logsumexp(-1).clamp(max=-1e-7)
    remaining = torch.log(-torch.expm1(mass))
    bins = torch.cat((chosen, remaining[:, None]), dim=-1)
    return (teacher * (teacher.clamp_min(1e-30).log() - bins)).sum() * temperature ** 2


def teacher_cache(config, data):
    if config['method'] != 'kd': return None
    manifest = json.loads(Path(config['teacher_manifest']).read_text())
    if (manifest['split'] != 'train' or manifest['temperature'] != config['kd_temperature']
            or manifest['tokenizer_sha256'] != config['tokenizer_sha256']
            or manifest['data_manifest_sha256'] != file_sha256(Path(config['data_manifest']))):
        raise ValueError('Training-only teacher cache changed')
    if manifest['training_IDs'] != [r['id'] for r in data.rows['train']]:
        raise ValueError('Teacher and student training rows differ')
    if manifest['teacher_parameters'] <= config['pretrained_parameters']:
        raise ValueError('Teacher must be larger than student')
    if (file_sha256(Path(config['teacher_checkpoint'])) != manifest['checkpoint_sha256']
            or set(manifest['files']) != {'ids', 'probabilities', 'tail'}):
        raise ValueError('Selected teacher checkpoint/cache files differ')
    result = {}
    for key, row in manifest['files'].items():
        if file_sha256(Path(row['path'])) != row['sha256']: raise ValueError('Teacher cache checksum changed')
        result[key] = np.load(row['path'], mmap_mode='r', allow_pickle=False)
    n = sum(row['target_tokens'] + 1 for row in data.rows['train'])
    topk = config['kd_topk']
    if (result['ids'].shape != (n, topk) or result['probabilities'].shape != (n, topk)
            or result['tail'].shape != (n,)):
        raise ValueError('Incomplete teacher target distribution coverage')
    for start in range(0, n, 65536):
        ids = result['ids'][start:start + 65536]
        p = result['probabilities'][start:start + 65536]
        tail = result['tail'][start:start + 65536]
        if (ids.min() < 0 or ids.max() >= 16000 or not np.isfinite(p).all()
                or not np.isfinite(tail).all() or (p < 0).any() or (tail < 0).any()
                or not np.allclose(p.astype(np.float32).sum(-1) + tail, 1., atol=.002)):
            raise ValueError('Invalid teacher probability distribution')
    return result


@torch.inference_mode()
def create_teacher_cache(model, data, config, device, output):
    output = Path(output); output.mkdir(exist_ok=False)
    n = sum(row['target_tokens'] + 1 for row in data.rows['train'])
    topk = config['kd_topk']; temperature = config['kd_temperature']
    ids = np.lib.format.open_memmap(output / 'ids.npy', mode='w+', dtype='<u2', shape=(n, topk))
    probabilities = np.lib.format.open_memmap(output / 'probabilities.npy', mode='w+', dtype='<f2', shape=(n, topk))
    tail = np.lib.format.open_memmap(output / 'tail.npy', mode='w+', dtype='<f4', shape=(n,))
    jobs = data.jobs('train', config['target_block']); model.eval(); covered = 0
    for number in range(0, len(jobs), config['microbatch']):
        chosen = jobs[number:number + config['microbatch']]
        source, input_ids, truth, first = batch(data, 'train', chosen, config, device)
        logits = model(input_ids, source=source, valid=input_ids != 0, output_start=first)[0]
        for i, (index, start, end, kind) in enumerate(chosen):
            p = F.softmax(logits[i][truth[i] != -100].float() / temperature, dim=-1)
            values, positions = p.topk(topk, dim=-1)
            offset = data.rows['train'][index]['target_offset'] + index + start
            count = end - start
            ids[offset:offset + count] = positions.cpu().numpy().astype('<u2')
            probabilities[offset:offset + count] = values.cpu().numpy().astype('<f2')
            tail[offset:offset + count] = (1 - values.sum(-1)).clamp_min(0).cpu().numpy()
            covered += count
        if number % 128 == 0:
            atomic_json(output / 'progress.json', {'stage': 'training_only_token_distributions',
                        'windows_done': number + len(chosen), 'windows_total': len(jobs), 'targets': covered})
    if covered != n: raise ValueError('Teacher cache missed reference targets')
    for value in (ids, probabilities, tail): value.flush()
    files = {key: {'path': str(output / (key + '.npy')), 'sha256': file_sha256(output / (key + '.npy'))}
             for key in ['ids', 'probabilities', 'tail']}
    manifest = {'split': 'train', 'targets': n, 'topk': topk, 'temperature': temperature,
        'teacher_parameters': config['pretrained_parameters'], 'files': files,
        'training_IDs': [r['id'] for r in data.rows['train']],
        'tokenizer_sha256': config['tokenizer_sha256'], 'data_manifest_sha256': file_sha256(Path(config['data_manifest'])),
        'checkpoint_sha256': file_sha256(Path(config['output']) / 'production/best.pt'),
        'distribution': 'Top32 probabilities plus aggregated remaining mass, normalized after FP16 storage',
        'no_dev_or_test_teacher_targets': True}
    atomic_json(output / 'manifest.json', manifest)
    return str(output / 'manifest.json')


def train(config_path, device, resume=False):
    config_path = Path(config_path); config = json.loads(config_path.read_text()); digest = file_sha256(config_path)
    verify(config)
    output = Path(config['output']) / 'production'; output.mkdir(exist_ok=resume)
    data = SummaryData(config['data_manifest'], config['source_budget'])
    model = build(config, device); frozen = frozen_parameter_sha256(model)
    atomic_json(output / 'parameters.json', parameter_summary(model))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    progress = {'epoch': 0, 'next_batch': 0, 'updates': 0, 'best_epoch': None,
                'best_loss': None, 'bad_epochs': 0, 'train_loss_sum': 0., 'train_target_count': 0}
    if resume: progress = load(output / 'latest.pt', model, optimizer, digest)
    teacher = teacher_cache(config, data)
    jobs = data.jobs('train', config['target_block'])
    selected_positions = sum(len(tokens) for tokens, spans in data.selected['train'])
    source_positions = sum(row['source_tokens'] for row in data.rows['train'])
    atomic_json(output / 'training_coverage.json', {'training_documents': len(data.rows['train']),
        'reference_targets': sum(row['target_tokens'] + 1 for row in data.rows['train']),
        'teacher_targets': sum(row['target_tokens'] + 1 for row in data.rows['train']) if teacher else 0,
        'KD': '0.5 reference CE +0.5 partitioned top32+tail KL, T=2' if teacher else None,
        'windows_per_epoch': len(jobs), 'sampled_documents': False, 'reference_truncated': False,
        'source_selection': config['source_selection'], 'selection_metric': 'Full heldout summary-token cross entropy',
        'selected_source_positions_per_document_pass': selected_positions,
        'complete_source_positions_per_document_pass': source_positions,
        'selected_source_position_fraction': selected_positions / source_positions,
        'metric_limitation': 'A loss proxy, not validation BERTScore or ROUGE; generation evaluation is preserved separately'})
    started = time.monotonic(); log = output / ('updates_%d.jsonl' % time.time_ns())
    micro = config['microbatch']; steps = math.ceil(len(jobs) / micro)
    atomic_json(output / 'progress.json', {**progress, 'stage': 'training', 'batches_per_epoch': steps,
                                           'elapsed_seconds': time.monotonic() - started})
    for epoch in range(progress['epoch'], config['max_epochs']):
        order = np.random.default_rng(config['seed'] + epoch).permutation(len(jobs))
        for number in range(progress['next_batch'], steps):
            tick = time.monotonic(); model.train(); optimizer.zero_grad(set_to_none=True)
            selected = [jobs[int(i)] for i in order[number * micro:(number + 1) * micro]]
            value, count = loss(model, data, 'train', selected, config, device, teacher=teacher)
            (value / count).backward()
            norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                       1., error_if_nonfinite=True))
            step = progress['updates'] + 1
            factor = min(1., step / config['warmup_updates'])
            for group in optimizer.param_groups: group['lr'] = config['learning_rate'] * factor
            optimizer.step()
            progress.update(epoch=epoch, next_batch=number + 1, updates=step,
                train_loss_sum=progress['train_loss_sum'] + float(value.detach()),
                train_target_count=progress['train_target_count'] + count)
            append(log, {'epoch': epoch + 1, 'batch': number + 1, 'batches': steps, 'updates': step,
                         'summary_loss': float(value.detach()) / count, 'valid_targets': count,
                         'gradient_norm': norm, 'seconds': time.monotonic() - tick})
            if step % config['checkpoint_updates'] == 0:
                save(output, model, optimizer, progress, digest, False)
            if step % 25 == 0:
                atomic_json(output / 'progress.json', {**progress, 'stage': 'training', 'batches_per_epoch': steps,
                                                       'elapsed_seconds': time.monotonic() - started})
        atomic_json(output / 'progress.json', {**progress, 'stage': 'validation'})
        validation = evaluate(model, data, config, device)
        score = validation['summary_token_cross_entropy']
        improved = progress['best_loss'] is None or score < progress['best_loss']
        if improved: progress.update(best_loss=score, best_epoch=epoch + 1, bad_epochs=0)
        else: progress['bad_epochs'] += 1
        append(output / 'epochs.jsonl', {'epoch': epoch + 1,
            'train_summary_CE': progress['train_loss_sum'] / progress['train_target_count'],
            'validation': validation, 'improved': improved})
        progress.update(epoch=epoch + 1, next_batch=0, train_loss_sum=0., train_target_count=0)
        if frozen_parameter_sha256(model) != frozen:
            raise ValueError('Frozen pretrained parameters changed')
        save(output, model, optimizer, progress, digest, improved)
        atomic_json(output / 'progress.json', {**progress, 'stage': 'epoch_complete'})
        if progress['bad_epochs'] >= config['patience']: break
    best = load(output / 'best.pt', model, None, digest, restore_state=False)
    teacher_manifest = None
    if config.get('create_teacher_cache'):
        atomic_json(output / 'progress.json', {**progress, 'stage': 'creating_training_only_teacher_cache'})
        teacher_manifest = create_teacher_cache(model, data, config, device, output / 'teacher_cache')
    atomic_json(output / 'progress.json', {**progress, 'stage': 'final_test_generation'})
    archive = export(model, data, config, device, output / 'final_test')
    # Reference summaries are read only on the heldout split, never test.
    atomic_json(output / 'progress.json', {**progress, 'stage': 'final_heldout_generation'})
    generated_dev = export(model, data, config, device, output / 'final_dev', 'dev')
    raw = json.loads(Path(config['raw_train']).read_text())
    reference = {r['id']: r['summary'] for r in raw}
    predicted_dev = json.loads((output / 'final_dev/prediction.json').read_text())
    from rouge_score.rouge_scorer import RougeScorer
    scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(reference[r['id']], r['summary']) for r in predicted_dev]
    metrics = {metric: float(np.mean([s[metric].fmeasure for s in scores])) for metric in ['rouge1', 'rougeL']}
    atomic_json(output / 'final_dev/truth.json', {r['id']: reference[r['id']] for r in predicted_dev})
    atomic_json(output / 'final_dev/rouge.json', metrics)
    verify(config)
    result = {'status': 'completed', 'candidate': config['name'], 'best_epoch': best['best_epoch'],
        'best_validation_summary_CE': best['best_loss'], 'completed_epochs': progress['epoch'],
        'stopped_early': progress['bad_epochs'] >= config['patience'], 'submission': archive,
        'generated_dev': generated_dev, 'generated_dev_ROUGE': metrics,
        'teacher_manifest': str(teacher_manifest) if teacher_manifest else None,
        'checkpoint_sha256': file_sha256(output / 'best.pt'), 'config_sha256': digest,
        'uploaded': False, 'leaderboard_changes': False, 'elapsed_seconds': time.monotonic() - started}
    atomic_json(output / 'completed.json', result)
    atomic_json(output / 'progress.json', {**progress, 'stage': 'completed'})
    return result
