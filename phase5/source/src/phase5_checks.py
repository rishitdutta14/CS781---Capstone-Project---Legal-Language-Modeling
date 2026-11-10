"""Real-device gates before LSI production; diagnostic subsets never cap training."""

import math
from pathlib import Path
import time

import numpy as np
import torch

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase1_full_training import state_digest
from src.phase2_adaptation import frozen_parameter_sha256
from src.phase4_deadline import gradient_metrics
from src.phase4_model import batches, rng_restore, rng_snapshot, trainability
from src.phase4_training import load, rng, save, verify_bindings
from src.phase5_model import backward_document, document_loss
from src.phase5_tasks import StatuteData
from src.phase5_training import build, make_optimizer, read_json


def gradients(model, loss):
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    values = torch.autograd.grad(loss, [value for _, value in named], allow_unused=True)
    return [(name, gradient.detach().cpu() if gradient is not None else torch.zeros_like(value, device='cpu'))
            for (name, value), gradient in zip(named, values)]


def manual_gate(model, data):
    model.eval()
    model.precision = 'fp32'
    device = next(model.parameters()).device
    ids, lengths = next(batches([data.chunks('train', 0)[0][:64]], 1, device))
    base = model.backbone
    x = base.tok_emb(ids) + base.pos_emb[:ids.shape[1]].unsqueeze(0)
    if model.architecture == 'encoder_decoder':
        memory = base.encode(ids)
        for block in base.decoder:
            x = block(x, memory)
        hidden = base.decoder_norm(x)
    else:
        for block in base.blocks:
            x = block(x)
        hidden = base.ln_f(x)
    loss_a = document_loss(model.classifier(hidden[0, -1]), data.rows['train'][0]['labels'])[0]
    a = gradients(model, loss_a)
    loss_b = document_loss(model.classifier(model.chunk_vectors(ids, lengths)[0]), data.rows['train'][0]['labels'])[0]
    b = gradients(model, loss_b)
    result = gradient_metrics(a, b)
    result.update(manual_loss=float(loss_a.detach()), efficient_loss=float(loss_b.detach()))
    result['passed'] = (abs(result['manual_loss'] - result['efficient_loss']) < 1e-4
                        and result['gradient_cosine'] > .999 and result['relative_gradient_L2_error'] < .01)
    result['scope'] = 'Actual TRAIN prefix at most64 tokens; original attention versus efficient FP32 path'
    return result


def precision_gate(model, data):
    model.eval()
    device = next(model.parameters()).device
    long_index = next(i for i, row in enumerate(data.rows['train']) if row['tokens'] >= 3198)
    short_index = next(i for i, row in enumerate(data.rows['train']) if row['tokens'] < 512)
    long_chunks = data.chunks('train', long_index)
    probes = [(long_chunks[0], long_index), (long_chunks[len(long_chunks) // 2], long_index),
              (data.chunks('train', short_index)[0], short_index)]
    checks = []
    for chunk, index in probes:
        ids, lengths = next(batches([chunk], 1, device))
        values, losses = [], []
        for precision in ('fp32', 'bf16'):
            model.precision = precision
            loss = document_loss(model.classifier(model.chunk_vectors(ids, lengths)[0]), data.rows['train'][index]['labels'])[0]
            values.append(gradients(model, loss))
            losses.append(float(loss.detach()))
        result = gradient_metrics(*values)
        result.update(FP32_loss=losses[0], BF16_loss=losses[1], tokens=len(chunk))
        projection = result['minimum_query_key_projection_cosine']
        result['passed'] = (all(math.isfinite(v) for v in losses)
                            and abs(losses[0] - losses[1]) < .05 and result['gradient_cosine'] >= .99
                            and result['relative_gradient_L2_error'] <= .15
                            and (projection is None or projection >= .95))
        checks.append(result)
    model.precision = 'bf16'
    return {'passed': all(row['passed'] for row in checks), 'checks': checks,
            'scope': 'Two full1600-token TRAIN chunks and one separate short TRAIN case, dropout disabled'}


def replay_gate(model, chunks, labels, teacher=None):
    device = next(model.parameters()).device
    model.train()
    model.zero_grad(set_to_none=True)
    before = rng_snapshot(device)
    vectors = [model.chunk_vectors(ids, lengths) for ids, lengths in batches(chunks, 1, device)]
    loss = document_loss(model.classifier(torch.cat(vectors).mean(0)), labels, teacher)[0]
    reference = gradients(model, loss)
    expected_loss = float(loss.detach())
    expected_rng = rng_snapshot(device)
    del vectors, loss
    rng_restore(before, device)
    record = backward_document(model, chunks, labels, 1, 1, teacher, force_replay=True)
    actual = [(name, value.grad.detach().cpu() if value.grad is not None else torch.zeros_like(value, device='cpu'))
              for name, value in model.named_parameters() if value.requires_grad]
    result = gradient_metrics(reference, actual)
    after = rng_snapshot(device)
    same_rng = torch.equal(expected_rng[0], after[0]) and (after[1] is None or torch.equal(expected_rng[1], after[1]))
    result.update(loss_error=record['loss'] - expected_loss, dropout_RNG_matches_joint_forward=same_rng,
                  chunks=len(chunks), passed=abs(record['loss'] - expected_loss) < 1e-4
                  and result['gradient_cosine'] > .999 and result['relative_gradient_L2_error'] < .01 and same_rng)
    model.zero_grad(set_to_none=True)
    return result


def recovery_gate(output, model, optimizer, digest, chunks, labels):
    def update():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        row = backward_document(model, chunks, labels, 1, 1)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return row

    load(Path(output) / 'latest.pt', model, optimizer, digest)
    first = update()
    expected = (state_digest(model.state_dict()), state_digest(optimizer.state_dict()))
    expected_rng = rng(next(model.parameters()).is_cuda)
    load(Path(output) / 'latest.pt', model, optimizer, digest)
    second = update()
    actual = (state_digest(model.state_dict()), state_digest(optimizer.state_dict()))
    actual_rng = rng(next(model.parameters()).is_cuda)
    same_rng = torch.equal(expected_rng['cpu'], actual_rng['cpu']) and (
        expected_rng['cuda'] is None or torch.equal(expected_rng['cuda'], actual_rng['cuda']))
    load(Path(output) / 'latest.pt', model, optimizer, digest)
    return {'passed': expected == actual and same_rng and first == second,
            'weights_exact': expected[0] == actual[0], 'optimizer_exact': expected[1] == actual[1],
            'dropout_rng_exact': same_rng, 'loss_exact': first == second}


def pilot(config_path, device):
    config = read_json(config_path)
    verify_bindings(config)
    output = Path(config['output']) / 'pilot'
    output.mkdir(exist_ok=False)
    data = StatuteData(config['data_manifest'])
    model = build(config, device)
    atomic_json(output / 'trainability.json', trainability(model))
    manual = manual_gate(model, data)
    atomic_json(output / 'manual_gradient_gate.json', manual)
    if not manual['passed']:
        raise ValueError('Original/efficient FP32 gradient comparison failed')
    precision = precision_gate(model, data)
    atomic_json(output / 'BF16_gradient_gate.json', precision)
    if not precision['passed']:
        raise ValueError('BF16 gradient comparison failed; no hidden precision fallback')
    # Actual full chunks, TRAIN-only diagnostic. Production always uses all chunks.
    long_index = next(i for i, row in enumerate(data.rows['train']) if row['tokens'] >= 3198)
    chunks = data.chunks('train', long_index)[:2]
    labels = data.rows['train'][long_index]['labels']
    synthetic_teacher = np.linspace(.05, .95, 100, dtype=np.float32) if config['method'] == 'knowledge_distillation' else None
    replay = replay_gate(model, chunks, labels, synthetic_teacher)
    atomic_json(output / 'replay_gate.json', replay)
    if not replay['passed']:
        raise ValueError('Stochastic full-context replay differs from joint autograd')
    optimizer = make_optimizer(model, config)
    frozen = frozen_parameter_sha256(model)
    order = np.random.default_rng(config['seed']).permutation(len(data.rows['train'])).tolist()[:16]
    timing = []
    for start in range(0, len(order), 4):
        selected = order[start:start + 4]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        values = [backward_document(model, data.chunks('train', i), data.rows['train'][i]['labels'],
                                    len(selected), config['chunk_microbatch']) for i in selected]
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                   config['clip_norm'], error_if_nonfinite=True))
        if norm > config['maximum_gradient_norm']:
            raise ValueError('Pilot gradient exceeds safety limit')
        optimizer.step()
        torch.cuda.synchronize()
        timing.append({'seconds': time.perf_counter() - tick, 'cases': len(selected),
                       'chunks': sum(row['chunks'] for row in values), 'gradient_norm': norm})
    if frozen_parameter_sha256(model) != frozen:
        raise ValueError('Pilot changed frozen pretrained parameters')
    digest = file_sha256(Path(config_path))
    save(output, model, optimizer, {'epoch': 0, 'next_batch': 0, 'updates': 4}, digest, True)
    recovery = recovery_gate(output, model, optimizer, digest, [chunks[0][:64]], labels)
    atomic_json(output / 'recovery_gate.json', recovery)
    if not recovery['passed']:
        raise ValueError('Exact next-update optimizer/RNG recovery failed')
    seconds = sum(row['seconds'] for row in timing)
    count = sum(row['chunks'] for row in timing)
    record = {'passed': True, 'config_sha256': digest, 'manual_gradient_gate': manual,
              'BF16_gradient_gate': precision, 'replay_gate': replay, 'recovery_gate': recovery,
              'frozen_parameters_unchanged': True, 'timing': timing,
              'estimated_train_seconds_per_epoch': seconds / count * data.manifest['splits']['train']['chunks'],
              'estimate_caveat': '16 complete TRAIN cases, excludes dev/test/I/O; small cases and chunk lengths can skew projection',
              'peak_allocated_GPU_bytes': torch.cuda.max_memory_allocated(),
              'KD_actual_teacher_gate_pending': config['method'] == 'knowledge_distillation',
              'production_starts_from_Phase1_not_pilot': True}
    atomic_json(output / 'completed.json', record)
    return record
