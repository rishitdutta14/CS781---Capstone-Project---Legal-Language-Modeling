"""Separate loss and chunk-aggregation experiments on the original Phase1 GQA.

ASL follows the authors' AsymmetricLoss defaults, averaging over100 labels.
Attention pooling uses a zero-initialized scalar score on normalized chunks.
Every document/chunk remains present; long-document gradients are replayed.
"""

from contextlib import contextmanager
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase3_model import load_backbone
from src.phase4_model import batches, rng_restore, rng_snapshot
import src.phase5_checks as checks
from src.phase5_gqa_methods import GQAMethodClassifier
import src.phase5_model as model_module
from src.phase5_prepare import binary_target
import src.phase5_training as training

UNWEIGHTED_LOSS = model_module.document_loss
MEAN_REPLAY_GATE = checks.replay_gate

ASL_DEFAULTS = {'gamma_negative': 4., 'gamma_positive': 1., 'negative_clip': .05,
                'epsilon': 1e-8, 'detach_focusing_weights': True}


def asymmetric_loss(logits, labels, teacher=None, temperature=2., kd_weight=.5,
                    gamma_negative=4., gamma_positive=1., negative_clip=.05,
                    epsilon=1e-8, detach_focusing_weights=True):
    if logits.shape != (100,) or teacher is not None:
        raise ValueError('ASL requires100 independent supervised logits and no teacher')
    if min(gamma_negative, gamma_positive) < 0 or not 0 <= negative_clip < 1 or not 0 < epsilon < 1:
        raise ValueError('Invalid asymmetric-loss settings')
    logits = logits.float()
    target = torch.as_tensor(binary_target(labels), device=logits.device)
    positive = logits.sigmoid()
    negative = (1 - positive + negative_clip).clamp(max=1)
    terms = target * positive.clamp_min(epsilon).log() + (1 - target) * negative.clamp_min(epsilon).log()
    probability = target * positive + (1 - target) * negative
    exponent = target * gamma_positive + (1 - target) * gamma_negative
    focusing = (1 - probability).pow(exponent)
    if detach_focusing_weights:
        focusing = focusing.detach()
    loss = -(terms * focusing).mean()
    # This diagnostic stays on the same BCE scale as the unweighted baseline.
    # Its detached value contributes no additional training objective.
    bce = F.binary_cross_entropy_with_logits(logits.detach(), target)
    return loss, bce, bce.new_zeros(())


class AttentionHead(nn.Module):
    def __init__(self, linear, width):
        super().__init__()
        self.linear = linear
        # No random initialization: preserve the reference head and RNG stream.
        self.chunk_score = nn.Parameter(torch.zeros(width))

    def weights(self, vectors):
        if vectors.ndim != 2 or not len(vectors):
            raise ValueError('At least one chunk representation is required')
        normalized = F.layer_norm(vectors.float(), (vectors.shape[-1],))
        return torch.softmax(normalized @ self.chunk_score, dim=0)

    def aggregate(self, vectors):
        return (vectors.float() * self.weights(vectors)[:, None]).sum(0)

    def forward(self, pooled):
        return self.linear(pooled)


class AttentionGQA(GQAMethodClassifier):
    def __init__(self, backbone, precision='bf16', head_dropout=.1):
        super().__init__(backbone, 'full_fine_tuning', precision=precision, head_dropout=head_dropout)
        self.classifier = AttentionHead(self.classifier, backbone.cfg.d_model)

    def aggregate(self, vectors):
        return self.classifier.aggregate(vectors)


def validate_config(config):
    if (config['architecture'] != 'gqa' or config['method'] != 'full_fine_tuning'
            or config['initialization'] != 'own_Phase1_GQA_only_fresh_task_head'
            or config['kd_weight'] != 0. or config.get('export_each_validation_improvement') is not False):
        raise ValueError('Use a fresh Phase1 GQA start and final-only test exports')
    expected = {'asl_mean': ('asymmetric_loss', 'mean'), 'bce_attention': ('unweighted_BCE', 'attention')}
    if config['variant'] not in expected or (config['loss'], config['pooling']) != expected[config['variant']]:
        raise ValueError('Do not combine loss and pooling changes in these two ablations')
    if config['variant'] == 'asl_mean' and config['asymmetric_loss'] != ASL_DEFAULTS:
        raise ValueError('Keep the declared author-default ASL settings')


def build(config, device):
    validate_config(config)
    if file_sha256(Path(config['checkpoint'])) != config['checkpoint_sha256']:
        raise ValueError('Original Phase1 GQA checkpoint changed')
    backbone = load_backbone(config['checkpoint'], config['model_config'])
    if config['variant'] == 'bce_attention':
        return AttentionGQA(backbone, config['compute_precision'], config['head_dropout']).to(device)
    return GQAMethodClassifier(backbone, 'full_fine_tuning', precision=config['compute_precision'],
                               head_dropout=config['head_dropout']).to(device)


def backward_attention(model, chunks, labels, denominator, microbatch=2, teacher=None,
                       temperature=2., kd_weight=.5, force_replay=False):
    if denominator < 1 or not model.training or not chunks or teacher is not None:
        raise ValueError('Require a complete supervised document and actual batch denominator')
    device = next(model.parameters()).device
    replay = force_replay or len(chunks) > microbatch
    if replay:
        states, values = [], []
        with torch.no_grad():
            for ids, lengths in batches(chunks, microbatch, device):
                states.append(rng_snapshot(device))
                values.append(model.chunk_vectors(ids, lengths))
        vectors = torch.cat(values).detach().requires_grad_(True)
    else:
        ids, lengths = next(batches(chunks, microbatch, device))
        vectors = model.chunk_vectors(ids, lengths)
    logits = model.classifier(model.aggregate(vectors))
    loss, bce, distillation = model_module.document_loss(logits, labels, teacher, temperature, kd_weight)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite attention-pooling document loss')
    (loss / denominator).backward()
    if replay:
        derivative = vectors.grad.detach()
        after_head = rng_snapshot(device)
        offset = 0
        try:
            for state, (ids, lengths) in zip(states, batches(chunks, microbatch, device)):
                rng_restore(state, device)
                actual = model.chunk_vectors(ids, lengths)
                actual.backward(derivative[offset:offset + len(actual)])
                offset += len(actual)
        finally:
            rng_restore(after_head, device)
        if offset != len(chunks):
            raise ValueError('Attention replay omitted chunks')
    return {'loss': float(loss.detach()), 'supervised_BCE': float(bce),
            'distillation_BCE_T2': float(distillation), 'chunks': len(chunks)}


@torch.no_grad()
def predict_attention(model, chunks, microbatch=2):
    if model.training or not chunks:
        raise ValueError('Disable dropout and provide every document chunk')
    device = next(model.parameters()).device
    values = [model.chunk_vectors(ids, lengths) for ids, lengths in batches(chunks, microbatch, device)]
    vectors = torch.cat(values)
    if len(vectors) != len(chunks):
        raise ValueError('Attention inference lost document chunks')
    logits = model.classifier(model.aggregate(vectors))
    if logits.shape != (100,) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Invalid attention-pooled LSI logits')
    return logits


def attention_replay_gate(model, chunks, labels, teacher=None):
    """Exercise nonuniform learned attention, then restore the initial scorer."""
    device = next(model.parameters()).device
    saved_score = model.classifier.chunk_score.detach().clone()
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        with torch.no_grad():
            model.classifier.chunk_score.copy_(torch.linspace(-.02, .02, len(saved_score), device=device))
        before = rng_snapshot(device)
        values = [model.chunk_vectors(ids, lengths) for ids, lengths in batches(chunks, 1, device)]
        vectors = torch.cat(values)
        weights = model.classifier.weights(vectors).detach()
        loss = model_module.document_loss(model.classifier(model.aggregate(vectors)), labels)[0]
        reference = checks.gradients(model, loss)
        expected_loss = float(loss.detach())
        expected_rng = rng_snapshot(device)
        del values, vectors, loss
        rng_restore(before, device)
        record = backward_attention(model, chunks, labels, 1, 1, force_replay=True)
        actual = [(name, value.grad.detach().cpu() if value.grad is not None else torch.zeros_like(value, device='cpu'))
                  for name, value in model.named_parameters() if value.requires_grad]
        result = checks.gradient_metrics(reference, actual)
        after = rng_snapshot(device)
        same_rng = torch.equal(expected_rng[0], after[0]) and (
            after[1] is None or torch.equal(expected_rng[1], after[1]))
        score_gradient = float(model.classifier.chunk_score.grad.norm())
        result.update(loss_error=record['loss'] - expected_loss, dropout_RNG_matches_joint_forward=same_rng,
                      chunks=len(chunks), nonuniform_attention_probe=True, score_gradient_norm=score_gradient,
                      attention_weights=weights.cpu().tolist())
        result['passed'] = (abs(result['loss_error']) < 1e-4 and result['gradient_cosine'] > .999
                            and result['relative_gradient_L2_error'] < .01 and same_rng and score_gradient > 0)
        return result
    finally:
        with torch.no_grad():
            model.classifier.chunk_score.copy_(saved_score)
        model.zero_grad(set_to_none=True)


@contextmanager
def functions(config):
    """Override process-local entry points; no shared historical file edits."""
    validate_config(config)
    previous = (training.build, checks.build, training.backward_document, checks.backward_document,
                training.predict_document, model_module.document_loss, checks.document_loss, checks.replay_gate)
    training.build = checks.build = build
    if config['variant'] == 'asl_mean':
        model_module.document_loss = checks.document_loss = asymmetric_loss
        training.backward_document = checks.backward_document = model_module.backward_document
        training.predict_document = model_module.predict_document
        checks.replay_gate = MEAN_REPLAY_GATE
    else:
        model_module.document_loss = checks.document_loss = UNWEIGHTED_LOSS
        training.backward_document = checks.backward_document = backward_attention
        training.predict_document = predict_attention
        checks.replay_gate = attention_replay_gate
    try:
        yield
    finally:
        (training.build, checks.build, training.backward_document, checks.backward_document,
         training.predict_document, model_module.document_loss, checks.document_loss, checks.replay_gate) = previous


def pilot(config_path, device):
    config = training.read_json(config_path)
    with functions(config):
        result = checks.pilot(config_path, device)
    result.update(variant=config['variant'], loss=config['loss'], pooling=config['pooling'],
                  export_each_validation_improvement=False, pilot_weights_removed_after_success=True)
    output = Path(config['output']) / 'pilot'
    atomic_json(output / 'completed.json', result)
    # The numerical/recovery receipts remain; production starts fresh from Phase1.
    for name in ('best.pt', 'latest.pt'):
        (output / name).unlink()
    return result


def train(config_path, device, resume=False):
    config = training.read_json(config_path)
    with functions(config):
        result = training.train(config_path, device, resume)
    result.update(variant=config['variant'], loss=config['loss'], pooling=config['pooling'],
                  export_each_validation_improvement=False,
                  training_loss_note='ASL training loss and BCE have different scales; logged supervised_BCE and dev BCE are unweighted diagnostics')
    atomic_json(Path(config['output']) / 'production/completed.json', result)
    return result
