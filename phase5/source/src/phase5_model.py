"""A document-level, 100-label head on the unchanged Phase1 backbones."""

import torch
from torch import nn
from torch.nn import functional as F

from src.phase2_adaptation import LoRMAPlusLinear
from src.phase4_model import CaseClassifier, batches, rng_restore, rng_snapshot
from src.phase5_prepare import binary_target


class StatuteClassifier(CaseClassifier):
    def __init__(self, backbone, architecture, method, precision='bf16', partial_layers=2,
                 rank=8, alpha=8., head_dropout=.1):
        super().__init__(backbone, 'full_fine_tuning', head='linear', architecture=architecture)
        self.method, self.precision = method, precision
        self.classifier = nn.Sequential(nn.LayerNorm(backbone.cfg.d_model), nn.Dropout(head_dropout),
                                        nn.Linear(backbone.cfg.d_model, 100))
        if method == 'partial_layerwise':
            backbone.requires_grad_(False)
            stack = backbone.decoder if architecture == 'encoder_decoder' else backbone.blocks
            if not 0 < partial_layers < len(stack):
                raise ValueError('Partial tuning must leave earlier blocks frozen')
            for block in stack[-partial_layers:]:
                block.requires_grad_(True)
            norm = backbone.decoder_norm if architecture == 'encoder_decoder' else backbone.ln_f
            norm.requires_grad_(True)
        elif method == 'lorma_plus':
            if architecture not in ('mla', 'dense'):
                raise ValueError('Only the announced MLA/dense LoRMA mappings are supported')
            backbone.requires_grad_(False)
            for index, block in enumerate(backbone.blocks):
                names = ('q_proj', 'v_up_proj') if architecture == 'mla' else ('q_proj', 'v_proj')
                for name in names:
                    setattr(block.attn, name, LoRMAPlusLinear(getattr(block.attn, name), rank, alpha))
                    self.targets.append(f'backbone.blocks.{index}.attn.{name}')
        elif method not in ('full_fine_tuning', 'knowledge_distillation'):
            raise ValueError('Unannounced Phase5 adaptation method')

    def chunk_vectors(self, ids, lengths):
        with torch.autocast(ids.device.type, dtype=torch.bfloat16,
                            enabled=self.precision == 'bf16' and ids.device.type == 'cuda'):
            return super().chunk_vectors(ids, lengths)


def document_loss(logits, labels, teacher=None, temperature=2., kd_weight=.5):
    if logits.shape != (100,):
        raise ValueError('LSI requires 100 independent logits')
    target = torch.as_tensor(binary_target(labels), device=logits.device)
    supervised = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='mean')
    if teacher is None:
        return supervised, supervised.detach(), supervised.detach().new_zeros(())
    soft = torch.as_tensor(teacher, dtype=torch.float32, device=logits.device).detach()
    if (soft.shape != (100,) or not torch.isfinite(soft).all() or bool(((soft < 0) | (soft > 1)).any())
            or temperature <= 0 or not 0 <= kd_weight <= 1):
        raise ValueError('Invalid independent Bernoulli teacher targets/settings')
    # Independent sigmoid targets: unlike Phase4, these probabilities do not sum to one.
    distillation = F.binary_cross_entropy_with_logits(logits.float() / temperature, soft,
                                                      reduction='mean') * temperature ** 2
    return (1 - kd_weight) * supervised + kd_weight * distillation, supervised.detach(), distillation.detach()


def backward_document(model, chunks, labels, denominator, microbatch=2, teacher=None,
                      temperature=2., kd_weight=.5, force_replay=False):
    """One loss per complete document; replay bounds memory, never coverage."""
    if denominator < 1 or not model.training or not chunks:
        raise ValueError('Training requires a complete case and actual positive batch denominator')
    device = next(model.parameters()).device
    states, pooled = [], None
    replay = force_replay or len(chunks) > microbatch
    if replay:
        with torch.no_grad():
            for ids, lengths in batches(chunks, microbatch, device):
                states.append(rng_snapshot(device))
                value = model.chunk_vectors(ids, lengths).sum(0)
                pooled = value if pooled is None else pooled + value
        pooled = (pooled / len(chunks)).detach().requires_grad_(True)
    else:
        ids, lengths = next(batches(chunks, microbatch, device))
        pooled = model.chunk_vectors(ids, lengths).mean(0)
    logits = model.classifier(pooled.float())
    loss, supervised, distillation = document_loss(logits, labels, teacher, temperature, kd_weight)
    if not bool(torch.isfinite(loss)):
        raise ValueError('Nonfinite multi-label document loss')
    (loss / denominator).backward()
    if replay:
        derivative = pooled.grad.detach() / len(chunks)
        after_head = rng_snapshot(device)
        try:
            for state, (ids, lengths) in zip(states, batches(chunks, microbatch, device)):
                rng_restore(state, device)
                vectors = model.chunk_vectors(ids, lengths)
                vectors.backward(derivative.expand_as(vectors))
        finally:
            rng_restore(after_head, device)
    return {'loss': float(loss.detach()), 'supervised_BCE': float(supervised),
            'distillation_BCE_T2': float(distillation), 'chunks': len(chunks)}


@torch.no_grad()
def predict_document(model, chunks, microbatch=2):
    if model.training:
        raise ValueError('Explicitly disable dropout for inference')
    device = next(model.parameters()).device
    pooled, count = None, 0
    for ids, lengths in batches(chunks, microbatch, device):
        vectors = model.chunk_vectors(ids, lengths)
        value = vectors.sum(0)
        pooled = value if pooled is None else pooled + value
        count += len(vectors)
    if count != len(chunks) or not count:
        raise ValueError('Inference lost document chunks')
    logits = model.classifier(pooled.float() / count)
    if logits.shape != (100,) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Invalid LSI logits')
    return logits
