"""Whole-case classification with bounded-memory exact chunk-gradient replay.

The original A3 attention types, dimensions, parameters and causal semantics
are retained. Efficient attention and LoRMA+ are the previously tested helpers.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.phase2_adaptation import LoRMAPlusLinear
from src.phase3_model import decoder_hidden
from src.phase3_rebuilt_model import hidden_states


class CaseClassifier(nn.Module):
    def __init__(self, backbone, method, head="linear", partial_layers=2,
                 rank=8, alpha=8., head_dropout=.1, architecture="original"):
        super().__init__()
        self.backbone, self.method, self.head_type = backbone, method, head
        self.architecture = architecture
        self.targets = []
        backbone.requires_grad_(False)
        if method in ("full_fine_tuning", "knowledge_distillation"):
            backbone.requires_grad_(True)
        elif method == "partial_layerwise":
            if not 0 < partial_layers < len(backbone.blocks):
                raise ValueError("Partial training must leave some earlier layers frozen")
            for block in backbone.blocks[-partial_layers:]:
                block.requires_grad_(True)
            backbone.ln_f.requires_grad_(True)
        elif method == "lorma_plus":
            if backbone.cfg.attention_type != "mla":
                raise ValueError("This campaign maps LoRMA+ to MLA")
            for number, block in enumerate(backbone.blocks):
                for name in ("q_proj", "v_up_proj"):
                    setattr(block.attn, name, LoRMAPlusLinear(getattr(block.attn, name), rank, alpha))
                    self.targets.append(f"backbone.blocks.{number}.attn.{name}")
        else:
            raise ValueError("Unknown case adaptation method")
        width = backbone.cfg.d_model
        if head == "linear":
            self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Dropout(head_dropout), nn.Linear(width, 2))
        elif head == "mlp":
            self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256), nn.GELU(),
                nn.Dropout(head_dropout), nn.Linear(256, 2))
        else:
            raise ValueError("Choose linear or mlp head")

    def chunk_vectors(self, ids, lengths):
        if (ids.ndim != 2 or lengths.shape != (len(ids),)
                or bool((lengths < 1).any()) or bool((lengths > ids.shape[1]).any())):
            raise ValueError("Invalid right-padded chunk batch")
        hidden = (hidden_states(self.backbone, self.architecture, ids)
                  if self.architecture in ("dense", "encoder_decoder")
                  else decoder_hidden(self.backbone, ids))
        return hidden[torch.arange(len(ids), device=ids.device), lengths - 1].float()


def batches(chunks, microbatch, device):
    if not chunks or microbatch < 1:
        raise ValueError("Every case needs at least one chunk")
    for start in range(0, len(chunks), microbatch):
        group = chunks[start:start + microbatch]
        lengths = torch.tensor([len(row) for row in group], device=device)
        ids = torch.zeros((len(group), int(lengths.max())), dtype=torch.long, device=device)
        for index, row in enumerate(group):
            ids[index, :len(row)] = torch.as_tensor(row, device=device)
        yield ids, lengths


def rng_snapshot(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def rng_restore(state, device):
    torch.set_rng_state(state[0])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state[1], device)


def case_loss(logits, label, teacher=None, temperature=2., kd_weight=.5, class_weights=None):
    ce = F.cross_entropy(logits.float().view(1, 2),
        torch.as_tensor([label], device=logits.device), reduction="sum")
    if class_weights is not None:
        ce = ce * class_weights[label]
    if teacher is None:
        return ce, ce.detach(), ce.detach().new_zeros(())
    teacher = torch.as_tensor(teacher, device=logits.device, dtype=torch.float32)
    if (teacher.shape != (2,) or not torch.isfinite(teacher).all()
            or bool((teacher < 0).any()) or not torch.allclose(teacher.sum(), teacher.new_tensor(1.), atol=1e-5)
            or temperature <= 0 or not 0 <= kd_weight <= 1):
        raise ValueError("Invalid detached teacher probability distribution/settings")
    kl = F.kl_div(F.log_softmax(logits.float().view(2) / temperature, -1),
                  teacher.detach(), reduction="sum") * temperature ** 2
    return (1 - kd_weight) * ce + kd_weight * kl, ce.detach(), kl.detach()


def backward_case(model, chunks, label, denominator, microbatch=2, teacher=None,
                  temperature=2., kd_weight=.5, class_weights=None):
    """Exact document-level gradient, keeping graphs for only one chunk batch.

    Pass1 computes the equal mean and remembers dropout RNG before each batch.
    Backpropagate the head once, then replay each chunk with its exact dropout
    RNG and the common dLoss/dMean / chunk_count. Restore the post-head RNG so
    future documents do not replay random streams. No synthetic chunk labels.
    """
    if denominator < 1 or not model.training:
        raise ValueError("Training and positive actual case denominator required")
    device = next(model.parameters()).device
    states, pooled = [], None
    with torch.no_grad():
        for ids, lengths in batches(chunks, microbatch, device):
            states.append(rng_snapshot(device))
            vectors = model.chunk_vectors(ids, lengths)
            value = vectors.sum(0)
            pooled = value if pooled is None else pooled + value
    pooled = (pooled / len(chunks)).detach().requires_grad_(True)
    logits = model.classifier(pooled)
    loss, ce, kl = case_loss(logits, label, teacher, temperature, kd_weight, class_weights)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite case loss")
    (loss / denominator).backward()
    derivative = pooled.grad.detach() / len(chunks)
    after = rng_snapshot(device)
    try:
        for state, (ids, lengths) in zip(states, batches(chunks, microbatch, device)):
            rng_restore(state, device)
            vectors = model.chunk_vectors(ids, lengths)
            vectors.backward(derivative.expand_as(vectors))
    finally:
        rng_restore(after, device)
    return {"loss": float(loss.detach()), "ce": float(ce), "kl": float(kl), "chunks": len(chunks)}


@torch.no_grad()
def predict_case(model, chunks, microbatch=2):
    if model.training:
        raise ValueError("Inference must explicitly disable dropout")
    device = next(model.parameters()).device
    pooled = None
    count = 0
    for ids, lengths in batches(chunks, microbatch, device):
        vectors = model.chunk_vectors(ids, lengths)
        value = vectors.sum(0)
        pooled = value if pooled is None else pooled + value
        count += len(vectors)
    if count != len(chunks):
        raise ValueError("Inference omitted chunks")
    logits = model.classifier(pooled / count)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Nonfinite case logits")
    return logits


def trainability(model):
    trainable = {name: value.numel() for name, value in model.named_parameters() if value.requires_grad}
    return {"total_parameters": sum(value.numel() for value in model.parameters()),
        "trainable_parameters": sum(trainable.values()), "trainable_names": list(trainable),
        "adapter_targets": model.targets, "method": model.method, "head": model.head_type}
