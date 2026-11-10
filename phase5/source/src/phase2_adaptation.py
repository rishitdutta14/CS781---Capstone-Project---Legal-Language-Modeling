"""NER adaptation without replacing the existing Phase 1 backbone.

LoRMA+ follows Bihany, Patel and Modi (ACL 2025), equation 8:
    W_eff = (I + alpha/r * B A) W0
This is an independent implementation of the pre-multiplicative identity
inflation variant. It is not the permutation variant or an additive LoRA.
"""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


METHODS = ("full_fine_tuning", "lorma_plus", "partial_layerwise_fine_tuning",
           "knowledge_distillation")


class LoRMAPlusLinear(nn.Module):
    def __init__(self, original, rank=8, alpha=8.):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("LoRMA+ needs an existing linear projection")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 < rank <= original.out_features:
            raise ValueError("rank must be a positive integer within the output width")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.original = original
        self.original.requires_grad_(False)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.rank, self.alpha, self.scaling = rank, float(alpha), float(alpha)/rank
        self.A = nn.Parameter(original.weight.new_empty(rank, self.out_features))
        self.B = nn.Parameter(original.weight.new_zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        # Bias is not part of the multiplicative transformation. Associativity
        # avoids constructing a full output-width-square matrix every forward.
        projected = F.linear(x, self.original.weight)
        adapted = projected + self.scaling * F.linear(F.linear(projected, self.A), self.B)
        return adapted if self.original.bias is None else adapted + self.original.bias

    def effective_weight(self):
        return self.original.weight + self.scaling * (self.B @ (self.A @ self.original.weight))

    @torch.no_grad()
    def merged_copy(self):
        """Separate inference layer; never overwrite the source or adapter."""
        layer = nn.Linear(self.in_features, self.out_features, self.original.bias is not None,
                          device=self.A.device, dtype=self.A.dtype)
        layer.weight.copy_(self.effective_weight())
        if layer.bias is not None:
            layer.bias.copy_(self.original.bias)
        layer.requires_grad_(False)
        layer.train(self.training)
        return layer


def parameter_summary(model):
    trainable, frozen = {}, {}
    for name, parameter in model.named_parameters():
        (trainable if parameter.requires_grad else frozen)[name] = parameter.numel()
    return {"total_parameters": sum(trainable.values()) + sum(frozen.values()),
            "trainable_parameters": sum(trainable.values()),
            "frozen_parameters": sum(frozen.values()),
            "trainable_parameter_names": list(trainable),
            "frozen_parameter_names": list(frozen)}


def frozen_parameter_sha256(model):
    """Bind names, shapes, dtypes, and exact bytes, including BF16 tensors."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(f"{name}:{tuple(parameter.shape)}:{parameter.dtype}\n".encode())
            values = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def apply_adaptation(model, method="full_fine_tuning", rank=8, alpha=8., partial_layers=2):
    """Configure trainability once, after exact Phase 1 loading and before DDP."""
    if method not in METHODS:
        raise ValueError("unknown adaptation method")
    if not hasattr(model, "bio") or not hasattr(model, "boundary"):
        raise ValueError("adaptation expects the verified BIO plus boundary NER model")
    bio = model.bio
    targets = []
    if method in ("full_fine_tuning", "knowledge_distillation"):
        model.requires_grad_(True)
    elif method == "lorma_plus":
        if not hasattr(bio, "blocks") or getattr(bio.cfg, "attention_type", None) != "standard":
            raise ValueError("this campaign's LoRMA+ mapping requires the dense decoder")
        # Validate all targets before mutating the model.
        for number, block in enumerate(bio.blocks):
            for name in ("q_proj", "v_proj"):
                if not isinstance(getattr(block.attn, name, None), nn.Linear):
                    raise ValueError("expected unchanged dense query/value projections")
                targets.append((number, name))
        if not targets:
            raise ValueError("LoRMA+ has no target blocks")
        model.requires_grad_(False)
        for number, name in targets:
            original = getattr(bio.blocks[number].attn, name)
            setattr(bio.blocks[number].attn, name, LoRMAPlusLinear(original, rank, alpha))
        bio.classifier.requires_grad_(True)
        model.boundary.requires_grad_(True)
    else:
        if not hasattr(bio, "encoder") or not hasattr(bio, "decoder"):
            raise ValueError("partial layer-wise mapping requires both encoder-decoder stacks")
        if isinstance(partial_layers, bool) or not isinstance(partial_layers, int) or not (
                0 < partial_layers < min(len(bio.encoder), len(bio.decoder))):
            raise ValueError("partial depth must leave earlier layers frozen in both stacks")
        model.requires_grad_(False)
        for stack in (bio.encoder, bio.decoder):
            for block in stack[-partial_layers:]:
                block.requires_grad_(True)
        bio.encoder_norm.requires_grad_(True)
        bio.decoder_norm.requires_grad_(True)
        bio.classifier.requires_grad_(True)
        model.boundary.requires_grad_(True)
    summary = parameter_summary(model)
    summary.update({"method": method,
        "target_modules": [f"bio.blocks.{i}.attn.{name}" for i, name in targets],
        "rank": rank if method == "lorma_plus" else None,
        "alpha": float(alpha) if method == "lorma_plus" else None,
        "variant": "pre-multiplicative LoRMA+ identity inflation (equation 8)" if targets else None,
        "adapter_dropout": 0. if targets else None,
        "partial_layers_per_stack": partial_layers if method == "partial_layerwise_fine_tuning" else None,
        "frozen_initial_sha256": frozen_parameter_sha256(model)})
    return summary
