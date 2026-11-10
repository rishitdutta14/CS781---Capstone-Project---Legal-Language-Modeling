"""Token NER objectives with explicit, globally composable normalization."""

import torch
import torch.nn.functional as F


def training_class_weights(counts, power=0.5, maximum=10.0):
    """Capped inverse-frequency weights relative to O, from training tokens only."""
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or (counts <= 0).any() or power < 0 or maximum < 1:
        raise ValueError("all BIO tags need positive training counts and valid weight settings")
    return (counts[0] / counts).pow(power).clamp(min=1, max=maximum).float()


def token_loss_sum(logits, targets, kind="ce", weights=None, gamma=2.0):
    """Return numerator and denominator, not a mean of local-rank means.

    CE uses PyTorch's weighted-mean denominator. Multiclass focal loss uses
    the same fixed target-weight denominator; gamma=0 exactly recovers CE.
    Ignored tokens have zero loss and zero gradient in every objective.
    """
    if kind not in ("ce", "weighted_ce", "focal") or gamma < 0:
        raise ValueError("invalid token objective")
    logits = logits.float().reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    valid = targets != -100
    if kind == "weighted_ce" and weights is None:
        raise ValueError("weighted CE requires training-derived weights")
    if weights is not None:
        weights = weights.to(device=logits.device, dtype=logits.dtype)
        if weights.shape != (logits.size(-1),) or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("invalid class weights")
    if kind == "ce" and weights is not None:
        raise ValueError("plain CE must not silently use class weights")
    # Gather only real targets; empty tensors still preserve the graph for a
    # no-supervision rank in an odd final DDP group.
    real_logits, real_targets = logits[valid], targets[valid]
    ce = F.cross_entropy(real_logits, real_targets, reduction="none")
    mass = torch.ones_like(ce) if weights is None else weights[real_targets]
    if kind == "focal":
        ce = ce * (-torch.expm1(-ce)).pow(gamma)
    return (ce * mass).sum(), mass.sum()


def rank_group(group, rank, world):
    """Equal forward/backward counts, without repeating supervised windows.

    None is a zero-loss synchronization placeholder, not another training row.
    This allows the original tokenizer's odd window count with two GPUs.
    """
    if not group or not 0 <= rank < world:
        raise ValueError("invalid distributed group")
    padded = list(group) + [None] * ((-len(group)) % world)
    return padded[rank::world]
