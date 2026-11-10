"""Read-only precision policies and gradient-comparison helpers."""

import contextlib
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.phase1_full_encdec import PaddedEncoderAttention, PaddedCausalAttention, PaddedCrossAttention

POLICIES = ("fp32_reference", "bf16_original", "bf16_math", "bf16_fp32_attention")


@contextlib.contextmanager
def precision_policy(model, policy, device_type="cuda"):
    if policy not in POLICIES:
        raise ValueError("Unknown precision study policy")
    originals = []
    try:
        if policy == "bf16_fp32_attention":
            for module in model.modules():
                if isinstance(module, (PaddedEncoderAttention, PaddedCausalAttention, PaddedCrossAttention)):
                    original = module.forward
                    originals.append((module, original))
                    def forward(*args, _original=original, **kwargs):
                        def cast(value):
                            return value.float() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
                        with torch.autocast(device_type, enabled=False):
                            return _original(*(cast(value) for value in args), **{k: cast(v) for k, v in kwargs.items()})
                    module.forward = forward
        kernel = contextlib.nullcontext() if policy == "bf16_original" else sdpa_kernel(SDPBackend.MATH)
        autocast = torch.autocast(device_type, dtype=torch.bfloat16) if policy != "fp32_reference" else contextlib.nullcontext()
        with kernel, autocast:
            yield
    finally:
        for module, original in originals:
            module.forward = original


def gradient_snapshot(model):
    return {name: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for name, p in model.named_parameters()}


def compare_gradients(model, reference):
    device = next(model.parameters()).device
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    largest = []
    for name, parameter in model.named_parameters():
        actual = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        expected = reference[name]
        # FP32 elementwise terms; FP64 accumulation of scalar tensor norms.
        terms = torch.stack((actual.square().sum(), expected.square().sum(),
                             (actual * expected).sum(), (actual - expected).square().sum()))
        totals += terms.double()
        largest.append((name, float(terms[0].sqrt())))
    a2, b2, dot, error2 = map(float, totals)
    if not all(math.isfinite(v) for v in (a2, b2, dot, error2)):
        return {"finite": False, "gradient_norm": None, "relative_L2_error": None, "cosine": None}
    return {"finite": True, "gradient_norm": math.sqrt(max(a2, 0)),
            "reference_gradient_norm": math.sqrt(max(b2, 0)),
            "relative_L2_error": math.sqrt(max(error2, 0) / b2) if b2 else math.sqrt(max(error2, 0)),
            "cosine": dot / math.sqrt(a2 * b2) if a2 and b2 else None,
            "zero_gradient_matches": a2 == 0 and b2 == 0,
            "largest_gradient_parameters": sorted(largest, key=lambda x: x[1], reverse=True)[:8]}


def agreement_passes(comparison, loss_error, loss_tolerance=.01, cosine_minimum=.98, relative_error_maximum=.2):
    if not comparison["finite"] or not math.isfinite(loss_error) or abs(loss_error) > loss_tolerance:
        return False
    if comparison.get("zero_gradient_matches"):
        return True
    return (comparison["cosine"] is not None and comparison["cosine"] >= cosine_minimum
            and comparison["relative_L2_error"] <= relative_error_maximum)
