"""Rotary positional embeddings using the Assignment 3 split-half layout."""

import torch


def build_rope_cache(max_seq_len, d_head, base=10000.0):
    assert d_head % 2 == 0
    indices = torch.arange(0, d_head, 2, dtype=torch.float32)
    inverse_frequencies = 1.0 / (base ** (indices / d_head))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse_frequencies)
    angles = torch.cat([frequencies, frequencies], dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=x.dtype, device=x.device)
    return x * cos + rotate_half(x) * sin
