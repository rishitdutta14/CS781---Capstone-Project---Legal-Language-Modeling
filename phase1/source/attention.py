"""Manual MHA, GQA, and MQA with optional rotary positions."""

import math

import torch
import torch.nn as nn

try:
    from .rope import apply_rope, build_rope_cache
except ImportError:
    from rope import apply_rope, build_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_kv_heads,
        max_seq_len,
        dropout=0.0,
        use_rope=False,
        attention_scale=None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.attention_scale = (
            1.0 / math.sqrt(self.d_head)
            if attention_scale is None
            else float(attention_scale)
        )

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(
            d_model, n_kv_heads * self.d_head, bias=False
        )
        self.v_proj = nn.Linear(
            d_model, n_kv_heads * self.d_head, bias=False
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        if use_rope:
            cos, sin = build_rope_cache(max_seq_len, self.d_head)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.n_kv_heads, self.d_head
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.n_kv_heads, self.d_head
        ).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, self.rope_cos[:seq_len], self.rope_sin[:seq_len])
            k = apply_rope(k, self.rope_cos[:seq_len], self.rope_sin[:seq_len])
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * self.attention_scale
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        return self.resid_dropout(self.out_proj(output))
