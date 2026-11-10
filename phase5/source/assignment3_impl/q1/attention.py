"""Multi-head causal self-attention, implemented from raw tensor ops."""

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        attention_scale=None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attention_scale = (
            1.0 / math.sqrt(self.d_head)
            if attention_scale is None
            else float(attention_scale)
        )

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.attention_scale
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        return self.resid_dropout(self.out_proj(output))
