"""Manual MHA, GQA, MQA, and the Q2 latent-attention variant."""

import math

import torch
import torch.nn as nn


def _causal_attention(q, k, v, mask, scale, dropout):
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = dropout(torch.softmax(scores, dim=-1))
    return weights @ v


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        attention_scale=None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.attention_scale = (
            1.0 / math.sqrt(self.d_head)
            if attention_scale is None
            else float(attention_scale)
        )

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward_with_cache(self, x, cache=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.n_kv_heads, self.d_head
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.n_kv_heads, self.d_head
        ).transpose(1, 2)

        if cache is not None:
            old_k, old_v = cache
            k = torch.cat((old_k, k), dim=2)
            v = torch.cat((old_v, v), dim=2)
        new_cache = (k, v)

        total_len = k.size(2)
        start = total_len - seq_len
        attended_k = k
        attended_v = v
        if self.n_rep > 1:
            attended_k = k.repeat_interleave(self.n_rep, dim=1)
            attended_v = v.repeat_interleave(self.n_rep, dim=1)

        mask = self.causal_mask[start:total_len, :total_len]
        output = _causal_attention(
            q,
            attended_k,
            attended_v,
            mask,
            self.attention_scale,
            self.attn_dropout,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_head
        )
        return self.resid_dropout(self.out_proj(output)), new_cache

    def forward(self, x):
        output, _ = self.forward_with_cache(x)
        return output


class MultiHeadLatentAttention(nn.Module):
    """Q2's shared low-rank K/V representation."""

    def __init__(
        self,
        d_model,
        n_heads,
        kv_latent_dim,
        max_seq_len,
        dropout=0.0,
        attention_scale=None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert 0 < kv_latent_dim < d_model

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.kv_latent_dim = kv_latent_dim
        self.attention_scale = (
            1.0 / math.sqrt(self.d_head)
            if attention_scale is None
            else float(attention_scale)
        )

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down_proj = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.k_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward_with_cache(self, x, cache=None):
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)

        latent = self.kv_down_proj(x)
        k = self.k_up_proj(latent).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)
        v = self.v_up_proj(latent).view(
            batch_size, seq_len, self.n_heads, self.d_head
        ).transpose(1, 2)

        if cache is not None:
            old_k, old_v = cache
            k = torch.cat((old_k, k), dim=2)
            v = torch.cat((old_v, v), dim=2)
        new_cache = (k, v)
        total_len = k.size(2)
        start = total_len - seq_len
        mask = self.causal_mask[start:total_len, :total_len]
        output = _causal_attention(
            q, k, v, mask, self.attention_scale, self.attn_dropout
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        return self.resid_dropout(self.out_proj(output)), new_cache

    def forward(self, x):
        output, _ = self.forward_with_cache(x)
        return output
