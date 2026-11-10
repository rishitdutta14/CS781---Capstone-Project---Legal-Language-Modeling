"""Decoder-only Transformer language model for Assignment 3 Question 1."""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

try:
    from .attention import CausalSelfAttention
except ImportError:
    from attention import CausalSelfAttention


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"
    attention_scale: float = None
    norm_first: bool = False


def sinusoidal_positional_encoding(max_len, d_model):
    if d_model % 2:
        raise ValueError("d_model must be even")
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(max_len, d_model)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_first = cfg.norm_first
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.d_model,
            cfg.n_heads,
            cfg.max_seq_len,
            cfg.dropout,
            cfg.attention_scale,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x):
        if self.norm_first:
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        x = self.ln1(x + self.attn(x))
        return self.ln2(x + self.ffn(x))


class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.pos_emb_type != "sinusoidal":
            raise ValueError("Question 1 uses sinusoidal positions")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer(
            "pos_emb",
            sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model),
            persistent=False,
        )
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        _, seq_len = idx.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError("sequence is longer than max_seq_len")
        x = self.tok_emb(idx) + self.pos_emb[:seq_len].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))
