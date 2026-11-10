"""Four distinct from-scratch Phase 1 language-model architectures.

The dense decoder is the Assignment 3 implementation. Other models reuse its
hand-written attention, feed-forward layers, and positional encoding where
those pieces apply, without editing the Assignment 3 deliverables.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from assignment3_impl.q2.attention import CausalSelfAttention
from assignment3_impl.q2.model_arch import (
    Block,
    DecoderLM,
    FeedForward,
    LMConfig,
    sinusoidal_positional_encoding,
)


@dataclass(frozen=True)
class V2Config:
    architecture: str
    vocab_size: int = 16_000
    d_model: int = 768
    n_heads: int = 8
    d_ff: int = 3_072
    layers: int = 12
    max_seq_len: int = 1_600
    dropout: float = 0.1
    encoder_layers: int = 7
    decoder_layers: int = 7
    local_window: int = 256
    global_stride: int = 64


def default_config(architecture: str, vocab_size: int = 16_000) -> V2Config:
    if architecture == "encoder_decoder":
        return V2Config(
            architecture=architecture,
            vocab_size=vocab_size,
            d_model=640,
            d_ff=2_560,
        )
    if architecture not in {"dense", "sparse", "hybrid"}:
        raise ValueError(f"unknown architecture: {architecture}")
    return V2Config(architecture=architecture, vocab_size=vocab_size)


def a3_config(cfg: V2Config) -> LMConfig:
    return LMConfig(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        pos_emb_type="sinusoidal",
        norm_first=True,
        attention_type="standard",
    )


class SparseLocalGlobalAttention(nn.Module):
    """Causal sliding-window attention with periodic global key positions."""

    def __init__(self, cfg: V2Config):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.scale = 1 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        width = cfg.local_window + math.ceil(cfg.max_seq_len / cfg.global_stride)
        indices = torch.zeros(cfg.max_seq_len, width, dtype=torch.long)
        valid = torch.zeros(cfg.max_seq_len, width, dtype=torch.bool)
        for position in range(cfg.max_seq_len):
            local = range(max(0, position - cfg.local_window + 1), position + 1)
            global_keys = range(0, position + 1, cfg.global_stride)
            neighbors = sorted(set(local) | set(global_keys))
            indices[position, :len(neighbors)] = torch.tensor(neighbors)
            valid[position, :len(neighbors)] = True
        self.register_buffer("neighbor_indices", indices, persistent=False)
        self.register_buffer("neighbor_valid", valid, persistent=False)

    def forward_with_cache(self, x: torch.Tensor, cache=None):
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        total_length = k.size(2)
        start = total_length - length
        neighbors = self.neighbor_indices[start:total_length]
        valid = self.neighbor_valid[start:total_length]
        keys = k[:, :, neighbors, :]
        values = v[:, :, neighbors, :]
        scores = torch.matmul(q.unsqueeze(-2), keys.transpose(-2, -1)).squeeze(-2) * self.scale
        scores = scores.masked_fill(~valid[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
        result = torch.matmul(weights.unsqueeze(-2), values).squeeze(-2)
        result = result.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.out_proj(result)), (k, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cache(x)[0]


class SparseBlock(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = SparseLocalGlobalAttention(cfg)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

    def forward_with_cache(self, x: torch.Tensor, cache=None):
        attended, cache = self.attn.forward_with_cache(self.ln1(x), cache)
        x = x + attended
        return x + self.ffn(self.ln2(x)), cache


class GatedCausalConv(nn.Module):
    """A causal depthwise convolution followed by a learned content gate."""

    def __init__(self, cfg: V2Config, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(
            cfg.d_model, cfg.d_model, kernel_size, groups=cfg.d_model, bias=False
        )
        self.gate = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.output = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward_with_cache(self, x: torch.Tensor, cache=None):
        channels = x.transpose(1, 2)
        if cache is None:
            channels = F.pad(channels, (self.kernel_size - 1, 0))
        else:
            channels = torch.cat((cache, channels), dim=2)
        convolved = self.depthwise(channels).transpose(1, 2)
        content, gate = self.gate(convolved).chunk(2, dim=-1)
        result = self.dropout(self.output(torch.tanh(content) * torch.sigmoid(gate)))
        return result, channels[:, :, -(self.kernel_size - 1):]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cache(x)[0]


class ConvBlock(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.conv = GatedCausalConv(cfg)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.conv(self.ln1(x))
        return x + self.ffn(self.ln2(x))

    def forward_with_cache(self, x: torch.Tensor, cache=None):
        convolved, cache = self.conv.forward_with_cache(self.ln1(x), cache)
        x = x + convolved
        return x + self.ffn(self.ln2(x)), cache


class CausalDecoderVariant(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer(
            "pos_emb", sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model),
            persistent=False,
        )
        if cfg.architecture == "sparse":
            self.blocks = nn.ModuleList(SparseBlock(cfg) for _ in range(cfg.layers))
        elif cfg.architecture == "hybrid":
            base = a3_config(cfg)
            self.blocks = nn.ModuleList(
                Block(base) if layer % 2 == 0 else ConvBlock(cfg)
                for layer in range(cfg.layers)
            )
        else:
            raise ValueError("expected sparse or hybrid architecture")
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.size(1)
        if length > self.cfg.max_seq_len:
            raise ValueError("sequence is longer than the model context")
        x = self.tok_emb(ids) + self.pos_emb[:length].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def forward_with_cache(self, ids, caches=None, start_pos=0, last_token_only=False):
        end = start_pos + ids.size(1)
        if end > self.cfg.max_seq_len:
            raise ValueError("sequence is longer than the model context")
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError("one cache entry is required per decoder block")
        x = self.tok_emb(ids) + self.pos_emb[start_pos:end].unsqueeze(0)
        updated = []
        for block, cache in zip(self.blocks, caches):
            x, cache = block.forward_with_cache(x, cache)
            updated.append(cache)
        if last_token_only:
            x = x[:, -1:]
        return self.head(self.ln_f(x)), updated


class BidirectionalAttention(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.output = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(value):
            return value.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = self.dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
        result = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(result)


class CrossAttention(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.output = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward_with_cache(self, x: torch.Tensor, memory: torch.Tensor, cache=None):
        batch, target_len, width = x.shape
        source_len = memory.size(1)
        def heads(value, length):
            return value.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        q = heads(self.q_proj(x), target_len)
        if cache is None:
            k = heads(self.k_proj(memory), source_len)
            v = heads(self.v_proj(memory), source_len)
        else:
            k, v = cache
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = self.dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
        result = (weights @ v).transpose(1, 2).contiguous().view(batch, target_len, width)
        return self.output(result), (k, v)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cache(x, memory)[0]


class EncoderBlock(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = BidirectionalAttention(cfg)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class CrossDecoderBlock(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ln3 = nn.LayerNorm(cfg.d_model)
        self.self_attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.n_heads, cfg.max_seq_len, cfg.dropout
        )
        self.cross_attn = CrossAttention(cfg)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), memory)
        return x + self.ffn(self.ln3(x))

    def forward_with_cache(self, x: torch.Tensor, memory: torch.Tensor, cache=None):
        self_cache, cross_cache = (None, None) if cache is None else cache
        attended, self_cache = self.self_attn.forward_with_cache(self.ln1(x), self_cache)
        x = x + attended
        attended, cross_cache = self.cross_attn.forward_with_cache(self.ln2(x), memory, cross_cache)
        x = x + attended
        return x + self.ffn(self.ln3(x)), (self_cache, cross_cache)


class EncoderDecoderLM(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer(
            "pos_emb", sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model),
            persistent=False,
        )
        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.encoder_layers))
        self.decoder = nn.ModuleList(CrossDecoderBlock(cfg) for _ in range(cfg.decoder_layers))
        self.encoder_norm = nn.LayerNorm(cfg.d_model)
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def encode(self, source_ids: torch.Tensor) -> torch.Tensor:
        if source_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("source is longer than the model context")
        x = self.tok_emb(source_ids) + self.pos_emb[:source_ids.size(1)].unsqueeze(0)
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def decode(self, target_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if target_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("target is longer than the model context")
        x = self.tok_emb(target_ids) + self.pos_emb[:target_ids.size(1)].unsqueeze(0)
        for block in self.decoder:
            x = block(x, memory)
        return self.head(self.decoder_norm(x))

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        return self.decode(target_ids, self.encode(source_ids))

    def decode_with_cache(self, ids, memory, caches=None, start_pos=0, last_token_only=False):
        end = start_pos + ids.size(1)
        if end > self.cfg.max_seq_len:
            raise ValueError("target is longer than the model context")
        if caches is None:
            caches = [None] * len(self.decoder)
        if len(caches) != len(self.decoder):
            raise ValueError("one cache entry is required per cross-attention decoder block")
        x = self.tok_emb(ids) + self.pos_emb[start_pos:end].unsqueeze(0)
        updated = []
        for block, cache in zip(self.decoder, caches):
            x, cache = block.forward_with_cache(x, memory, cache)
            updated.append(cache)
        if last_token_only:
            x = x[:, -1:]
        return self.head(self.decoder_norm(x)), updated


def build_model(cfg: V2Config) -> nn.Module:
    if cfg.architecture == "dense":
        return DecoderLM(a3_config(cfg))
    if cfg.architecture in {"sparse", "hybrid"}:
        return CausalDecoderVariant(cfg)
    if cfg.architecture == "encoder_decoder":
        return EncoderDecoderLM(cfg)
    raise ValueError(f"unknown architecture: {cfg.architecture}")


def model_config_dict(cfg: V2Config) -> dict:
    return asdict(cfg)
