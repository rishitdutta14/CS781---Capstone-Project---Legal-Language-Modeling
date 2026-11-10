"""Sentence classification over the unchanged A3 decoder parameters.

References: LoRA (arxiv:2106.09685), IA3 (arxiv:2205.05638), and
LoRMA+ identity inflation (ACL 2025.findings-acl.527, equation 8).
Original assignment files are not edited. SDPA evaluates the same causal
attention algebra, retaining GQA/MQA sharing and MLA's latent projections.
"""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from assignment3_impl.q2.model_arch import DecoderLM, LMConfig


class LowRankLinear(nn.Module):
    def __init__(self, original, method, rank=8, alpha=8.0):
        super().__init__()
        if method not in ("lora", "lorma_plus") or not isinstance(original, nn.Linear):
            raise ValueError("Expected a linear projection and a declared low-rank method")
        if not 0 < rank <= min(original.in_features, original.out_features) or alpha <= 0:
            raise ValueError("Invalid adapter rank/alpha")
        self.original = original.requires_grad_(False)
        self.method, self.scaling = method, alpha / rank
        width = original.in_features if method == "lora" else original.out_features
        self.A = nn.Parameter(original.weight.new_empty(rank, width))
        self.B = nn.Parameter(original.weight.new_zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        projected = F.linear(x, self.original.weight)
        # LoRA: W0 x + s BA x. LoRMA+: (I + s BA) W0 x.
        adapter_input = x if self.method == "lora" else projected
        result = projected + self.scaling * F.linear(F.linear(adapter_input, self.A), self.B)
        return result if self.original.bias is None else result + self.original.bias


class ActivationScale(nn.Module):
    def __init__(self, original, scale_input=False):
        super().__init__()
        self.original = original.requires_grad_(False)
        self.scale_input = scale_input
        width = original.in_features if scale_input else original.out_features
        self.scale = nn.Parameter(original.weight.new_ones(width))

    def forward(self, x):
        return self.original(x * self.scale) if self.scale_input else self.original(x) * self.scale


def attention_hidden(attention, x):
    batch, length, width = x.shape
    heads, head_width = attention.n_heads, attention.d_head
    q = attention.q_proj(x).view(batch, length, heads, head_width).transpose(1, 2)
    if hasattr(attention, "kv_down_proj"):
        latent = attention.kv_down_proj(x)
        k, v = attention.k_up_proj(latent), attention.v_up_proj(latent)
        kv_heads = heads
    else:
        k, v = attention.k_proj(x), attention.v_proj(x)
        kv_heads = attention.n_kv_heads
    k = k.view(batch, length, kv_heads, head_width).transpose(1, 2)
    v = v.view(batch, length, kv_heads, head_width).transpose(1, 2)
    if kv_heads != heads:
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.repeat_interleave(heads // kv_heads, dim=1)
    output = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=attention.attention_scale,
        dropout_p=attention.attn_dropout.p if attention.training else 0.0)
    output = output.transpose(1, 2).contiguous().view(batch, length, width)
    return attention.resid_dropout(attention.out_proj(output))


def decoder_hidden(backbone, ids):
    length = ids.shape[1]
    if length > backbone.cfg.max_seq_len:
        raise ValueError("Input exceeds the original context capacity")
    x = backbone.tok_emb(ids) + backbone.pos_emb[:length].unsqueeze(0)
    for block in backbone.blocks:
        if block.norm_first:
            x = x + attention_hidden(block.attn, block.ln1(x))
            x = x + block.ffn(block.ln2(x))
        else:
            x = block.ln1(x + attention_hidden(block.attn, x))
            x = block.ln2(x + block.ffn(x))
    return backbone.ln_f(x)


class RRClassifier(nn.Module):
    def __init__(self, backbone, method, rank=8, alpha=8.0, labels=13):
        super().__init__()
        self.backbone = backbone
        backbone.requires_grad_(method == "knowledge_distillation")
        self.targets = []
        for number, block in enumerate(backbone.blocks):
            if method in ("lora", "lorma_plus"):
                if hasattr(block.attn, "kv_down_proj"):
                    raise ValueError("This campaign maps low-rank methods to standard attention")
                for name in ("q_proj", "v_proj"):
                    setattr(block.attn, name, LowRankLinear(getattr(block.attn, name), method, rank, alpha))
                    self.targets.append(f"backbone.blocks.{number}.attn.{name}")
            elif method == "ia3":
                if not hasattr(block.attn, "kv_down_proj"):
                    raise ValueError("This campaign maps IA3 to the MLA backbone")
                for name in ("k_up_proj", "v_up_proj"):
                    setattr(block.attn, name, ActivationScale(getattr(block.attn, name)))
                    self.targets.append(f"backbone.blocks.{number}.attn.{name}")
                block.ffn.net[3] = ActivationScale(block.ffn.net[3], scale_input=True)
                self.targets.append(f"backbone.blocks.{number}.ffn.net.3")
            elif method != "knowledge_distillation":
                raise ValueError("Unknown RR adaptation method")
        self.classifier = nn.Linear(backbone.cfg.d_model, labels)
        self.method = method

    def forward(self, ids, target_mask, owners, sentence_count):
        if target_mask.shape != ids.shape or target_mask.dtype != torch.bool:
            raise ValueError("Target mask must be boolean and match token IDs")
        counts = target_mask.sum(1)
        if (counts == 0).any():
            raise ValueError("Every chunk must contain target tokens")
        hidden = decoder_hidden(self.backbone, ids)
        # Right padding lies after every valid token. Causal attention therefore
        # prevents it from influencing targets; no left padding is permitted.
        sums = (hidden.float() * target_mask.unsqueeze(-1)).sum(1)
        pooled = sums.new_zeros(sentence_count, sums.shape[-1])
        pooled.index_add_(0, owners, sums)
        token_counts = sums.new_zeros(sentence_count)
        token_counts.index_add_(0, owners, counts.float())
        if (token_counts == 0).any():
            raise ValueError("Chunk aggregation omitted a sentence")
        return self.classifier(pooled / token_counts.unsqueeze(1))


def load_backbone(path, expected_config=None):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]["model"]
    if expected_config is not None and config != expected_config:
        raise ValueError("Phase1 checkpoint config differs from the inspected binding")
    backbone = DecoderLM(LMConfig(**config))
    backbone.load_state_dict(saved["model"], strict=True)
    for name, tensor in backbone.state_dict().items():
        if not torch.equal(tensor, saved["model"][name]) or not torch.isfinite(tensor).all():
            raise ValueError(f"Backbone tensor was not restored exactly: {name}")
    return backbone


def parameter_summary(model):
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(trainable.values()), "trainable": trainable,
            "adapter_targets": model.targets, "method": model.method}


def frozen_sha256(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            digest.update(f"{name}:{tuple(p.shape)}:{p.dtype}\n".encode())
            digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
