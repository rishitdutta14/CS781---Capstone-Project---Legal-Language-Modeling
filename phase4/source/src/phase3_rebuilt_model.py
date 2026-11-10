"""RR heads for the two Phase-1 rebuilds, without modifying their source files.

Both sides of the encoder-decoder receive the same unlabelled sentence/context
IDs. Its encoder remains bidirectional and its decoder remains causal, with
cross-attention to the complete valid encoder input. Only current-sentence
decoder states are pooled. No role label is an input token.

LoRA follows Hu et al., https://arxiv.org/abs/2106.09685. Encoder Q/V updates
are separate low-rank factors around the existing fused QKV linear layer; K is
unchanged. SDPA replaces only the attention calculation, not its parameters.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.phase1_v2_models import V2Config, build_model
from src.phase3_model import LowRankLinear, attention_hidden, decoder_hidden
from src.phase3_submission import file_sha256


class LowRankDelta(nn.Module):
    def __init__(self, original, rank, alpha):
        super().__init__()
        width = original.in_features
        if not 0 < rank <= width or alpha <= 0:
            raise ValueError("Invalid fused-projection LoRA rank/alpha")
        self.A = nn.Parameter(original.weight.new_empty(rank, width))
        self.B = nn.Parameter(original.weight.new_zeros(width, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.scaling * F.linear(F.linear(x, self.A), self.B)


class FusedQvLoRA(nn.Module):
    """Preserve the fused pretrained matrix and adapt only its Q/V slices."""

    def __init__(self, original, rank, alpha):
        super().__init__()
        if original.out_features != 3 * original.in_features:
            raise ValueError("Expected a fused D -> 3D QKV projection")
        self.original = original.requires_grad_(False)
        self.q_lora = LowRankDelta(original, rank, alpha)
        self.v_lora = LowRankDelta(original, rank, alpha)

    def forward(self, x):
        q, k, v = self.original(x).chunk(3, dim=-1)
        return torch.cat((q + self.q_lora(x), k, v + self.v_lora(x)), dim=-1)


def split_heads(x, heads):
    batch, length, width = x.shape
    return x.view(batch, length, heads, width // heads).transpose(1, 2)


def encoder_attention(attention, x, valid_keys):
    q, k, v = attention.qkv(x).chunk(3, dim=-1)
    output = F.scaled_dot_product_attention(
        split_heads(q, attention.n_heads), split_heads(k, attention.n_heads),
        split_heads(v, attention.n_heads),
        attn_mask=valid_keys[:, None, None, :],
        dropout_p=attention.dropout.p if attention.training else 0.0)
    return attention.output(output.transpose(1, 2).contiguous().view_as(x))


def cross_attention(attention, x, memory, valid_keys):
    q = split_heads(attention.q_proj(x), attention.n_heads)
    k = split_heads(attention.k_proj(memory), attention.n_heads)
    v = split_heads(attention.v_proj(memory), attention.n_heads)
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=valid_keys[:, None, None, :],
        dropout_p=attention.dropout.p if attention.training else 0.0)
    return attention.output(output.transpose(1, 2).contiguous().view_as(x))


def encoder_hidden(backbone, ids, valid_keys):
    x = backbone.tok_emb(ids) + backbone.pos_emb[:ids.shape[1]].unsqueeze(0)
    for block in backbone.encoder:
        x = x + encoder_attention(block.attn, block.ln1(x), valid_keys)
        x = x + block.ffn(block.ln2(x))
    return backbone.encoder_norm(x)


def cross_decoder_hidden(backbone, ids, memory, valid_keys):
    x = backbone.tok_emb(ids) + backbone.pos_emb[:ids.shape[1]].unsqueeze(0)
    for block in backbone.decoder:
        x = x + attention_hidden(block.self_attn, block.ln1(x))
        x = x + cross_attention(block.cross_attn, block.ln2(x), memory, valid_keys)
        x = x + block.ffn(block.ln3(x))
    return backbone.decoder_norm(x)


def hidden_states(backbone, architecture, ids):
    if ids.ndim != 2 or ids.shape[1] > backbone.cfg.max_seq_len:
        raise ValueError("Invalid input dimensions or original context limit exceeded")
    valid = ids != 0
    if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Only nonempty, right-padded inputs are supported")
    if architecture == "dense":
        return decoder_hidden(backbone, ids)
    if architecture == "encoder_decoder":
        memory = encoder_hidden(backbone, ids, valid)
        return cross_decoder_hidden(backbone, ids, memory, valid)
    raise ValueError("Only the two explicitly requested rebuilt architectures are allowed")


class RebuiltRRClassifier(nn.Module):
    def __init__(self, backbone, architecture, rank=8, alpha=8.0):
        super().__init__()
        if architecture not in ("dense", "encoder_decoder"):
            raise ValueError("Unexpected rebuilt architecture")
        self.backbone = backbone.requires_grad_(False)
        self.architecture, self.method, self.targets = architecture, "lora", []

        def adapt(attention, prefix):
            for name in ("q_proj", "v_proj"):
                setattr(attention, name, LowRankLinear(getattr(attention, name), "lora", rank, alpha))
                self.targets.append(prefix + "." + name)

        if architecture == "dense":
            for number, block in enumerate(backbone.blocks):
                adapt(block.attn, f"backbone.blocks.{number}.attn")
        else:
            for number, block in enumerate(backbone.encoder):
                block.attn.qkv = FusedQvLoRA(block.attn.qkv, rank, alpha)
                self.targets.extend([f"backbone.encoder.{number}.attn.qkv.{side}_lora"
                                     for side in ("q", "v")])
            for number, block in enumerate(backbone.decoder):
                adapt(block.self_attn, f"backbone.decoder.{number}.self_attn")
                adapt(block.cross_attn, f"backbone.decoder.{number}.cross_attn")
        self.classifier = nn.Linear(backbone.cfg.d_model, 13)

    def forward(self, ids, target_mask, owners, sentence_count):
        if (target_mask.shape != ids.shape or target_mask.dtype != torch.bool
                or (target_mask & (ids == 0)).any()):
            raise ValueError("Invalid current-sentence pooling mask")
        if (owners.shape != (ids.shape[0],) or owners.dtype != torch.long
                or (owners < 0).any() or (owners >= sentence_count).any()):
            raise ValueError("Invalid original-sentence chunk ownership")
        counts = target_mask.sum(1)
        if (counts == 0).any():
            raise ValueError("Each chunk must retain current-sentence tokens")
        hidden = hidden_states(self.backbone, self.architecture, ids)
        sums = (hidden.float() * target_mask.unsqueeze(-1)).sum(1)
        pooled = sums.new_zeros(sentence_count, sums.shape[-1])
        totals = sums.new_zeros(sentence_count)
        pooled.index_add_(0, owners, sums)
        totals.index_add_(0, owners, counts.float())
        if (totals == 0).any():
            raise ValueError("An original sentence lost all its chunks")
        return self.classifier(pooled / totals.unsqueeze(1))


def load_rebuilt_backbone(path, expected_config, expected_sha256):
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Rebuilt Phase-1 checkpoint hash changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["model_config"] != expected_config or saved["architecture"] != expected_config["architecture"]:
        raise ValueError("Checkpoint and separately saved model configuration disagree")
    backbone = build_model(V2Config(**expected_config))
    backbone.load_state_dict(saved["model"], strict=True)
    for name, value in backbone.state_dict().items():
        if not torch.equal(value, saved["model"][name]) or not torch.isfinite(value).all():
            raise ValueError(f"Pretrained tensor failed exact/finite restoration: {name}")
    return backbone
