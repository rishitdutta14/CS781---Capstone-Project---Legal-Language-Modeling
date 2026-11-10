"""Phase1 summarizers: unchanged attention weights, explicit adaptation and SDPA.

LoRA: Hu et al. (2021), arxiv:2106.09685.
LoRMA+: Bihany et al. (2025), identity inflation, equation8.
The adapter is a Pfeiffer-style single sequential FFN-output bottleneck;
its pre-normalization/zero-up initialization are recorded implementation choices.
"""

from contextlib import nullcontext
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from assignment3_impl.q2.model_arch import DecoderLM, LMConfig
from src.phase1_v2_models import build_model, V2Config
from src.phase1_full_encdec import FullCorpusEncoderDecoder, production_config
from src.phase3_model import LowRankLinear
from src.phase2_adaptation import LoRMAPlusLinear


class SequentialAdapter(nn.Module):
    def __init__(self, width, bottleneck=64):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.relu(self.down(self.norm(x))))


def self_attention(layer, x, cache=None, valid=None):
    b, length, width = x.shape
    heads, d = layer.n_heads, layer.d_head
    q = layer.q_proj(x).view(b, length, heads, d).transpose(1, 2)
    if hasattr(layer, 'kv_down_proj'):
        z = layer.kv_down_proj(x)
        k, v, kv_heads = layer.k_up_proj(z), layer.v_up_proj(z), heads
    else:
        k, v, kv_heads = layer.k_proj(x), layer.v_proj(x), layer.n_kv_heads
    k = k.view(b, length, kv_heads, d).transpose(1, 2)
    v = v.view(b, length, kv_heads, d).transpose(1, 2)
    if cache is not None:
        k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
    updated = (k, v)
    total = k.shape[2]
    if kv_heads != heads:
        k, v = k.repeat_interleave(heads // kv_heads, 1), v.repeat_interleave(heads // kv_heads, 1)
    mask = None
    causal = cache is None and valid is None
    if valid is not None or (cache is not None and length != 1):
        positions = torch.arange(total, device=x.device)
        query = torch.arange(total - length, total, device=x.device)
        mask = positions[None, :] <= query[:, None]
        mask = mask[None, None]
        if valid is not None: mask = mask & valid[:, None, None, :]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
        dropout_p=layer.attn_dropout.p if layer.training else 0., scale=layer.attention_scale)
    y = y.transpose(1, 2).contiguous().view(b, length, width)
    return layer.resid_dropout(layer.out_proj(y)), updated


def cross_attention(layer, x, memory, valid, cache=None):
    b, length, width = x.shape
    def heads(value):
        return value.view(b, -1, layer.n_heads, layer.d_head).transpose(1, 2)
    q = heads(layer.q_proj(x))
    k, v = (heads(layer.k_proj(memory)), heads(layer.v_proj(memory))) if cache is None else cache
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                      dropout_p=layer.dropout.p if layer.training else 0.)
    return layer.output(y.transpose(1, 2).contiguous().view(b, length, width)), (k, v)


class Summarizer(nn.Module):
    def __init__(self, backbone, method='full', encoder_decoder=False, rank=16, alpha=32., bottleneck=64):
        super().__init__()
        self.backbone, self.method, self.encoder_decoder = backbone, method, encoder_decoder
        self.targets = []
        backbone.requires_grad_(method in ('full', 'kd'))
        self.adapters = nn.ModuleList()
        if encoder_decoder and method != 'full':
            raise ValueError('This campaign fully fine-tunes its own encoder-decoder teacher')
        if not encoder_decoder:
            for i, block in enumerate(backbone.blocks):
                if method in ('lora', 'lorma'):
                    names = ('q_proj', 'v_up_proj') if hasattr(block.attn, 'kv_down_proj') else ('q_proj', 'v_proj')
                    for name in names:
                        original = getattr(block.attn, name)
                        adapted = (LowRankLinear(original, 'lora', rank, alpha) if method == 'lora'
                                   else LoRMAPlusLinear(original, rank, alpha))
                        setattr(block.attn, name, adapted)
                        self.targets.append('backbone.blocks.%d.attn.%s' % (i, name))
                elif method == 'adapter':
                    self.adapters.append(SequentialAdapter(backbone.cfg.d_model, bottleneck))
                    self.targets.append('adapters.%d' % i)
                elif method not in ('full', 'kd'):
                    raise ValueError('Unexpected adaptation method')

    def precision(self):
        return torch.autocast('cuda', dtype=torch.bfloat16) if next(self.parameters()).is_cuda else nullcontext()

    def encode(self, ids):
        with self.precision():
            return self.backbone.encode(ids, ids != 0)

    def forward(self, ids, source=None, memory=None, caches=None, start=0, valid=None, last=False, output_start=0):
        base = self.backbone
        if start + ids.shape[1] > base.cfg.max_seq_len:
            raise ValueError('Original positional capacity exceeded')
        # Right padding cannot influence earlier causal predictions. Keep the
        # explicit key mask only for internal holes (short-source padding).
        # This permits the fused causal kernel for the common full-source case.
        if valid is not None and not bool((valid[:, 1:] & ~valid[:, :-1]).any()):
            valid = None
        with self.precision():
            x = base.tok_emb(ids) + base.pos_emb[start:start + ids.shape[1]]
            updated = []
            if self.encoder_decoder:
                if memory is None: memory = self.encode(source)
                source_valid = source != 0
                stack = base.decoder
            else:
                stack = base.blocks
            if caches is None: caches = [None] * len(stack)
            if len(caches) != len(stack): raise ValueError('Decoder cache count mismatch')
            for i, (block, cache) in enumerate(zip(stack, caches)):
                if self.encoder_decoder:
                    self_cache, cross_cache = (None, None) if cache is None else cache
                    attended, self_cache = self_attention(block.self_attn, block.ln1(x), self_cache, valid)
                    x = x + attended
                    attended, cross_cache = cross_attention(block.cross_attn, block.ln2(x), memory, source_valid, cross_cache)
                    x = x + attended
                    x = x + block.ffn(block.ln3(x))
                    updated.append((self_cache, cross_cache))
                else:
                    if not block.norm_first: raise ValueError('This campaign requires inspected pre-LN Phase1 models')
                    attended, cache = self_attention(block.attn, block.ln1(x), cache, valid)
                    x = x + attended
                    x = x + block.ffn(block.ln2(x))
                    if self.method == 'adapter': x = self.adapters[i](x)
                    updated.append(cache)
            norm = base.decoder_norm if self.encoder_decoder else base.ln_f
            if last: x = x[:, -1:]
            elif output_start: x = x[:, output_start:]
            return base.head(norm(x)), updated


def build(config, device):
    state = torch.load(config['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if config['architecture'] == 'encdec251':
        backbone = FullCorpusEncoderDecoder(production_config())
    elif config['architecture'] == 'dense':
        if state['model_config'] != config['model_config']:
            raise ValueError('Rebuilt Phase1 configuration changed')
        backbone = build_model(V2Config(**config['model_config']))
    else:
        if state['config']['model'] != config['model_config']:
            raise ValueError('Original Phase1 configuration changed')
        backbone = DecoderLM(LMConfig(**config['model_config']))
    backbone.load_state_dict(state['model'], strict=True)
    for name, value in backbone.state_dict().items():
        if not torch.equal(value, state['model'][name]) or not bool(torch.isfinite(value).all()):
            raise ValueError('Phase1 tensor was not restored exactly: ' + name)
    del state
    return Summarizer(backbone, config['method'], config['architecture'] == 'encdec251',
                      config['rank'], config['alpha'], config['bottleneck']).to(device)
