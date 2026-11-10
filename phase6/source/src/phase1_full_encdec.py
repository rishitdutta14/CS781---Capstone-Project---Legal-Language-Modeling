"""A3-derived encoder-decoder with equivalent efficient attention and padding.

Original assignment and previous model implementations remain unchanged.
Each batch row belongs to one document, so no cross-document packing occurs.
"""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from assignment3_impl.q2.attention import CausalSelfAttention
from src.phase1_v2_models import (
    BidirectionalAttention, CrossAttention, EncoderDecoderLM, V2Config,
)


def production_config():
    return V2Config(architecture="encoder_decoder", vocab_size=16000,
                    d_model=1024, n_heads=16, d_ff=4096,
                    encoder_layers=8, decoder_layers=8, max_seq_len=1600,
                    dropout=0.1)


class PaddedEncoderAttention(BidirectionalAttention):
    def forward(self, x, valid=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(value):
            return value.reshape(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        mask = None if valid is None else valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            heads(q), heads(k), heads(v), attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class PaddedCausalAttention(CausalSelfAttention):
    def forward(self, x, valid=None):
        batch, length, width = x.shape
        if self.n_kv_heads != self.n_heads:
            raise ValueError("This workflow uses standard multi-head attention")
        def heads(projection):
            return projection(x).reshape(batch, length, self.n_heads, self.d_head).transpose(1, 2)
        mask = (None if valid is None else
                self.causal_mask[:length, :length][None, None] & valid[:, None, None, :])
        attended = F.scaled_dot_product_attention(
            heads(self.q_proj), heads(self.k_proj), heads(self.v_proj),
            attn_mask=mask, is_causal=valid is None, scale=self.attention_scale,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        return self.resid_dropout(self.out_proj(attended.transpose(1, 2).reshape(batch, length, width)))


class PaddedCrossAttention(CrossAttention):
    def forward(self, x, memory, source_valid=None):
        batch, length, width = x.shape
        def heads(value):
            return value.reshape(batch, -1, self.n_heads, self.d_head).transpose(1, 2)
        mask = None if source_valid is None else source_valid[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            heads(self.q_proj(x)), heads(self.k_proj(memory)), heads(self.v_proj(memory)),
            attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class FullCorpusEncoderDecoder(EncoderDecoderLM):
    def __init__(self, cfg=None, activation_checkpointing=False):
        cfg = cfg or production_config()
        if cfg.architecture != "encoder_decoder" or cfg.d_model % cfg.n_heads:
            raise ValueError("Expected a valid multi-head encoder-decoder configuration")
        super().__init__(cfg)
        self.activation_checkpointing = activation_checkpointing
        for block in self.encoder:
            block.attn = PaddedEncoderAttention(cfg)
        for block in self.decoder:
            block.self_attn = PaddedCausalAttention(
                cfg.d_model, cfg.n_heads, cfg.n_heads, cfg.max_seq_len, cfg.dropout)
            block.cross_attn = PaddedCrossAttention(cfg)

    def _encoder_block(self, block, x, source_valid):
        x = x + block.attn(block.ln1(x), source_valid)
        return x + block.ffn(block.ln2(x))

    def _decoder_block(self, block, x, memory, source_valid, target_valid):
        x = x + block.self_attn(block.ln1(x), target_valid)
        x = x + block.cross_attn(block.ln2(x), memory, source_valid)
        return x + block.ffn(block.ln3(x))

    def encode(self, source_ids, source_valid=None):
        if not 1 <= source_ids.shape[1] <= self.cfg.max_seq_len:
            raise ValueError("Source position capacity exceeded")
        x = self.tok_emb(source_ids) + self.pos_emb[:source_ids.shape[1]]
        for block in self.encoder:
            if self.activation_checkpointing and self.training:
                # Bind each block now; backward recomputation must not use the
                # final loop block accidentally.
                def run(value, layer=block):
                    return self._encoder_block(layer, value, source_valid)
                x = checkpoint(run, x, use_reentrant=False)
            else:
                x = self._encoder_block(block, x, source_valid)
        return self.encoder_norm(x)

    def decode(self, target_ids, memory, source_valid=None, target_valid=None):
        if not 1 <= target_ids.shape[1] <= self.cfg.max_seq_len:
            raise ValueError("Decoder position capacity exceeded")
        x = self.tok_emb(target_ids) + self.pos_emb[:target_ids.shape[1]]
        for block in self.decoder:
            if self.activation_checkpointing and self.training:
                def run(value, encoded, layer=block):
                    return self._decoder_block(layer, value, encoded, source_valid, target_valid)
                x = checkpoint(run, x, memory, use_reentrant=False)
            else:
                x = self._decoder_block(block, x, memory, source_valid, target_valid)
        return self.head(self.decoder_norm(x))

    def forward(self, source_ids, target_ids, source_valid=None, target_valid=None):
        if source_ids.shape[1] + target_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError("Combined source and decoder inputs exceed context budget")
        return self.decode(target_ids, self.encode(source_ids, source_valid), source_valid, target_valid)

    def decode_with_cache(self, *args, **kwargs):
        raise NotImplementedError("This training workflow does not use inference caching")


def parameter_breakdown(model):
    result = {name: sum(p.numel() for p in module.parameters()) for name, module in (
        ("shared_embedding_and_output", model.tok_emb), ("encoder", model.encoder),
        ("decoder", model.decoder), ("encoder_final_norm", model.encoder_norm),
        ("decoder_final_norm", model.decoder_norm))}
    total = sum(p.numel() for p in model.parameters())
    if sum(result.values()) != total or model.head.weight is not model.tok_emb.weight:
        raise ValueError("Unique parameter/shared-weight accounting failed")
    return {"components": result, "total_unique_parameters": total,
            "shared_output_weight_counted_once": True}
