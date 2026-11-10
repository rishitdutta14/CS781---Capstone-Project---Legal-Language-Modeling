"""Optional execution experiments; the original v2/A3 modules stay unchanged.

Masked matmul keeps sparse *connections*, but computes a dense score matrix.
It is not an O(TK) sparse kernel. At our short context it may be faster than
materializing per-query key/value gathers; benchmark before using it.
"""

from types import MethodType

import torch
import torch.nn.functional as F

from src.phase1_v2_dataset import run_block
from src.phase1_v2_models import SparseLocalGlobalAttention


def masked_sparse_forward(self, x, cache=None):
    batch, length, width = x.shape
    def heads(projection):
        return projection(x).view(batch, length, self.n_heads, self.d_head).transpose(1, 2)
    q, k, v = heads(self.q_proj), heads(self.k_proj), heads(self.v_proj)
    if cache is not None:
        k = torch.cat((cache[0], k), dim=2)
        v = torch.cat((cache[1], v), dim=2)
    end = k.size(2)
    mask = self.execution_mask[end - length:end, :end]
    scores = (q @ k.transpose(-2, -1)) * self.scale
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = self.attn_dropout(torch.softmax(scores.float(), dim=-1).to(scores.dtype))
    result = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
    return self.resid_dropout(self.out_proj(result)), (k, v)


def install_sparse_masked_matmul(model):
    """Replace only execution; add no trainable/state-dict parameters."""
    for module in model.modules():
        if not isinstance(module, SparseLocalGlobalAttention):
            continue
        length = module.neighbor_indices.size(0)
        mask = torch.zeros(length, length, device=module.neighbor_indices.device, dtype=torch.bool)
        rows = torch.arange(length, device=mask.device)[:, None].expand_as(module.neighbor_indices)
        valid = module.neighbor_valid
        mask[rows[valid], module.neighbor_indices[valid]] = True
        module.register_buffer("execution_mask", mask, persistent=False)
        module.forward_with_cache = MethodType(masked_sparse_forward, module)
    return model


def decoder_batch_loss_sum(model, examples, *, checkpoint_blocks=False):
    """Right-pad document-safe causal examples; sum losses over real targets.

Future right-padding cannot affect a causal model's earlier real positions.
Do not use this for encoder-decoder models: their encoders need padding masks.
The caller divides by the entire optimizer group's target count, not batch size.
"""
    if getattr(model.cfg, "architecture", None) == "encoder_decoder":
        raise ValueError("encoder padding needs separate masks")
    inputs = [torch.cat((context, target[:-1])) for context, target in examples]
    labels = [torch.cat((torch.full_like(context[:-1], -100), target))
              for context, target in examples]
    length = max(len(row) for row in inputs)
    if length > model.cfg.max_seq_len:
        raise ValueError("training input exceeds the model context")
    ids = torch.stack([F.pad(row, (0, length - len(row)), value=0) for row in inputs])
    truth = torch.stack([F.pad(row, (0, length - len(row)), value=-100) for row in labels])
    x = model.tok_emb(ids) + model.pos_emb[:length].unsqueeze(0)
    for block in model.blocks:
        x = run_block(block, x) if checkpoint_blocks else block(x)
    selected = truth != -100
    logits = model.head(model.ln_f(x[selected]))
    return F.cross_entropy(logits.float(), truth[selected], reduction="sum")
