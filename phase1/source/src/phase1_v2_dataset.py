"""Document-safe examples shared by the four Phase 1 models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


SOURCES = ("acts", "high_courts", "supreme_court", "tribunals")


class PlannedCorpus:
    def __init__(self, encoded_dir: Path, plan_dir: Path, bos_token_id: int):
        self.encoded_dir = Path(encoded_dir)
        self.plan_dir = Path(plan_dir)
        self.bos_token_id = bos_token_id
        self.arrays = {
            split: [
                np.memmap(self.encoded_dir / f"{split}_{source}.bin", dtype=np.uint16, mode="r")
                if (self.encoded_dir / f"{split}_{source}.bin").stat().st_size
                else np.empty(0, dtype=np.uint16)
                for source in SOURCES
            ]
            for split in ("train", "val")
        }
        self.rows = {
            "train": np.load(self.plan_dir / "train_examples.npy", mmap_mode="r"),
            "val": np.load(self.plan_dir / "val_examples.npy", mmap_mode="r"),
        }

    def example(self, split: str, row_number: int, context_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, offset, length, start, target_length = (
            int(value) for value in self.rows[split][row_number]
        )
        document = self.arrays[split][source][offset:offset + length]
        prefix_start = max(0, start - context_size)
        context = np.asarray(document[prefix_start:start], dtype=np.int64)
        target = np.asarray(document[start:start + target_length], dtype=np.int64)
        if len(context) == 0:
            context = np.asarray([self.bos_token_id], dtype=np.int64)
        if len(target) != target_length:
            raise RuntimeError("training plan references a truncated document")
        return torch.from_numpy(context), torch.from_numpy(target)


def continuation_loss(
    model,
    architecture: str,
    context: torch.Tensor,
    target: torch.Tensor,
    bos_token_id: int,
    *,
    checkpoint_blocks: bool = False,
) -> torch.Tensor:
    """Every target position receives exactly one next-token loss."""

    context = context.unsqueeze(0)
    target = target.unsqueeze(0)
    if architecture == "encoder_decoder":
        start = torch.full((1, 1), bos_token_id, dtype=target.dtype, device=target.device)
        decoder_input = torch.cat((start, target[:, :-1]), dim=1)
        if checkpoint_blocks:
            logits = checkpointed_encoder_decoder(model, context, decoder_input)
        else:
            logits = model(context, decoder_input)
    else:
        inputs = torch.cat((context, target[:, :-1]), dim=1)
        if checkpoint_blocks:
            logits = checkpointed_decoder(model, inputs, context.size(1) - 1)
        else:
            logits = model(inputs)[:, context.size(1) - 1:]
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1))


def run_block(block, x, memory=None):
    if torch.is_grad_enabled():
        if memory is None:
            return checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
        return checkpoint(block, x, memory, use_reentrant=False, preserve_rng_state=True)
    return block(x) if memory is None else block(x, memory)


def checkpointed_decoder(model, inputs, target_start):
    """Use the existing blocks, recomputing their activations during backward."""
    length = inputs.size(1)
    if length > model.cfg.max_seq_len:
        raise ValueError("training input exceeds the model context")
    x = model.tok_emb(inputs) + model.pos_emb[:length].unsqueeze(0)
    for block in model.blocks:
        x = run_block(block, x)
    # The conditioning prefix receives gradients through the blocks, but does
    # not need vocabulary logits because its loss is masked.
    return model.head(model.ln_f(x[:, target_start:]))


def checkpointed_encoder_decoder(model, source, target_input):
    if max(source.size(1), target_input.size(1)) > model.cfg.max_seq_len:
        raise ValueError("training input exceeds the model context")
    memory = model.tok_emb(source) + model.pos_emb[:source.size(1)].unsqueeze(0)
    for block in model.encoder:
        memory = run_block(block, memory)
    memory = model.encoder_norm(memory)
    x = model.tok_emb(target_input) + model.pos_emb[:target_input.size(1)].unsqueeze(0)
    for block in model.decoder:
        x = run_block(block, x, memory)
    return model.head(model.decoder_norm(x))
