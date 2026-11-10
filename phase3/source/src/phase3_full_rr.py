"""RR adaptation of the 251M full-corpus encoder-decoder, with its own tokenizer.

The saved model enforces a combined encoder+decoder budget of1600. Reserve257
source slots (BOS+256 preceding same-document tokens) and1343 decoder slots
(BOS+1342 current-sentence tokens). Overlong sentences use multiple pieces with
token-weighted pooling, never a dropped tail or multiple sentence-level losses.
"""

from pathlib import Path

import torch

from src.phase1_full_encdec import FullCorpusEncoderDecoder, production_config
from src.phase3_rebuilt_model import RebuiltRRClassifier
from src.phase3_submission import file_sha256


def full_chunks(row, context_tokens=256, capacity=1600):
    if not 0 <= context_tokens < capacity - 2:
        raise ValueError("Invalid combined encoder/decoder context budget")
    doc, number = row["doc"], row["sentence_index"]
    ids, offsets = doc["ids"], doc["offsets"]
    a, b = map(int, offsets[number:number + 2])
    target = ids[a:b].tolist()
    if not target:
        raise ValueError("An original sentence has no target tokens")
    preceding, previous = [], number - 1
    while previous >= 0 and len(preceding) < context_tokens:
        a, b = map(int, offsets[previous:previous + 2])
        preceding = ids[a:b].tolist() + [3] + preceding
        previous -= 1
    width = capacity - context_tokens - 2
    result = []
    for start in range(0, len(target), width):
        before = preceding + target[:start]
        source = [2] + (before[-context_tokens:] if context_tokens else [])
        decoder = [2] + target[start:start + width]
        if len(source) + len(decoder) > capacity:
            raise ValueError("Combined context limit exceeded")
        result.append((source, decoder))
    if sum(len(decoder) - 1 for _, decoder in result) != len(target):
        raise ValueError("Sentence chunking lost or duplicated target tokens")
    return result


def full_collate(rows, context_tokens=256, capacity=1600, device="cpu"):
    pieces, owners = [], []
    for owner, row in enumerate(rows):
        chunks = full_chunks(row, context_tokens, capacity)
        pieces.extend(chunks)
        owners.extend([owner] * len(chunks))
    source = torch.zeros(len(pieces), max(len(s) for s, _ in pieces), dtype=torch.long)
    decoder = torch.zeros(len(pieces), max(len(t) for _, t in pieces), dtype=torch.long)
    mask = torch.zeros_like(decoder, dtype=torch.bool)
    for i, (s, t) in enumerate(pieces):
        source[i, :len(s)] = torch.tensor(s)
        decoder[i, :len(t)] = torch.tensor(t)
        mask[i, 1:len(t)] = True
    if source.shape[1] + decoder.shape[1] > capacity:
        raise ValueError("Padded source+decoder dimensions exceed the saved combined budget")
    return {"source_ids": source.to(device), "decoder_ids": decoder.to(device),
        "target_mask": mask.to(device), "owners": torch.tensor(owners, dtype=torch.long, device=device),
        "sentence_count": len(rows)}


def full_decoder_hidden(base, source_ids, decoder_ids):
    if source_ids.shape[1] + decoder_ids.shape[1] > base.cfg.max_seq_len:
        raise ValueError("Combined original context capacity exceeded")
    source_valid, target_valid = source_ids != 0, decoder_ids != 0
    for valid in (source_valid, target_valid):
        if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Nonempty right-padded inputs required")
    memory = base.encode(source_ids, source_valid)
    x = base.tok_emb(decoder_ids) + base.pos_emb[:decoder_ids.shape[1]]
    for block in base.decoder:
        x = base._decoder_block(block, x, memory, source_valid, target_valid)
    return base.decoder_norm(x)


class FullRRClassifier(RebuiltRRClassifier):
    def __init__(self, backbone, rank=8, alpha=8.):
        super().__init__(backbone, "encoder_decoder", rank, alpha)

    def forward(self, source_ids, decoder_ids, target_mask, owners, sentence_count):
        if (source_ids.shape[0] != decoder_ids.shape[0] or target_mask.dtype != torch.bool
                or target_mask.shape != decoder_ids.shape or (target_mask & (decoder_ids == 0)).any()
                or target_mask[:, 0].any()):
            raise ValueError("Only current-sentence nonpadding decoder tokens may be pooled")
        if (owners.shape != (decoder_ids.shape[0],) or owners.dtype != torch.long
                or (owners < 0).any() or (owners >= sentence_count).any()):
            raise ValueError("Invalid sentence chunk ownership")
        hidden = full_decoder_hidden(self.backbone, source_ids, decoder_ids)
        sums = (hidden.float() * target_mask.unsqueeze(-1)).sum(1)
        counts = target_mask.sum(1).float()
        pooled = sums.new_zeros(sentence_count, sums.shape[-1])
        totals = sums.new_zeros(sentence_count)
        pooled.index_add_(0, owners, sums)
        totals.index_add_(0, owners, counts)
        if (counts == 0).any() or (totals == 0).any():
            raise ValueError("A target chunk or original sentence was omitted")
        return self.classifier(pooled / totals.unsqueeze(1))


def load_full_base(path, expected_sha, tokenizer_sha):
    if file_sha256(path) != expected_sha:
        raise ValueError("Selected251M Phase1 checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if saved["bindings"]["tokenizer_sha256"] != tokenizer_sha:
        raise ValueError("Checkpoint and full-corpus tokenizer do not match")
    base = FullCorpusEncoderDecoder(production_config(), activation_checkpointing=False)
    base.load_state_dict(saved["model"], strict=True)
    if sum(p.numel() for p in base.parameters()) != 251432960:
        raise ValueError("Wrong full-corpus backbone parameter count")
    for name, value in base.state_dict().items():
        if not torch.equal(value, saved["model"][name]) or not torch.isfinite(value).all():
            raise ValueError("Full-corpus pretrained tensor restoration failed")
    return base


def adapter_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_adapter_state(model, state):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state):
        raise ValueError("Adapter checkpoint does not match all and only trainable parameters")
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                raise ValueError("Invalid adapter/head checkpoint tensor")
            parameters[name].copy_(value)


def atomic_torch_save(path, payload):
    import os
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
