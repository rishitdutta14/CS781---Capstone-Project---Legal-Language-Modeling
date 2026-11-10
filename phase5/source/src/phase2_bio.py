"""Plain BIO NER using unchanged Phase 1 backbone modules.

Whole-token decoding is intentional: no learned or rule-based boundary repair.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def bio_tags(labels):
    return ["O"] + [tag for label in labels for tag in (f"B-{label}", f"I-{label}")]


class DenseBIO(nn.Module):
    def __init__(self, backbone, num_tags, checkpoint_blocks=True):
        super().__init__()
        # Reuse the actual A3 modules, not a reimplementation of attention.
        self.cfg = backbone.cfg
        self.tok_emb = backbone.tok_emb
        self.blocks = backbone.blocks
        self.ln_f = backbone.ln_f
        self.register_buffer("pos_emb", backbone.pos_emb.detach().clone(), persistent=False)
        self.classifier = nn.Linear(self.cfg.d_model, num_tags)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, ids):
        return self.classifier(self.forward_features(ids))

    def forward_features(self, ids):
        if ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("input exceeds the existing backbone context")
        x = self.tok_emb(ids) + self.pos_emb[:ids.size(1)].unsqueeze(0)
        for block in self.blocks:
            if self.training and self.checkpoint_blocks:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.ln_f(x)


class EncoderDecoderBIO(nn.Module):
    """Both pretrained stacks active; decoder tags the same text as the encoder.

    Encoder self-attention stays bidirectional, decoder self-attention stays
    causal, and decoder cross-attention can use the complete encoder window.
    Inputs are text tokens only, never gold tags or gold boundary coordinates.
    """

    def __init__(self, backbone, num_tags, checkpoint_blocks=True):
        super().__init__()
        self.cfg = backbone.cfg
        self.tok_emb = backbone.tok_emb
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.encoder_norm = backbone.encoder_norm
        self.decoder_norm = backbone.decoder_norm
        self.register_buffer("pos_emb", backbone.pos_emb.detach().clone(), persistent=False)
        self.classifier = nn.Linear(self.cfg.d_model, num_tags)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, ids):
        return self.classifier(self.forward_features(ids))

    def forward_features(self, ids):
        if ids.size(1) > self.cfg.max_seq_len:
            raise ValueError("input exceeds the existing backbone context")
        positions = self.pos_emb[:ids.size(1)].unsqueeze(0)
        memory = self.tok_emb(ids) + positions
        for block in self.encoder:
            if self.training and self.checkpoint_blocks:
                memory = checkpoint(block, memory, use_reentrant=False)
            else:
                memory = block(memory)
        memory = self.encoder_norm(memory)
        x = self.tok_emb(ids) + positions
        for block in self.decoder:
            if self.training and self.checkpoint_blocks:
                x = checkpoint(block, x, memory, use_reentrant=False)
            else:
                x = block(x, memory)
        return self.decoder_norm(x)


def preferred_window(document, token, candidates=None):
    """Maximum minimum distance to either window edge; earlier window wins ties."""
    windows = document["windows"]
    if candidates is None:
        candidates = [i for i, w in enumerate(windows) if w["token_start"] <= token < w["token_end"]]
    if not candidates:
        raise ValueError("token has no containing window")
    return max(candidates, key=lambda i: (
        min(token-windows[i]["token_start"], windows[i]["token_end"]-1-token), -i))


def supervision(document, tags):
    """Each real token supervised once; every entity assigned a complete window.

    All tokens of an entity belong to the same containing window. Other windows
    retain it as context, with loss ignored. Partial entities are never negatives.
    Synthetic zero-width A1 space pieces are always ignored for loss/decoding.
    """
    tag_ids = {tag: i for i, tag in enumerate(tags)}
    offsets = document["token_offsets"]
    target = [0] * len(offsets)
    owners = [preferred_window(document, t) for t in range(len(offsets))]
    occupied = set()
    for entity in document["entities"]:
        left, right = entity["token_start"], entity["token_end"]
        if occupied.intersection(range(left, right)):
            raise ValueError("plain BIO cannot represent entities sharing tokens")
        occupied.update(range(left, right))
        candidates = [i for i, w in enumerate(document["windows"])
                      if left >= w["token_start"] and right <= w["token_end"]]
        owner = preferred_window(document, (left+right-1)//2, candidates)
        first = True
        for token in range(left, right):
            owners[token] = owner
            if offsets[token][0] == offsets[token][1]:
                continue
            prefix = "B" if first else "I"
            target[token] = tag_ids[f"{prefix}-{entity['label']}"]
            first = False
        if first:
            raise ValueError("entity has no real-character token")
    result = []
    for i, window in enumerate(document["windows"]):
        a, b = window["token_start"], window["token_end"]
        result.append([target[t] if owners[t] == i and offsets[t][0] < offsets[t][1] else -100
                       for t in range(a, b)])
    expected = sum(a < b for a, b in offsets)
    if sum(tag != -100 for row in result for tag in row) != expected:
        raise ValueError("supervision omitted or repeated real tokens")
    return result, target


def decode_tags(tag_ids, offsets, tags, text_length):
    """Document BIO decoding; orphan/different-type I tags start a new entity."""
    if len(tag_ids) != len(offsets):
        raise ValueError("tag/offset lengths differ")
    result, active = [], None

    def flush():
        nonlocal active
        if active is not None:
            if not 0 <= active[0] < active[1] <= text_length:
                raise ValueError("decoded span is outside the original document")
            result.append(active)
            active = None

    for index, (a, b) in zip(tag_ids, offsets):
        if a == b:
            continue
        tag = tags[index]
        if tag == "O":
            flush()
            continue
        prefix, label = tag.split("-", 1)
        if active is not None and prefix == "I" and active[2] == label:
            active[1] = max(active[1], b)
        else:
            flush()
            active = [a, b, label]
    flush()
    return [list(span) for span in sorted(set(tuple(span) for span in result))]


def strict_scores(gold, predictions, labels):
    if set(gold) != set(predictions):
        raise ValueError("gold/prediction IDs differ")
    per_label = {}
    for label in labels:
        tp = fp = fn = 0
        for key in gold:
            truth = {tuple(span[:3]) for span in gold[key] if span[2] == label}
            predicted = {tuple(span[:3]) for span in predictions[key] if span[2] == label}
            tp += len(truth & predicted)
            fp += len(predicted - truth)
            fn += len(truth - predicted)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return {"strict_macro_f1": sum(row["f1"] for row in per_label.values())/len(labels),
            "per_label": per_label}
