"""Exact raw-character boundary candidates; no tokenizer changes or offset caps."""

import math

import torch
from torch import nn


def candidate_positions(text, offset):
    """All internal character boundaries plus immediately adjacent whitespace.

    Start must precede the token end; end must follow its start. Candidates are
    generated from raw text alone, never gold labels, validation statistics, or
    fixed signed-displacement classes. Synthetic zero-width pieces have none.
    """
    a, b = offset
    if not 0 <= a <= b <= len(text):
        raise ValueError("invalid raw character offset")
    if a == b:
        return {"start": [], "end": []}
    left, right = a, b
    while left and text[left-1].isspace():
        left -= 1
    while right < len(text) and text[right].isspace():
        right += 1
    return {"start": list(range(left, b)), "end": list(range(a+1, right+1))}


def _character_bytes(character):
    # UTF-8 has at most four bytes per Unicode character. 256 is padding/BOS/EOS,
    # not an unknown character. Every actual byte remains representable.
    values = list(character.encode("utf-8")) if character else []
    return values + [256] * (4-len(values))


def candidate_features(text, offset, positions):
    """Only local bytes and relative positions; no absolute document indices."""
    a, b = offset
    width = max(1, b-a)
    left, right = a, b
    while left and text[left-1].isspace():
        left -= 1
    while right < len(text) and text[right].isspace():
        right += 1
    byte_rows, numeric_rows = [], []
    for position in positions:
        # Do not read another token's non-whitespace characters at region edges.
        previous = text[position-1] if position > left else ""
        following = text[position] if position < right else ""
        byte_rows.append(_character_bytes(previous) + _character_bytes(following))
        delta_left, delta_right = position-a, b-position
        signed_log = lambda value: math.copysign(math.log1p(abs(value)), value)
        numeric_rows.append([delta_left/width, delta_right/width, signed_log(delta_left), signed_log(delta_right),
                             float(position == a), float(position == b),
                             float(previous.isspace()), float(following.isspace())])
    return byte_rows, numeric_rows


def make_candidate_table(text, offsets):
    """Ragged flattened candidates: no padding to a maximum whitespace gap."""
    tokens, positions, byte_rows, numeric_rows, ranges = [], [], [], [], []
    for token, offset in enumerate(offsets):
        candidates = candidate_positions(text, offset)
        union = sorted(set(candidates["start"]) | set(candidates["end"]))
        start_set, end_set = set(candidates["start"]), set(candidates["end"])
        begin = len(positions)
        tokens.extend([token]*len(union))
        positions.extend(union)
        byte, numeric = candidate_features(text, offset, union)
        byte_rows.extend(byte)
        numeric_rows.extend(numeric)
        ranges.append({"begin": begin, "end": len(positions),
                       "start_indices": [begin+i for i, p in enumerate(union) if p in start_set],
                       "end_indices": [begin+i for i, p in enumerate(union) if p in end_set]})
    return {"tokens": tokens, "positions": positions, "bytes": byte_rows, "numeric": numeric_rows, "ranges": ranges}


class BoundaryScorer(nn.Module):
    """Score every legal character candidate for each label and endpoint role.

    The label dimension is an output dimension, not a supplied gold-label input.
    All embeddings are newly initialized; no external weights or language task.
    """
    def __init__(self, width, labels=12, hidden=64):
        super().__init__()
        self.labels = labels
        self.byte_embedding = nn.Embedding(257, 8, padding_idx=256)
        self.token_projection = nn.Linear(width, hidden)
        self.candidate_projection = nn.Linear(8*8+8, hidden)
        self.score = nn.Linear(hidden, labels*2)

    def forward(self, states, table):
        device = states.device
        tokens = torch.as_tensor(table["tokens"], device=device, dtype=torch.long)
        byte_ids = torch.as_tensor(table["bytes"], device=device, dtype=torch.long).reshape(-1, 8)
        numeric = torch.as_tensor(table["numeric"], device=device, dtype=states.dtype).reshape(-1, 8)
        local = torch.cat((self.byte_embedding(byte_ids).flatten(1), numeric), dim=-1)
        hidden = self.token_projection(states)[tokens] + self.candidate_projection(local)
        return self.score(torch.tanh(hidden)).reshape(-1, self.labels, 2)


def best_boundary_pair(start_positions, start_scores, end_positions, end_scores):
    """Maximize endpoint scores jointly with start < end; never round or clip."""
    starts = torch.as_tensor(start_positions, device=start_scores.device)
    ends = torch.as_tensor(end_positions, device=end_scores.device)
    scores = start_scores.float()[:, None] + end_scores.float()[None, :]
    legal = starts[:, None] < ends[None, :]
    if not legal.any() or not torch.isfinite(scores[legal]).all():
        raise ValueError("no finite legal character-boundary pair")
    scores = scores.masked_fill(~legal, float("-inf"))
    index = int(scores.argmax())
    return int(starts[index // len(ends)]), int(ends[index % len(ends)])


class BoundaryNER(nn.Module):
    """Unchanged BIO backbone plus an exact-character candidate-scoring head."""
    def __init__(self, bio, labels=12):
        super().__init__()
        self.bio = bio
        self.cfg = bio.cfg
        self.boundary = BoundaryScorer(bio.cfg.d_model, labels)

    def forward(self, ids, table):
        if ids.size(0) != 1:
            raise ValueError("ragged boundary head expects one document window per microbatch")
        states = self.bio.forward_features(ids)
        return {"bio": self.bio.classifier(states), "boundaries": self.boundary(states[0], table)}


def boundary_targets(document, window_number, bio_supervision, labels):
    """Gold targets only; never pass these records into the network forward."""
    window = document["windows"][window_number]
    left, right = window["token_start"], window["token_end"]
    result = []
    for entity in document["entities"]:
        if not left <= entity["token_start"] < entity["token_end"] <= right:
            continue
        real = [t for t in range(entity["token_start"], entity["token_end"])
                if document["token_offsets"][t][0] < document["token_offsets"][t][1]]
        first, last = real[0], real[-1]
        if bio_supervision[first-left] == -100:
            continue
        if bio_supervision[last-left] == -100:
            raise ValueError("entity endpoints have different training owners")
        result.extend([{"token": first-left, "kind": 0, "label_index": labels.index(entity["label"]), "position": entity["start"]},
                       {"token": last-left, "kind": 1, "label_index": labels.index(entity["label"]), "position": entity["end"]}])
    return result


def boundary_loss_sum(scores, table, targets):
    """Ragged candidate CE, summed over exact gold endpoints and no other tokens."""
    if not targets:
        return scores.sum() * 0, 0
    options, correct = [], []
    for target in targets:
        role = "start_indices" if target["kind"] == 0 else "end_indices"
        choices = table["ranges"][target["token"]][role]
        positions = [table["positions"][i] for i in choices]
        if target["position"] not in positions:
            raise ValueError("gold endpoint not representable; never round/drop it")
        options.append(choices)
        correct.append(positions.index(target["position"]))
    width = max(map(len, options))
    indices = [row + [row[0]]*(width-len(row)) for row in options]
    mask = [[True]*len(row) + [False]*(width-len(row)) for row in options]
    device = scores.device
    candidates = torch.tensor(indices, device=device)
    labels = torch.tensor([t["label_index"] for t in targets], device=device)
    roles = torch.tensor([t["kind"] for t in targets], device=device)
    logits = scores[candidates, labels[:, None], roles[:, None]].float()
    logits = logits.masked_fill(~torch.tensor(mask, device=device), float("-inf"))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(correct, device=device), reduction="sum")
    return loss, len(targets)


def token_entities(path, offsets, tags):
    """Return first/last real token indices, rather than rounded character spans."""
    entities, active = [], None
    for index, tag_id in enumerate(path):
        if offsets[index][0] == offsets[index][1]:
            continue
        tag = tags[tag_id]
        if tag == "O":
            if active is not None:
                entities.append(active)
                active = None
            continue
        prefix, label = tag.split("-", 1)
        if active is not None and prefix == "I" and active[2] == label:
            active[1] = index
        else:
            if active is not None:
                entities.append(active)
            active = [index, index, label]
    if active is not None:
        entities.append(active)
    return entities
