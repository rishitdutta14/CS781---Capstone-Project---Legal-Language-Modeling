"""Gold-independent BIO sequence decoding, without changing backbone attention."""

import numpy as np
import torch

from src.phase2_bio import decode_tags


def legal_bio_path(logits, offsets, tags):
    """Highest-scoring legal BIO path over real tokens in one complete document.

    Transition scores are zero for legal transitions and minus infinity otherwise.
    This is constrained Viterbi, not a trained CRF. Synthetic zero-width tokens
    contribute neither emissions nor transitions. No entity-specific rules.
    """
    if logits.shape != (len(offsets), len(tags)):
        raise ValueError("logit/offset/tag dimensions differ")
    real = [i for i, (a, b) in enumerate(offsets) if a < b]
    result = [0] * len(offsets)
    if not real:
        return result
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().float().cpu().numpy()
    emissions = np.asarray(logits, dtype=np.float64)[real]
    if not np.isfinite(emissions).all():
        raise ValueError("nonfinite emissions")
    transitions = np.zeros((len(tags), len(tags)), dtype=np.float64)
    starts = np.zeros(len(tags), dtype=np.float64)
    for current, tag in enumerate(tags):
        if tag.startswith("I-"):
            starts[current] = -np.inf
            for previous, prev in enumerate(tags):
                if prev not in ("B-" + tag[2:], tag):
                    transitions[previous, current] = -np.inf
    scores = emissions[0] + starts
    pointers = np.zeros((len(real), len(tags)), dtype=np.int16)
    for position in range(1, len(real)):
        candidates = scores[:, None] + transitions
        pointers[position] = candidates.argmax(axis=0)
        scores = candidates.max(axis=0) + emissions[position]
        # Avoid accumulating huge document-level magnitudes; path unchanged.
        scores -= scores.max()
    choice = int(scores.argmax())
    for position in range(len(real)-1, -1, -1):
        result[real[position]] = choice
        if position:
            choice = int(pointers[position, choice])
    return result


def decode_logits(logits, offsets, tags, text_length, method="argmax"):
    if method == "argmax":
        choices = logits.argmax(-1).tolist()
    elif method == "legal_bio":
        choices = legal_bio_path(logits, offsets, tags)
    else:
        raise ValueError(f"unknown decoding method: {method}")
    return decode_tags(choices, offsets, tags, text_length)
