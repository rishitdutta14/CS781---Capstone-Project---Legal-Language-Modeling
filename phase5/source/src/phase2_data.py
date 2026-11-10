"""Document windowing and exact gold-span bookkeeping, without a model head."""

import bisect
import hashlib


def choose_split(folds, seed):
    """Hold out five documents per official labelled fold, before windowing."""
    train, validation = [], []
    for fold, ids in sorted(folds.items()):
        if len(ids) != 35:
            raise ValueError(f"expected 35 documents in {fold}")
        ranked = sorted(ids, key=lambda key: (
            hashlib.sha256(f"{seed}:{fold}:{key}".encode()).hexdigest(), key))
        validation.extend(ranked[:5])
        train.extend(ranked[5:])
    if len(set(train + validation)) != len(train + validation):
        raise ValueError("document IDs are not globally unique")
    return {"train": sorted(train), "validation": sorted(validation)}


def window_ranges(length, capacity=1600, overlap=128):
    if length < 0 or not 0 <= overlap < capacity:
        raise ValueError("invalid window settings")
    result, start = [], 0
    while start < length:
        end = min(length, start + capacity)
        result.append((start, end))
        if end == length:
            break
        start += capacity - overlap
    return result


def make_entity_records(text, offsets, spans, allowed_labels):
    """Record character displacements from boundary token starts; never round.

    These are gold targets for auditing, not a decision about how a model will
    predict them. Displacements use Unicode characters, not byte positions.
    They may extend into whitespace omitted by A1. Such cases are flagged;
    preserving gold coordinates does not imply a token head can predict them.
    """
    starts = [a for a, _ in offsets]
    ends = [b for _, b in offsets]
    if starts != sorted(starts) or ends != sorted(ends):
        raise ValueError("offsets must be monotonic")
    records, seen = [], set()
    for number, span in enumerate(spans):
        if len(span) not in (3, 4):
            raise ValueError("invalid gold entity structure")
        a, b, label = span[:3]
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
            raise ValueError("invalid original character span")
        if label not in allowed_labels or (a, b, label) in seen:
            raise ValueError("unknown label or duplicate entity")
        seen.add((a, b, label))
        if len(span) == 4 and text[a:b] != span[3]:
            raise ValueError("gold entity text does not match original text")
        left = bisect.bisect_right(ends, a)
        right = bisect.bisect_left(starts, b)
        if right <= left:
            raise ValueError("entity has no token coverage")
        outside_tokens = not (starts[left] <= a < ends[left] and starts[right-1] < b <= ends[right-1])
        if a < starts[left] and not text[a:starts[left]].isspace():
            raise ValueError("unmapped start characters are not whitespace")
        if b > ends[right-1] and not text[ends[right-1]:b].isspace():
            raise ValueError("unmapped end characters are not whitespace")
        records.append({"entity_number": number, "start": a, "end": b,
                        "label": label, "token_start": left, "token_end": right,
                        "start_character_delta": a - starts[left],
                        "end_character_delta": b - starts[right-1],
                        "boundary_in_omitted_whitespace": outside_tokens,
                        "nonexact_token_boundary": starts[left] != a or ends[right-1] != b,
                        "boundary_inside_word":
                            (a > 0 and not text[a-1].isspace() and not text[a].isspace()) or
                            (b < len(text) and not text[b-1].isspace() and not text[b].isspace())})
    return records


def restore_entity(record, offsets, text=None):
    a = offsets[record["token_start"]][0] + record["start_character_delta"]
    b = offsets[record["token_end"]-1][0] + record["end_character_delta"]
    if not 0 <= a < b or (text is not None and b > len(text)):
        raise ValueError("invalid restored character span")
    return [a, b, record["label"]]


def make_windows(offsets, entities, capacity=1600, overlap=128):
    """Keep complete and cut-off gold entities explicitly separate.

    Partial entities are not negative examples. A future training adapter must
    ignore their supervision; no BIO labels or loss masks are chosen here.
    """
    windows = []
    for number, (a, b) in enumerate(window_ranges(len(offsets), capacity, overlap)):
        complete, partial = [], []
        for e in entities:
            left, right = e["token_start"], e["token_end"]
            if a <= left and right <= b:
                complete.append(e["entity_number"])
            elif left < b and a < right:
                partial.append(e["entity_number"])
        windows.append({"window_number": number, "token_start": a, "token_end": b,
                        "character_start": offsets[a][0], "character_end": offsets[b-1][1],
                        "complete_entity_numbers": complete,
                        "partial_entity_numbers_ignore_supervision": partial})
    covered = {number for w in windows for number in w["complete_entity_numbers"]}
    if covered != {e["entity_number"] for e in entities}:
        raise ValueError("some entities are not fully contained in any window")
    if offsets and (windows[0]["token_start"] != 0 or windows[-1]["token_end"] != len(offsets)):
        raise ValueError("document tail was dropped")
    return windows


def shared_entity_tokens(entities):
    owners = {}
    for e in entities:
        for token in range(e["token_start"], e["token_end"]):
            owners.setdefault(token, []).append(e["entity_number"])
    return [{"token": token, "entity_numbers": numbers}
            for token, numbers in sorted(owners.items()) if len(numbers) > 1]
