"""Training and decoding helpers for BIO plus learned raw-character boundaries."""

import json

import torch

from src.phase2_bio import preferred_window, strict_scores, supervision
from src.phase2_boundaries import (BoundaryNER, best_boundary_pair, boundary_targets,
                                  make_candidate_table, token_entities)
from src.phase2_decoding import legal_bio_path
from src.phase2_experiment import load_backbone, load_documents


def load_boundary_backbone(architecture, num_tags, tokenizer_hash, device="cuda"):
    bio, info = load_backbone(architecture, num_tags, tokenizer_hash, device="cpu")
    model = BoundaryNER(bio, (num_tags-1)//2)
    info["parameters"] = sum(p.numel() for p in model.parameters())
    if not 50_000_000 <= info["parameters"] <= 300_000_000:
        raise ValueError("boundary model parameter count outside required range")
    return model.to(device), info


def load_boundary_documents(directory):
    manifest, tags, docs, rows, histogram = load_documents(directory)
    by_key = {doc["document_id"]: doc for doc in docs["train"]}
    by_window = {}
    endpoints = 0
    for doc in docs["train"]:
        targets, _ = supervision(doc, tags)
        for number, target in enumerate(targets):
            by_window[(doc["document_id"], number)] = boundary_targets(doc, number, target, manifest["labels"])
    for row in rows:
        doc = by_key[row["document_id"]]
        window = doc["windows"][row["window_number"]]
        a, b = window["token_start"], window["token_end"]
        row["boundary_targets"] = by_window[(row["document_id"], row["window_number"])]
        row["candidate_table"] = make_candidate_table(doc["text"], doc["token_offsets"][a:b])
        endpoints += len(row["boundary_targets"])
    if endpoints != 2*sum(len(doc["entities"]) for doc in docs["train"]):
        raise ValueError("boundary training dropped or duplicated gold endpoints")
    return manifest, tags, docs, rows, histogram


@torch.no_grad()
def infer_boundary_document(model, doc, tags, decoding="legal_bio"):
    """Gold-independent windows, then token detection and exact endpoint choices."""
    length = len(doc["token_ids"])
    if length == 0:
        return [], []
    owners = [preferred_window(doc, t) for t in range(length)]
    combined = torch.empty(length, len(tags))
    candidates = [None]*length
    for number, window in enumerate(doc["windows"]):
        a, b = window["token_start"], window["token_end"]
        table = make_candidate_table(doc["text"], doc["token_offsets"][a:b])
        ids = torch.tensor([doc["token_ids"][a:b]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(ids, table)
        logits, scores = outputs["bio"][0].float().cpu(), outputs["boundaries"].float().cpu()
        for local in range(b-a):
            token = a+local
            if owners[token] == number:
                combined[token] = logits[local]
                span = table["ranges"][local]
                candidates[token] = {"start_positions": [table["positions"][i] for i in span["start_indices"]],
                                     "end_positions": [table["positions"][i] for i in span["end_indices"]],
                                     "start_scores": scores[span["start_indices"], :, 0],
                                     "end_scores": scores[span["end_indices"], :, 1]}
    if any(row is None for row in candidates):
        raise ValueError("boundary inference dropped token positions")
    if decoding == "legal_bio":
        path = legal_bio_path(combined, doc["token_offsets"], tags)
    elif decoding == "argmax":
        path = combined.argmax(-1).tolist()
    else:
        raise ValueError("unknown BIO decoder")
    labels = [tag[2:] for tag in tags if tag.startswith("B-")]
    refined, projected = [], []
    for first, last, label in token_entities(path, doc["token_offsets"], tags):
        index = labels.index(label)
        start, end = candidates[first], candidates[last]
        left, right = best_boundary_pair(start["start_positions"], start["start_scores"][:, index],
                                         end["end_positions"], end["end_scores"][:, index])
        refined.append([left, right, label])
        projected.append([doc["token_offsets"][first][0], doc["token_offsets"][last][1], label])
    return ([list(span) for span in sorted(set(map(tuple, refined)))],
            [list(span) for span in sorted(set(map(tuple, projected)))])


def evaluate_boundary_model(model, documents, tags, labels, decoding):
    model.eval()
    predictions, projected, truth = {}, {}, {}
    for doc in documents:
        key = doc["document_id"]
        predictions[key], projected[key] = infer_boundary_document(model, doc, tags, decoding)
        truth[key] = [[e["start"], e["end"], e["label"]] for e in doc["entities"]]
    scores = strict_scores(truth, predictions, labels)
    scores["same_checkpoint_whole_token_scores_NO_boundary_refinement"] = strict_scores(truth, projected, labels)
    return scores, predictions, truth
