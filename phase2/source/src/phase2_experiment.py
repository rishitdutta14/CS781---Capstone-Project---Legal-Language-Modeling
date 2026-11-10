"""Shared checks for the six-backbone Phase 2 repair campaign."""

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from assignment3_impl.q2.model_arch import DecoderLM, LMConfig
from src.phase1_v2_models import V2Config, build_model
from src.phase2_bio import DenseBIO, EncoderDecoderBIO, bio_tags, supervision
from src.phase2_decoding import decode_logits
from src.phase2_bio import preferred_window, strict_scores
from src.phase2_objectives import training_class_weights

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = ("mha", "mqa", "gqa", "mla", "dense", "encoder_decoder")
TOKENIZER_HASHES = {
    "original": "c723db3edc9393ba40cb9ba95ee5c0ee46c57659cfb92f84c29c2bd2229431eb",
    "rebuilt": "d8e16fc67292677f03e679f4c30c3cb66ca4735a558edcba19bd13e016b44ca9",
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize_fresh_config(path, resume=False):
    """One authoritative existence check, before rank 0 may create config.json.

    Checking independently on each rank races against rank 0's write. Broadcast
    the initial rank-0 observation instead; all ranks accept or reject together.
    """
    device = "cuda" if dist.get_backend() == "nccl" else "cpu"
    exists = torch.tensor(int(path.exists()) if dist.get_rank() == 0 else 0, device=device)
    dist.broadcast(exists, src=0)
    if exists.item() and not resume:
        raise ValueError("existing experiment requires explicit resume, not silent fresh restart")


def checkpoint_path(architecture):
    if architecture not in BACKBONES:
        raise ValueError("backbone is outside the six-model campaign")
    if architecture in ("dense", "encoder_decoder"):
        return ROOT / "outputs/phase1_v2" / architecture / "best_model.pt"
    return ROOT / "outputs/phase1" / architecture / "best_model/model.pt"


def load_backbone(architecture, num_tags, tokenizer_hash, device="cuda"):
    family = "rebuilt" if architecture in ("dense", "encoder_decoder") else "original"
    if tokenizer_hash != TOKENIZER_HASHES[family]:
        raise ValueError("tokenizer IDs do not match the Phase 1 checkpoint")
    path = checkpoint_path(architecture)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if family == "original":
        if saved["config"]["model_name"] != architecture:
            raise ValueError("wrong original checkpoint architecture")
        cfg = LMConfig(**saved["config"]["model"])
        backbone = DecoderLM(cfg)
        provenance = "Original Phase 1: 200M processed tokens from reused 100M shard; tokenizer fitted before split."
    else:
        if saved["architecture"] != architecture:
            raise ValueError("wrong rebuilt checkpoint architecture")
        cfg = V2Config(**saved["model_config"])
        backbone = build_model(cfg)
        provenance = "Rebuilt Phase 1: selected corpus pool, 200M distinct prediction targets; not the entire raw corpus."
    if cfg.max_seq_len != 1600 or cfg.vocab_size != 16000:
        raise ValueError("checkpoint context/vocabulary unexpectedly changed")
    backbone.load_state_dict(saved["model"], strict=True)
    for name, tensor in backbone.state_dict().items():
        if not torch.equal(tensor.cpu(), saved["model"][name]):
            raise ValueError(f"Phase 1 weight loading changed tensor {name}")
    adapter = EncoderDecoderBIO if architecture == "encoder_decoder" else DenseBIO
    model = adapter(backbone, num_tags, checkpoint_blocks=True)
    del backbone, saved
    parameters = sum(p.numel() for p in model.parameters())
    if not 50_000_000 <= parameters <= 300_000_000:
        raise ValueError("active NER parameter count outside permitted range")
    return model.to(device), {"model_config": vars(cfg), "parameters": parameters,
                             "phase1_checkpoint": str(path), "phase1_checkpoint_sha256": file_sha(path),
                             "tokenizer_family": family, "tokenizer_sha256": tokenizer_hash,
                             "phase1_provenance_limitation": provenance,
                             "backbone_loaded_exactly": True}


def load_documents(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    preflight = json.loads((directory / "preflight.json").read_text())
    if preflight["status"] != "passed" or not preflight["gold_window_roundtrip_exact"]:
        raise ValueError("data preflight did not pass")
    if manifest["config"]["context_limit"] != 1600 or manifest["config"]["added_special_tokens"] != []:
        raise ValueError("context/special-token policy changed")
    frozen = json.loads((ROOT / "data/processed/phase2_preflight_20261011_r2/manifest.json").read_text())
    if manifest["document_ids"] != frozen["document_ids"]:
        raise ValueError("document split changed")
    for path, digest in manifest["inputs_sha256"].items():
        if file_sha(path) != digest:
            raise ValueError(f"source hash changed: {path}")
    tags = bio_tags(manifest["labels"])
    documents, rows = {}, []
    histogram = torch.zeros(len(tags), dtype=torch.int64)
    for split in ("train", "validation"):
        documents[split] = []
        for key in manifest["document_ids"][split]:
            doc = json.loads((directory / "documents" / split / f"{key}.json").read_text())
            if doc["document_id"] != key or doc["split"] != split:
                raise ValueError("prepared document identity changed")
            targets, _ = supervision(doc, tags)
            documents[split].append(doc)
            if split == "train":
                for window, target in zip(doc["windows"], targets):
                    a, b = window["token_start"], window["token_end"]
                    real = torch.tensor([t for t in target if t != -100])
                    if not len(real):
                        raise ValueError("training window has no supervised real tokens")
                    histogram += torch.bincount(real, minlength=len(tags))
                    rows.append({"document_id": key, "window_number": window["window_number"],
                                 "ids": doc["token_ids"][a:b], "targets": target, "count": len(real)})
    return manifest, tags, documents, rows, histogram


def set_loss_normalizers(rows, histogram, kind):
    weights = training_class_weights(histogram) if kind == "weighted_ce" else None
    for row in rows:
        row["normalizer"] = (float(weights[torch.tensor([t for t in row["targets"] if t != -100])].sum())
                             if weights is not None else row["count"])
    return weights


@torch.no_grad()
def document_logits(model, doc):
    owners = [preferred_window(doc, t) for t in range(len(doc["token_ids"]))]
    combined, assigned = None, [False] * len(owners)
    for number, window in enumerate(doc["windows"]):
        a, b = window["token_start"], window["token_end"]
        ids = torch.tensor([doc["token_ids"][a:b]], device="cuda", dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)[0].float().cpu()
        if combined is None:
            combined = torch.empty(len(owners), logits.size(-1))
        indices = [i for i in range(a, b) if owners[i] == number]
        combined[indices] = logits[[i-a for i in indices]]
        for i in indices:
            assigned[i] = True
    if not all(assigned):
        raise ValueError("inference dropped token positions")
    return combined


def evaluate_model(model, documents, tags, labels, decoding):
    model.eval()
    predictions, truth = {}, {}
    for doc in documents:
        logits = document_logits(model, doc)
        predictions[doc["document_id"]] = [] if logits is None else decode_logits(
            logits, doc["token_offsets"], tags, len(doc["text"]), decoding)
        truth[doc["document_id"]] = [[e["start"], e["end"], e["label"]] for e in doc["entities"]]
    return strict_scores(truth, predictions, labels), predictions, truth
