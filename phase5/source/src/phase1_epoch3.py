"""Epoch3 transition and best/latest retention, never touching older runs."""

import json
import os
from pathlib import Path
import stat
import time

import numpy as np
import torch

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase1_full_production import Coverage
from src.phase1_full_training import save_checkpoint


def start_epoch3(parent, optimizer, features, document_lengths, config):
    progress = parent["progress"]
    if (progress["epoch"] != 2 or progress["next_batch"] != config["parent_epoch_steps"]
            or progress["lifetime_optimizer_steps"] != config["prior_optimizer_steps"]
            or progress["lifetime_processed_targets"] != config["prior_targets"]
            or progress["committed_targets"] != config["total_targets"]
            or progress.get("pilot_only") is not False):
        raise ValueError("Epoch3 must start from the complete epoch2 terminal checkpoint")
    previous = Coverage(features, document_lengths)
    previous.restore(parent["extra_state"]["coverage"])
    previous.verify_complete(config["expected_training_counts"])
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer.state) != len(parameters):
        raise ValueError("Full pretrained AdamW state must be restored, not reset")
    for parameter in parameters:
        fields = optimizer.state[parameter]
        if int(fields["step"]) != config["prior_optimizer_steps"]:
            raise ValueError("Lifetime optimizer step counters changed")
        for name in ("exp_avg", "exp_avg_sq"):
            value = fields[name]
            if value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Pretrained AdamW moments are invalid")
    for group in optimizer.param_groups:
        if (group["lr"] != config["learning_rate"] or group["betas"] != (.9, .95)
                or group["weight_decay"] != .1):
            raise ValueError("Continuation must preserve the declared AdamW settings")
    return Coverage(features, document_lengths)


def epoch3_progress(coverage, next_batch, config):
    return {"epoch": 3, "next_batch": next_batch, "epoch_optimizer_steps": next_batch,
        "lifetime_optimizer_steps": config["prior_optimizer_steps"] + next_batch,
        "lifetime_processed_targets": config["prior_targets"] + coverage.targets,
        "unique_training_corpus_positions": config["total_targets"], "pilot_only": False,
        "additional_epoch_targets": coverage.targets, **coverage.summary()}


def verify_recovery(state, coverage, config):
    progress = state["progress"]
    if (progress["epoch"] != 3 or progress["lifetime_optimizer_steps"] != config["prior_optimizer_steps"] + progress["next_batch"]
            or progress["lifetime_processed_targets"] != config["prior_targets"] + coverage.targets
            or state["scheduler"] != {"kind": "constant", "lr": config["learning_rate"], "epoch": 3,
                                      "epoch_targets": coverage.targets}
            or len(state["rank_rng_states"]) != config["world_size"]):
        raise ValueError("Epoch3 recovery cursor, optimizer schedule or rank layout changed")


def verify_retained(directory):
    """Two durable names, verified against their external hashes and metadata."""
    directory = Path(directory)
    record = json.loads((directory / "retention.json").read_text())
    for name, entry in record["retained"].items():
        if name not in ("latest", "best"):
            raise ValueError("Unexpected retention name")
        path = directory / (name + ".pt")
        if path.is_symlink() or not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise ValueError("Retained checkpoint missing or hash changed")
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        if metadata["sha256"] != entry["sha256"] or metadata["progress"] != entry["progress"]:
            raise ValueError("Checkpoint sidecar and durable retention record disagree")
    return record


def retain_checkpoint(directory, model, optimizer, progress, bindings, rank_rng_states,
                      coverage, monitor, improved, best_overall):
    """Verify a new file before atomically replacing this run's latest/best.

    A hardlink lets best and latest share a complete checkpoint when identical.
    The step-named candidate exists only during the transaction. Old epoch1/2
    paths can never be targets of this helper. Interrupted transactions are
    left for explicit reconciliation rather than guessed away on a retry.
    """
    directory = Path(directory).resolve(strict=True)
    if directory.name != "checkpoints" or not (directory.parent / "epoch3_config.json").is_file():
        raise ValueError("Retention is restricted to an explicitly configured epoch3 directory")
    metadata_paths = list(directory.glob("step_*"))
    if metadata_paths:
        raise ValueError("An interrupted retention transaction needs explicit reconciliation")
    if (directory / "retention.json").exists():
        verify_retained(directory)
    else:
        if any(directory.iterdir()):
            raise ValueError("Do not replace unindexed checkpoint files")
    for name in ("latest.pt", "best.pt"):
        path = directory / name
        if path.exists() and (path.is_symlink() or not stat.S_ISREG(path.stat().st_mode)):
            raise ValueError("Only this run's regular checkpoint files may be replaced")
    candidate = directory / f"step_{progress['next_batch']:06d}_candidate.pt"
    if len(rank_rng_states) != bindings["world_size"]:
        raise ValueError("Every logical rank must contribute its actual RNG state")
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError("Nonfinite model parameter cannot become a retained checkpoint")
    for fields in optimizer.state.values():
        if any(isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()) for value in fields.values()):
            raise ValueError("Nonfinite AdamW state cannot become a retained checkpoint")
    scheduler = {"kind": "constant", "lr": 3e-5, "epoch": 3,
                 "epoch_targets": coverage.targets}
    metadata = save_checkpoint(candidate, model, optimizer, progress, bindings, rank_rng_states,
        scheduler_state=scheduler, scaler_state={},
        extra_state={"coverage": coverage.state(), "gradient_monitor": monitor,
                     "best_overall": best_overall})
    # Readback all serialized model/optimizer tensors before promoting it.
    from src.phase1_epoch2 import verify_saved_payload
    saved = torch.load(candidate, map_location="cpu", weights_only=False, mmap=True)
    verify_saved_payload(saved, model, optimizer, coverage, progress, bindings)
    if len(saved["rank_rng_states"]) != bindings["world_size"] or saved["scheduler"] != scheduler:
        raise ValueError("New checkpoint RNG/scheduler payload failed readback")
    del saved
    previous = (json.loads((directory / "retention.json").read_text())["retained"]
                if (directory / "retention.json").exists() else {})
    updates = ["latest"] + (["best"] if improved else [])
    for name in updates:
        destination = directory / (name + ".pt")
        link = directory / ("." + name + f".{os.getpid()}.link")
        if link.exists():
            raise ValueError("Interrupted atomic link must be reconciled")
        os.link(candidate, link)
        os.replace(link, destination)
        entry = {**metadata, "path": str(destination)}
        atomic_json(destination.with_suffix(".metadata.json"), entry)
        previous[name] = {"path": str(destination), "sha256": metadata["sha256"],
                          "bytes": metadata["bytes"], "progress": progress}
    record = {"retained": previous, "epoch": 3, "policy": "best and latest/final only",
        "candidate_verified_all_model_optimizer_tensors": True,
        "updated_at_unix": time.time(), "historical_epoch1_epoch2_modified": False}
    atomic_json(directory / "retention.json", record)
    # Exact newly created transaction files, not historical checkpoints/globs.
    candidate.unlink()
    candidate.with_suffix(".metadata.json").unlink()
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    verify_retained(directory)
    return record
