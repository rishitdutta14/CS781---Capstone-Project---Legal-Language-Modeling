"""Explicit epoch transition and accounting; architecture/data stay frozen."""

import numpy as np
import torch

from src.phase1_full_production import Coverage


def start_new_epoch(parent, optimizer, features, document_lengths, prior_steps, prior_targets, lr):
    progress = parent["progress"]
    if (progress["next_batch"] != prior_steps or progress["committed_targets"] != prior_targets
            or progress.get("pilot_only") is not False):
        raise ValueError("Starting checkpoint is not the completed production epoch")
    coverage = parent["extra_state"]["coverage"]
    if (not np.all(coverage["seen"] == 1)
            or not np.array_equal(coverage["document_targets"], document_lengths)
            or not np.all(coverage["document_eos"] == 1)):
        raise ValueError("Starting checkpoint does not have complete epoch1 coverage")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer.state) != len(parameters):
        raise ValueError("Optimizer state is missing trained parameters")
    for parameter in parameters:
        fields = optimizer.state[parameter]
        if int(fields["step"]) != prior_steps:
            raise ValueError("Do not reset or alter lifetime AdamW step counters")
        for name in ("exp_avg", "exp_avg_sq"):
            if fields[name].shape != parameter.shape or not bool(torch.isfinite(fields[name]).all()):
                raise ValueError("Invalid restored AdamW moments")
    for group in optimizer.param_groups:
        group["lr"] = lr
    # Only the epoch ledger resets. Model/optimizer/RNG are not reinitialized.
    return Coverage(features, document_lengths)


def epoch_progress(coverage, next_batch, config):
    return {"epoch": 2, "next_batch": next_batch,
            "epoch_optimizer_steps": next_batch,
            "lifetime_optimizer_steps": config["prior_optimizer_steps"] + next_batch,
            "lifetime_processed_targets": config["prior_targets"] + coverage.targets,
            "unique_training_corpus_positions": config["total_targets"],
            "additional_epoch_targets": coverage.targets,
            "pilot_only": False, **coverage.summary()}


def verify_constant_resume(state, coverage, config):
    scheduler = state["scheduler"]
    if (scheduler != {"kind": "constant", "lr": config["learning_rate"], "epoch": 2,
                      "epoch_targets": coverage.targets}
            or state["progress"]["epoch"] != 2
            or state["progress"]["lifetime_optimizer_steps"] != config["prior_optimizer_steps"] + state["progress"]["next_batch"]):
        raise ValueError("Epoch2 scheduler/lifetime cursor changed")


def verify_saved_payload(state, model, optimizer, coverage, progress, bindings):
    if state["progress"] != progress or state["bindings"] != bindings:
        raise ValueError("Checkpoint config/progress readback mismatch")
    for name, tensor in model.state_dict().items():
        if not torch.equal(state["model"][name], tensor.detach().cpu()):
            raise ValueError(f"Checkpoint model tensor differs: {name}")
    current = optimizer.state_dict()
    if state["optimizer"]["param_groups"] != current["param_groups"] or set(state["optimizer"]["state"]) != set(current["state"]):
        raise ValueError("Checkpoint optimizer groups/state membership differs")
    for identity, fields in current["state"].items():
        if set(fields) != set(state["optimizer"]["state"][identity]):
            raise ValueError("Checkpoint optimizer state fields differ")
        for name, tensor in fields.items():
            stored = state["optimizer"]["state"][identity][name]
            if isinstance(tensor, torch.Tensor):
                if not torch.equal(stored, tensor.detach().cpu()):
                    raise ValueError("Checkpoint optimizer tensor differs")
            elif stored != tensor:
                raise ValueError("Checkpoint optimizer scalar differs")
    for name in ("seen", "document_targets", "document_eos", "source_targets"):
        if not np.array_equal(state["extra_state"]["coverage"][name], getattr(coverage, name)):
            raise ValueError("Checkpoint coverage array differs")
