"""Independent reconciliation of completed full-pass evidence, not training."""

import math

import numpy as np

from src.phase1_full_training import learning_rate


def reconcile_metrics(rows, order, blocks, lengths, config):
    if len(rows) != len(blocks):
        raise ValueError("Optimizer log does not cover exactly the full batch schedule")
    visited = np.zeros(len(order), dtype=np.uint8)
    targets, samples, seconds, weighted_loss = 0, 0, 0.0, 0.0
    workload = None
    checkpoint_steps = []
    next_checkpoint = config["checkpoint_every_targets"]
    for step, (row, block) in enumerate(zip(rows, blocks), 1):
        offset, count = map(int, block)
        ids = order[offset:offset + count]
        if len(ids) != count or len(np.unique(ids)) != count or np.any(visited[ids]):
            raise ValueError("Batch schedule repeats or loses an example")
        visited[ids] = 1
        expected = int(lengths[ids].sum())
        targets += expected
        samples += count
        if (row["optimizer_step"] != step or row["valid_targets"] != expected
                or row["samples"] != count or row["committed_targets"] != targets):
            raise ValueError("Logged optimizer targets/cursor disagree with original batch plan")
        lr = learning_rate(targets, config["total_targets"], config["peak_lr"],
                           config["floor_lr"], config["warmup_fraction"])
        if not math.isclose(row["learning_rate"], lr, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Learning-rate schedule does not match committed targets")
        for key in ("loss", "gradient_norm", "synchronized_local_seconds",
                    "local_data_preparation_and_transfer_seconds"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Nonfinite or negative training evidence")
        seconds += row["synchronized_local_seconds"]
        weighted_loss += row["loss"] * expected
        work = row["input_workload"]
        if workload is None:
            workload = dict.fromkeys(work, 0)
        for key, value in work.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError("Invalid logged input/padding count")
            workload[key] += value
        if targets >= next_checkpoint or step == len(blocks):
            checkpoint_steps.append(step)
            next_checkpoint = (targets // config["checkpoint_every_targets"] + 1) * config["checkpoint_every_targets"]
    if not np.all(visited == 1) or targets != config["total_targets"]:
        raise ValueError("Logged pass is not complete and unique")
    return {"passed": True, "optimizer_steps": len(rows), "samples": samples,
            "valid_targets": targets, "step_seconds_sum": seconds,
            "target_weighted_training_loss_over_changing_weights": weighted_loss / targets,
            "checkpoint_steps": checkpoint_steps, "input_workload": workload}


def reconcile_validation(result, plan):
    expected = plan["counts"]["val"]
    if (result["complete_validation"] is not True or result["examples"] != expected["samples"]
            or result["valid_targets"] != expected["distinct_supervised_targets"]):
        raise ValueError("Held-out evaluation is not complete")
    if not math.isfinite(result["cross_entropy"]) or result["cross_entropy"] < 0:
        raise ValueError("Invalid held-out loss")
    tokens, loss = 0, 0.0
    if set(result["by_source"]) != set(plan["source_counts"]):
        raise ValueError("Missing held-out source")
    for source, row in result["by_source"].items():
        count = plan["source_counts"][source]["val"]["distinct_supervised_targets"]
        if row["valid_targets"] != count or not math.isfinite(row["cross_entropy"]):
            raise ValueError("Held-out source targets/loss disagree")
        tokens += count
        loss += count * row["cross_entropy"]
    if tokens != result["valid_targets"] or not math.isclose(loss / tokens, result["cross_entropy"], rel_tol=1e-12):
        raise ValueError("Held-out weighted source losses do not reconcile")
    if not math.isclose(math.exp(result["cross_entropy"]), result["perplexity"], rel_tol=1e-12):
        raise ValueError("Held-out perplexity disagrees")
    return {"passed": True, "all_source_counts_and_weighted_losses_reconcile": True,
            "actual_numerical_validation_record": result,
            "not_an_independent_second_neural_forward_pass": True}


def reconcile_coverage(state, features, document_lengths, completed_samples):
    expected_seen = np.zeros(len(features["lengths"]), dtype=np.uint8)
    expected_seen[completed_samples] = 1
    if len(np.unique(completed_samples)) != len(completed_samples):
        raise ValueError("Checkpoint cursor repeats examples")
    if not np.array_equal(state["seen"], expected_seen):
        raise ValueError("Checkpoint seen bitmap does not match exact sampler cursor")
    lengths = features["lengths"][completed_samples]
    docs = features["document"][completed_samples]
    expected_doc = np.bincount(docs, weights=lengths, minlength=len(document_lengths)).astype(np.int64)
    eos = features["has_eos"][completed_samples]
    expected_eos = np.bincount(docs, weights=eos, minlength=len(document_lengths)).astype(np.uint8)
    source = np.bincount(features["source"][completed_samples], weights=lengths,
                         minlength=len(features["source_names"])).astype(np.int64)
    for key, value in (("document_targets", expected_doc), ("document_eos", expected_eos), ("source_targets", source)):
        if not np.array_equal(state[key], value):
            raise ValueError("Checkpoint per-document/EOS/source coverage differs from sampler prefix")
    summary = state["summary"]
    if summary["committed_targets"] != int(lengths.sum()) or summary["committed_samples"] != len(completed_samples):
        raise ValueError("Checkpoint coverage summary disagrees")
    return {"passed": True, "samples": len(completed_samples), "targets": int(lengths.sum()),
            "all_document_and_source_counters_match_exact_cursor": True}
