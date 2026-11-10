"""Independent epoch-2 log reconciliation; never performs optimizer updates."""

import math

import numpy as np

from src.phase1_full_production import batch_workload


def sampler_execution_order(order, blocks):
    # Blocks themselves are shuffled; a consumed prefix is NOT order[:count].
    return np.concatenate([order[int(offset):int(offset+count)] for offset, count in blocks])


def reconcile_epoch2_metrics(rows, features, config):
    order, blocks, lengths = (features[k] for k in ("order", "blocks", "lengths"))
    if len(rows) != len(blocks):
        raise ValueError("Log does not cover the complete epoch2 batch schedule")
    seen = np.zeros(len(lengths), dtype=np.uint8)
    targets, samples, seconds, weighted_loss = 0, 0, 0.0, 0.0
    work = None
    clipped, largest = 0, 0.0
    checkpoints = []
    threshold = config["checkpoint_every_targets"]
    for step, (row, block) in enumerate(zip(rows, blocks), 1):
        offset, count = map(int, block)
        ids = order[offset:offset + count]
        if (count < 1 or offset < 0 or len(ids) != count or len(np.unique(ids)) != count
                or np.any(ids < 0) or np.any(ids >= len(seen)) or np.any(seen[ids])):
            raise ValueError("Epoch2 batch schedule repeats, drops or corrupts examples")
        seen[ids] = 1
        valid = int(lengths[ids].sum())
        targets += valid
        samples += count
        if (row["epoch"] != 2 or row["optimizer_step"] != step
                or row["lifetime_optimizer_step"] != config["prior_optimizer_steps"] + step
                or row["samples"] != count or row["valid_targets"] != valid
                or row["committed_targets"] != targets
                or row["lifetime_processed_targets"] != config["prior_targets"] + targets):
            raise ValueError("Epoch/lifetime cursor or actual target accounting differs")
        if row["learning_rate"] != config["learning_rate"]:
            raise ValueError("Constant learning rate changed")
        for key in ("loss", "gradient_norm", "synchronized_local_seconds",
                    "local_data_preparation_and_transfer_seconds"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Nonfinite or negative training evidence")
        if row["clipped"] is not (row["gradient_norm"] > config["gradient_clipping"]):
            raise ValueError("Gradient clipping flag disagrees with preclip norm")
        expected_work = batch_workload(features, ids, config["world_size"], config["microbatch"])
        if row["input_workload"] != expected_work:
            raise ValueError("Logged input/padding/dummy workload differs from batch layout")
        if work is None:
            work = dict.fromkeys(expected_work, 0)
        for key, value in expected_work.items():
            work[key] += value
        clipped += int(row["clipped"])
        largest = max(largest, row["gradient_norm"])
        seconds += row["synchronized_local_seconds"]
        weighted_loss += row["loss"] * valid
        if targets >= threshold or step == len(blocks):
            checkpoints.append({"step": step, "targets": targets, "samples": samples,
                "gradient_monitor": {"steps": step, "clipped_steps": clipped,
                                     "maximum_preclip_norm": largest},
                "input_workload": work.copy()})
            threshold = (targets // config["checkpoint_every_targets"] + 1) * config["checkpoint_every_targets"]
    if not np.all(seen == 1) or targets != config["total_targets"]:
        raise ValueError("Epoch2 is not a complete unique pass")
    norms = np.array([row["gradient_norm"] for row in rows])
    return {"passed": True, "optimizer_steps": len(rows), "samples": samples,
            "valid_targets": targets, "lifetime_optimizer_steps": config["prior_optimizer_steps"] + len(rows),
            "lifetime_processed_targets": config["prior_targets"] + targets,
            "unique_training_corpus_positions": config["total_targets"],
            "step_seconds_sum": seconds, "compute_targets_per_second": targets / seconds,
            "target_weighted_training_loss_over_changing_weights": weighted_loss / targets,
            "gradient_monitor": {"steps": len(rows), "clipped_steps": clipped, "maximum_preclip_norm": largest},
            "clipping_fraction": clipped / len(rows),
            "preclip_norm_quantiles": {str(q): float(np.quantile(norms, q)) for q in (0, .5, .9, .99, 1)},
            "checkpoint_prefixes": checkpoints, "input_workload": work}
