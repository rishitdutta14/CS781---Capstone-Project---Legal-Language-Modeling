"""Group genuinely matching protocols and report seed variation honestly."""

import hashlib
import json
import statistics


def protocol(config, decoding):
    keys = ("architecture", "phase1_checkpoint_sha256", "tokenizer_sha256", "data_manifest_sha256",
            "output_head", "loss", "focal_gamma", "boundary_weight", "learning_rate", "schedule",
            "warmup_steps", "weight_decay", "dropout_override", "maximum_epochs", "patience",
            "effective_batch_windows", "context_limit", "overlap", "precision")
    value = {key: config.get(key) for key in keys}
    value["adaptation"] = config.get("adaptation", "full_fine_tuning")
    settings = config.get("adaptation_details") or {}
    value["adaptation_settings"] = {key: settings.get(key) for key in
                                    ("rank", "alpha", "variant", "adapter_dropout", "partial_layers_per_stack", "target_modules")}
    kd = config.get("knowledge_distillation") or {}
    value["distillation_settings"] = {key: kd.get(key) for key in
        ("cache_manifest_sha256", "targets_sha256", "temperature", "type_labels", "alignment")}
    value["kd_weight"] = config.get("kd_weight", 0.)
    value["checkpoint_selection_decoding"] = config.get("decoding", "argmax")
    value["submission_decoding"] = decoding
    if value["maximum_epochs"] is None:
        value["maximum_epochs"] = config.get("epochs")
    if value["schedule"] is None:
        value["schedule"] = config.get("lr_schedule")
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return digest, value


def seed_statistics(rows):
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("duplicate seeds in one protocol; do not cherry-pick a run per seed")
    scores = [row["strict_macro_f1"] for row in rows]
    return {"runs": rows, "number_of_seeds": len(rows),
            "mean": statistics.mean(scores), "median": statistics.median(scores),
            "minimum": min(scores), "maximum": max(scores),
            "sample_std": statistics.stdev(scores) if len(scores) > 1 else None,
            "stability_established": False,
            "interpretation": "Descriptive variation on the same fixed 10-document validation split. A few seeds do not prove generalization or statistical superiority; no best-seed stability claim."}
