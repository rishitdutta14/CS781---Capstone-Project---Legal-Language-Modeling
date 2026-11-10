"""Two declared LoRA runs; generic objective/evaluator/recovery code is reused."""

import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from src.phase3_data import RRData, collate, epoch_batches, predictions_from_rows
from src.phase3_model import frozen_sha256, parameter_summary
from src.phase3_rebuilt_model import RebuiltRRClassifier, load_rebuilt_backbone
from src.phase3_submission import file_sha256, load_test_contract, strict_loads, validate_submission
from src.phase3_training import (ROOT, append_json, evaluate, loss_per_sentence, now,
    predict, restore_checkpoint, save_checkpoint, seed_all, verify_bindings, write_json)


def build_classifier(config, device):
    if config["method"] != "lora" or config["architecture"] not in ("dense", "encoder_decoder"):
        raise ValueError("This campaign authorizes exactly two rebuilt LoRA classifiers")
    base = load_rebuilt_backbone(config["checkpoint"], config["model_config"], config["checkpoint_sha256"])
    return RebuiltRRClassifier(base, config["architecture"], config["rank"], config["alpha"]).to(device)


def make_optimizer(model, config, steps_per_epoch):
    body = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("classifier.")]
    optimizer = torch.optim.AdamW([
        {"params": body, "lr": config["learning_rate"]},
        {"params": model.classifier.parameters(), "lr": config["head_learning_rate"]}],
        betas=(0.9, 0.999), eps=1e-8, weight_decay=config["weight_decay"])
    total = steps_per_epoch * config["max_epochs"]
    warmup = max(1, round(total * config["warmup_fraction"]))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        position = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.1 + 0.45 * (1 + math.cos(math.pi * position))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(config_path, phase, device):
    config_path = Path(config_path)
    config = strict_loads(config_path.read_bytes())
    verify_bindings(config)
    output, config_hash = Path(config["output"]), file_sha256(config_path)
    if (output / "training_completed.json").exists():
        raise ValueError("Production already completed; audit it instead of retraining")
    seed_all(config["seed"], cuda=device.type == "cuda")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = RRData(config["input_manifest"], config["labels_path"])
    model = build_classifier(config, device)
    frozen_initial = frozen_sha256(model)
    summary = {**parameter_summary(model), "frozen_initial_sha256": frozen_initial}
    if (output / "parameters.json").exists():
        if strict_loads((output / "parameters.json").read_bytes()) != summary:
            raise ValueError("Trainability/frozen initialization changed on resume")
    else:
        write_json(output / "parameters.json", summary)
    batch_count = math.ceil(len(data.rows["train"]) / config["batch_size"])
    steps_per_epoch = math.ceil(batch_count / config["accumulation"])
    optimizer, scheduler = make_optimizer(model, config, steps_per_epoch)
    progress = {"epoch": 0, "next_batch": 0, "step": 0, "sentences_seen": 0,
        "best_macro_f1": -1.0, "bad_epochs": 0, "epoch_sentences": 0,
        "training_seconds": 0.0, "epoch_weighted_loss_sum": 0.0, "epoch_indices_seen": []}
    recovery = output / "recovery.pt"
    if recovery.exists():
        if phase == "pilot" and (output / "pilot_completed.json").exists():
            raise ValueError("Pilot already completed; resume production instead")
        progress = restore_checkpoint(recovery, model, optimizer, scheduler, config_hash, resume_rng=True)
    elif phase == "production":
        raise ValueError("Production must resume its committed pilot")
    weights = torch.tensor(config["class_weights"], device=device)
    session, started = now(), time.perf_counter()
    append_json(output / "sessions.jsonl", {"at_utc": session, "phase": phase,
        "resume_step": progress["step"], "resume_epoch": progress["epoch"],
        "resume_next_batch": progress["next_batch"], "config_sha256": config_hash})
    times = []
    while progress["epoch"] < config["max_epochs"] and not progress.get("protocol_complete", False):
        batches = epoch_batches(len(data.rows["train"]), config["batch_size"], config["seed"], progress["epoch"])
        expected_prefix = sum(batches[:progress["next_batch"]], [])
        if progress["epoch_indices_seen"] != expected_prefix:
            raise ValueError("Recovery cursor and actually processed example IDs disagree")
        model.train()
        for first in range(progress["next_batch"], len(batches), config["accumulation"]):
            selected = batches[first:first + config["accumulation"]]
            indices_this_update = sum(selected, [])
            count = len(indices_this_update)
            optimizer.zero_grad(set_to_none=True)
            tick, loss_sum = time.perf_counter(), 0.0
            for indices in selected:
                rows = [data.rows["train"][i] for i in indices]
                gold = torch.tensor([row["label"] for row in rows], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(**collate(rows, config["context_tokens"], config["capacity"], device))
                losses, _, _ = loss_per_sentence(logits, gold, weights)
                if not torch.isfinite(losses).all():
                    raise ValueError("Nonfinite training loss")
                (losses.sum() / count).backward()
                loss_sum += losses.detach().sum().item()
            trainable = [p for p in model.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(trainable, config["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            times.append(seconds)
            progress["step"] += 1
            progress["next_batch"] = first + len(selected)
            progress["sentences_seen"] += count
            progress["epoch_sentences"] += count
            progress["epoch_indices_seen"].extend(indices_this_update)
            progress["epoch_weighted_loss_sum"] += loss_sum
            progress["training_seconds"] += seconds
            record = {"session": session, "epoch": progress["epoch"] + 1, "step": progress["step"],
                "next_batch": progress["next_batch"], "loss": loss_sum / count,
                "sentences_seen": progress["sentences_seen"], "sentences_this_update": count,
                "train_indices": indices_this_update, "gradient_norm": float(norm),
                "learning_rates": scheduler.get_last_lr(), "seconds": seconds}
            append_json(output / "training.jsonl", record)
            write_json(output / "progress.json", {"status": "training", "phase": phase, **record})
            if progress["step"] % config["checkpoint_every_updates"] == 0:
                save_checkpoint(recovery, model, optimizer, scheduler, progress, config_hash)
            if phase == "pilot" and progress["step"] >= config["pilot_updates"]:
                if frozen_sha256(model) != frozen_initial:
                    raise ValueError("Frozen backbone changed in pilot")
                save_checkpoint(recovery, model, optimizer, scheduler, progress, config_hash)
                diagnostic = data.rows["dev"][:64] + data.rows["dev"][-64:]
                from src.phase3_training import metric_details
                predictions, inference_seconds = predict(model, diagnostic, config, device)
                measured = times[2:] or times
                median = float(np.median(measured))
                pilot = {"passed": True, "retained_for_production": True,
                    "progress": {k:v for k,v in progress.items() if k != "epoch_indices_seen"},
                    "steps_per_epoch": steps_per_epoch, "median_update_seconds_excluding_two_warmups": median,
                    "training_seconds_per_epoch_estimate": median * steps_per_epoch,
                    "maximum_training_hours_estimate_without_eval": median * steps_per_epoch * config["max_epochs"] / 3600,
                    "bounded_dev_128_diagnostic_not_selection": metric_details(diagnostic, predictions, data.labels),
                    "diagnostic_inference_seconds": inference_seconds,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                    "recovery_sha256": file_sha256(recovery), "config_sha256": config_hash,
                    "wall_seconds": time.perf_counter() - started}
                write_json(output / "pilot_completed.json", pilot)
                write_json(output / "progress.json", {"status": "pilot_complete_waiting_for_production_resume",
                    "step": progress["step"], "sentences_seen": progress["sentences_seen"]})
                return pilot
        expected = sum(batches, [])
        if (progress["epoch_indices_seen"] != expected
                or sorted(expected) != list(range(len(data.rows["train"])))
                or progress["epoch_sentences"] != len(expected)):
            raise ValueError("Actual epoch coverage omitted or duplicated training examples")
        verify_bindings(config)
        if frozen_sha256(model) != frozen_initial:
            raise ValueError("Frozen pretrained weights changed")
        epoch = progress["epoch"] + 1
        metrics = evaluate(model, data, config, device, output / "dev" / f"epoch_{epoch:03d}")
        epoch_record = {"session": session, "epoch": epoch,
            "train_loss": progress["epoch_weighted_loss_sum"] / progress["epoch_sentences"],
            "train_sentences": progress["epoch_sentences"], "dev_macro_f1": metrics["macro_f1"],
            "actual_example_order_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
            "actual_examples_equal_complete_epoch_order": True, "evaluation_output": metrics["evaluation_output"]}
        improved = metrics["macro_f1"] > progress["best_macro_f1"]
        progress["bad_epochs"] = 0 if improved else progress["bad_epochs"] + 1
        if improved:
            progress["best_macro_f1"], progress["best_epoch"] = metrics["macro_f1"], epoch
        progress.update(epoch=epoch, next_batch=0, epoch_sentences=0,
                        epoch_weighted_loss_sum=0.0, epoch_indices_seen=[])
        progress["protocol_complete"] = (epoch >= config["max_epochs"] or
            (epoch >= config["min_epochs"] and progress["bad_epochs"] >= config["patience"]))
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, progress, config_hash)
        save_checkpoint(recovery, model, optimizer, scheduler, progress, config_hash)
        append_json(output / "epochs.jsonl", epoch_record)
    save_checkpoint(output / "final.pt", model, optimizer, scheduler, progress, config_hash)
    write_json(output / "training_completed.json", {"passed": True, "progress": progress,
        "frozen_initial_sha256": frozen_initial, "frozen_final_sha256": frozen_sha256(model),
        "best_sha256": file_sha256(output / "best.pt"), "final_sha256": file_sha256(output / "final.pt"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "last_session_wall_seconds": time.perf_counter() - started, "config_sha256": config_hash})
    del optimizer, scheduler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return export_candidate(config_path, device)


def export_candidate(config_path, device):
    config = strict_loads(Path(config_path).read_bytes())
    verify_bindings(config)
    output = Path(config["output"])
    if not (output / "training_completed.json").exists() or (output / "candidate_completed.json").exists():
        raise ValueError("Export requires completed training and no existing completed candidate")
    data = RRData(config["input_manifest"], config["labels_path"])
    model = build_classifier(config, device)
    progress = restore_checkpoint(output / "best.pt", model, config_hash=file_sha256(config_path))
    metrics = evaluate(model, data, config, device, output / "selected_dev")
    if abs(metrics["macro_f1"] - progress["best_macro_f1"]) > 1e-12:
        raise ValueError("Selected checkpoint failed to reproduce its development metric")
    sentence_only = evaluate(model, data, config, device, output / "selected_dev_sentence_only", context_tokens=0)
    indices, seconds = predict(model, data.rows["test"], config, device)
    write_json(output / "prediction.json", predictions_from_rows(data.rows["test"], indices, data.labels))
    trace = {"selected_checkpoint": str(output / "best.pt"),
        "selected_checkpoint_sha256": file_sha256(output / "best.pt"),
        "config_sha256": file_sha256(config_path), "tokenizer_sha256": config["tokenizer_sha256"],
        "input_manifest_sha256": file_sha256(config["input_manifest"]), "labels": data.labels,
        "test_sentence_keys": [list(row["key"]) for row in data.rows["test"]],
        "test_prediction_ids": indices, "inference_seconds": seconds}
    write_json(output / "inference_trace.json", trace)
    archive = output / "submission.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(output / "prediction.json", arcname="prediction.json")
    contract = load_test_contract(ROOT / "data/raw/Phase_3_RR", config["download_manifest"])
    validation = validate_submission(output / "prediction.json", archive, contract)
    write_json(output / "submission_validation.json", validation)
    record = {"candidate_local_complete": True, "architecture": config["architecture"], "method": "lora",
        "selected_checkpoint": str(output / "best.pt"), "selected_checkpoint_sha256": trace["selected_checkpoint_sha256"],
        "dev_macro_f1": metrics["macro_f1"], "sentence_only_dev_macro_f1": sentence_only["macro_f1"],
        "context_check_scope": "Same selected weights with/without inference context, not matched training conditions",
        "zip": str(archive), "zip_sha256": validation["zip_sha256"],
        "inference_trace_sha256": file_sha256(output / "inference_trace.json"),
        "config_sha256": file_sha256(config_path), "submitted_to_codabench": False,
        "independent_final_audit_still_required": True}
    write_json(output / "candidate_completed.json", record)
    return record
