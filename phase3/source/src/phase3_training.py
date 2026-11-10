"""One frozen RR experiment, with sentence-normalized losses and exact cursors."""

import datetime
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.phase3_data import RRData, collate, epoch_batches, predictions_from_rows
from src.phase3_model import RRClassifier, frozen_sha256, load_backbone, parameter_summary
from src.phase3_submission import file_sha256, load_test_contract, strict_loads, validate_submission

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()


def verify_bindings(config):
    for path, expected in config["bindings"].items():
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen experiment input/code changed: {path}")


def seed_all(seed, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def rng_state(cuda=False):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": [torch.cuda.get_rng_state(0)] if cuda else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != 1:
            raise ValueError("This experiment supports one visible authorized CUDA device")
        torch.cuda.set_rng_state(state["cuda"][0], 0)


def save_checkpoint(path, model, optimizer, scheduler, progress, config_hash):
    record = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
              "scheduler": scheduler.state_dict(), "rng": rng_state(next(model.parameters()).is_cuda),
              "progress": progress, "config_sha256": config_hash}
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def restore_checkpoint(path, model, optimizer=None, scheduler=None, config_hash=None, resume_rng=False):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if config_hash is not None and saved["config_sha256"] != config_hash:
        raise ValueError("Recovery checkpoint belongs to another frozen configuration")
    model.load_state_dict(saved["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(saved["scheduler"])
    if resume_rng:
        restore_rng(saved["rng"])
    return saved["progress"]


def build_model(config, device):
    model = RRClassifier(load_backbone(config["checkpoint"], config["model_config"]),
                         config["method"], config["rank"], config["alpha"])
    return model.to(device)


def loss_per_sentence(logits, gold, class_weights, teacher=None, temperature=2.0, kd_weight=0.5):
    ce = F.cross_entropy(logits.float(), gold, reduction="none") * class_weights[gold]
    if teacher is None:
        return ce, ce, torch.zeros_like(ce)
    if teacher.shape != logits.shape or not torch.isfinite(teacher).all():
        raise ValueError("Teacher class distribution alignment/values are invalid")
    if (teacher < 0).any() or not torch.allclose(teacher.sum(-1), torch.ones_like(teacher[:, 0]), atol=1e-5):
        raise ValueError("Teacher probabilities do not sum to one")
    kl = F.kl_div(F.log_softmax(logits.float() / temperature, -1), teacher.float(),
                  reduction="none").sum(-1) * temperature ** 2
    return (1 - kd_weight) * ce + kd_weight * kl, ce, kl


@torch.no_grad()
def predict(model, rows, config, device, context_tokens=None, return_logits=False):
    model.eval()
    values = []
    started = time.perf_counter()
    for start in range(0, len(rows), config["batch_size"]):
        batch = rows[start:start + config["batch_size"]]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**collate(batch, config["context_tokens"] if context_tokens is None else context_tokens,
                                     config["capacity"], device))
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite sentence logits")
        values.append(logits.float().cpu())
    combined = torch.cat(values)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (combined if return_logits else combined.argmax(-1).tolist()), elapsed


def metric_details(rows, predicted, labels):
    gold = [row["label"] for row in rows]
    classes = list(range(len(labels)))
    return {"macro_f1": float(f1_score(gold, predicted, labels=classes, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(gold, predicted, labels=classes, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(gold, predicted, labels=classes, average="macro", zero_division=0)),
            "per_label": classification_report(gold, predicted, labels=classes, target_names=labels,
                                                output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(gold, predicted, labels=classes).tolist(),
            "label_order": labels, "sentences": len(rows)}


def evaluate(model, data, config, device, output, context_tokens=None):
    output = Path(output)
    original, attempt = output, 1
    while output.exists():
        attempt += 1
        output = original.with_name(original.name + f"_attempt_{attempt:03d}")
    output.mkdir(parents=True, exist_ok=False)
    rows = data.rows["dev"]
    predicted, elapsed = predict(model, rows, config, device, context_tokens)
    predictions = predictions_from_rows(rows, predicted, data.labels)
    prediction_path = output / "prediction.json"
    write_json(prediction_path, predictions)
    for domain in ("CL", "IT"):
        subset = [row for row in rows if row["doc"]["domain"] == domain]
        truth = predictions_from_rows(subset, [row["label"] for row in subset], data.labels)
        write_json(output / f"{domain}_truth.json", truth)
    command = [sys.executable, str(ROOT / "official/evaluation_scripts/evaluate_phase3.py"),
               "--truth-cl", str(output / "CL_truth.json"), "--truth-it", str(output / "IT_truth.json"),
               "--prediction", str(prediction_path), "--labels", config["labels_path"]]
    official = subprocess.run(command, capture_output=True, text=True, cwd=ROOT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    (output / "official_output.txt").write_text(official.stdout + official.stderr)
    if official.returncode:
        raise RuntimeError("Official dev evaluator failed")
    details = metric_details(rows, predicted, data.labels)
    printed = {line.split(":")[0].strip(): float(line.split(":")[1].strip())
               for line in official.stdout.splitlines() if line.startswith("Macro-")}
    for printed_key, key in (("Macro-F1", "macro_f1"), ("Macro-Precision", "macro_precision"),
                             ("Macro-Recall", "macro_recall")):
        if abs(printed[printed_key] - details[key]) > 0.00000051:
            raise ValueError("Diagnostic metrics disagree with the unchanged official evaluator")
    details["domains"] = {}
    for domain in ("CL", "IT"):
        indices = [i for i, row in enumerate(rows) if row["doc"]["domain"] == domain]
        details["domains"][domain] = metric_details([rows[i] for i in indices], [predicted[i] for i in indices], data.labels)
    details.update({"official_command": command, "official_returncode": official.returncode,
                    "inference_seconds": elapsed, "context_tokens": config["context_tokens"] if context_tokens is None else context_tokens,
                    "evaluation_output": str(output)})
    write_json(output / "metrics.json", details)
    return details


def load_teacher_cache(config, data):
    manifest_path = Path(config["teacher_cache"])
    manifest = strict_loads(manifest_path.read_bytes())
    for name, expected in manifest["bindings"].items():
        if file_sha256(name) != expected:
            raise ValueError("Teacher cache binding changed")
    keys = [list(row["key"]) for row in data.rows["train"]]
    if (manifest["keys"] != keys or manifest["labels"] != data.labels
            or manifest["temperature"] != config["temperature"]
            or manifest["context_tokens"] != config["context_tokens"]
            or manifest["capacity"] != config["capacity"]
            or manifest["tokenizer_sha256"] != config["tokenizer_sha256"]
            or manifest["input_manifest_sha256"] != file_sha256(config["input_manifest"])):
        raise ValueError("Teacher/student input, sentence or class alignment mismatch")
    probs = torch.from_numpy(np.load(manifest["probabilities"], allow_pickle=False)).float()
    if probs.shape != (16845, 13) or not torch.isfinite(probs).all():
        raise ValueError("Incomplete or nonfinite teacher targets")
    return probs, {"manifest": str(manifest_path), "sha256": file_sha256(manifest_path)}


def train(config_path, phase, device):
    config_path = Path(config_path)
    config = strict_loads(config_path.read_bytes())
    verify_bindings(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    config_hash = file_sha256(config_path)
    seed_all(config["seed"], cuda=device.type == "cuda")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = RRData(config["input_manifest"], config["labels_path"])
    model = build_model(config, device)
    summary = parameter_summary(model)
    frozen_initial = frozen_sha256(model)
    write_json(output / "parameters.json", {**summary, "frozen_initial_sha256": frozen_initial})
    body = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("classifier.")]
    head = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([{"params": body, "lr": config["learning_rate"]},
                                  {"params": head, "lr": config["head_learning_rate"]}],
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=config["weight_decay"])
    steps_per_epoch = math.ceil(math.ceil(len(data.rows["train"]) / config["batch_size"]) / config["accumulation"])
    total_steps = steps_per_epoch * config["max_epochs"]
    warmup = max(1, round(total_steps * config["warmup_fraction"]))
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        position = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * position))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    progress = {"epoch": 0, "next_batch": 0, "step": 0, "sentences_seen": 0,
                "best_macro_f1": -1.0, "bad_epochs": 0, "epoch_sentences": 0,
                "training_seconds": 0.0, "epoch_weighted_loss_sum": 0.0}
    recovery = output / "recovery.pt"
    if recovery.exists():
        if phase == "pilot" and (output / "pilot_completed.json").exists():
            raise ValueError("Pilot already completed; use production to resume it")
        progress = restore_checkpoint(recovery, model, optimizer, scheduler, config_hash, resume_rng=True)
    elif phase == "production":
        raise ValueError("Production must resume the committed pilot, not start over")
    teacher = None
    if config["method"] == "knowledge_distillation":
        teacher, binding = load_teacher_cache(config, data)
        write_json(output / "teacher_binding.json", binding)
    weights = torch.tensor(config["class_weights"], device=device)
    run_start = time.perf_counter()
    pilot_times = []
    session = now()
    while progress["epoch"] < config["max_epochs"] and not progress.get("protocol_complete", False):
        batches = epoch_batches(len(data.rows["train"]), config["batch_size"], config["seed"], progress["epoch"])
        model.train()
        for first in range(progress["next_batch"], len(batches), config["accumulation"]):
            selected = batches[first:first + config["accumulation"]]
            count = sum(map(len, selected))
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            loss_sum = ce_sum = kl_sum = 0.0
            for indices in selected:
                rows = [data.rows["train"][i] for i in indices]
                gold = torch.tensor([row["label"] for row in rows], device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(**collate(rows, config["context_tokens"], config["capacity"], device))
                targets = teacher[indices].to(device) if teacher is not None else None
                losses, ce, kl = loss_per_sentence(logits, gold, weights, targets,
                                                  config["temperature"], config["kd_weight"])
                if not torch.isfinite(losses).all():
                    raise ValueError("Nonfinite training loss")
                (losses.sum() / count).backward()
                loss_sum += losses.detach().sum().item()
                ce_sum += ce.detach().sum().item()
                kl_sum += kl.detach().sum().item()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                  config["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            pilot_times.append(seconds)
            progress["step"] += 1
            progress["next_batch"] = first + len(selected)
            progress["sentences_seen"] += count
            progress["epoch_sentences"] += count
            progress["epoch_weighted_loss_sum"] += loss_sum
            progress["training_seconds"] += seconds
            record = {"session": session, **progress, "loss": loss_sum / count, "ce": ce_sum / count,
                      "kl": kl_sum / count, "sentences_this_update": count, "gradient_norm": float(norm),
                      "clipped": bool(norm > config["clip_norm"]), "lr": scheduler.get_last_lr(),
                      "seconds": seconds}
            append_json(output / "training.jsonl", record)
            write_json(output / "progress.json", {"status": "training", "phase": phase, **record})
            if progress["step"] % config["checkpoint_every_updates"] == 0:
                save_checkpoint(recovery, model, optimizer, scheduler, progress, config_hash)
            if phase == "pilot" and progress["step"] >= config["pilot_updates"]:
                if frozen_sha256(model) != frozen_initial:
                    raise ValueError("Frozen backbone changed during pilot")
                save_checkpoint(recovery, model, optimizer, scheduler, progress, config_hash)
                # A bounded, explicitly non-selection dev check; full dev selects only at epoch end.
                diagnostic_rows = data.rows["dev"][:64] + data.rows["dev"][-64:]
                pred, inference_seconds = predict(model, diagnostic_rows, config, device)
                useful = pilot_times[2:] or pilot_times
                estimate = float(np.median(useful)) * steps_per_epoch
                pilot = {"passed": True, "progress": progress, "retained_for_production": True,
                    "median_update_seconds_excluding_two_warmups": float(np.median(useful)),
                    "training_seconds_per_epoch_estimate": estimate, "steps_per_epoch": steps_per_epoch,
                    "maximum_training_hours_estimate_without_eval": estimate * config["max_epochs"] / 3600,
                    "bounded_dev_128_diagnostic_not_selection": metric_details(diagnostic_rows, pred, data.labels),
                    "diagnostic_inference_seconds": inference_seconds,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                    "recovery_sha256": file_sha256(recovery), "config_sha256": config_hash,
                    "wall_seconds": time.perf_counter() - run_start}
                write_json(output / "pilot_completed.json", pilot)
                write_json(output / "progress.json", {"status": "pilot_complete_waiting_for_production_resume", **progress})
                return pilot
        if progress["epoch_sentences"] != len(data.rows["train"]):
            raise ValueError("An epoch omitted or repeated training sentences")
        verify_bindings(config)
        if frozen_sha256(model) != frozen_initial:
            raise ValueError("Frozen backbone changed during training")
        epoch = progress["epoch"] + 1
        metrics = evaluate(model, data, config, device, output / "dev" / f"epoch_{epoch:03d}")
        epoch_record = {"epoch": epoch, "train_loss": progress["epoch_weighted_loss_sum"] / progress["epoch_sentences"],
                        "train_sentences": progress["epoch_sentences"], "dev_macro_f1": metrics["macro_f1"]}
        append_json(output / "epochs.jsonl", epoch_record)
        improved = metrics["macro_f1"] > progress["best_macro_f1"]
        progress["bad_epochs"] = 0 if improved else progress["bad_epochs"] + 1
        if improved:
            progress["best_macro_f1"], progress["best_epoch"] = metrics["macro_f1"], epoch
        progress["epoch"], progress["next_batch"] = epoch, 0
        progress["epoch_sentences"], progress["epoch_weighted_loss_sum"] = 0, 0.0
        progress["protocol_complete"] = epoch >= config["max_epochs"] or (epoch >= config["min_epochs"] and progress["bad_epochs"] >= config["patience"])
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, progress, config_hash)
        save_checkpoint(recovery, model, optimizer, scheduler, progress, config_hash)
        if progress["protocol_complete"]:
            break
    save_checkpoint(output / "final.pt", model, optimizer, scheduler, progress, config_hash)
    write_json(output / "training_completed.json", {"passed": True, "progress": progress,
        "frozen_initial_sha256": frozen_initial, "frozen_final_sha256": frozen_sha256(model),
        "best_sha256": file_sha256(output / "best.pt"), "final_sha256": file_sha256(output / "final.pt"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "last_session_wall_seconds": time.perf_counter() - run_start, "config_sha256": config_hash})
    return export_candidate(config_path, device)


def export_candidate(config_path, device):
    config = strict_loads(Path(config_path).read_bytes())
    verify_bindings(config)
    output = Path(config["output"])
    if not (output / "training_completed.json").exists():
        raise ValueError("Cannot export before the declared training protocol finishes")
    data = RRData(config["input_manifest"], config["labels_path"])
    model = build_model(config, device)
    progress = restore_checkpoint(output / "best.pt", model, config_hash=file_sha256(config_path))
    metrics = evaluate(model, data, config, device, output / "selected_dev")
    if abs(metrics["macro_f1"] - progress["best_macro_f1"]) > 1e-12:
        raise ValueError("Reloaded selected checkpoint does not reproduce its dev score")
    sentence_only = evaluate(model, data, config, device, output / "selected_dev_sentence_only", context_tokens=0)
    indices, seconds = predict(model, data.rows["test"], config, device)
    predictions = predictions_from_rows(data.rows["test"], indices, data.labels)
    write_json(output / "prediction.json", predictions)
    trace = {"selected_checkpoint": str(output / "best.pt"), "selected_checkpoint_sha256": file_sha256(output / "best.pt"),
             "config_sha256": file_sha256(config_path), "tokenizer_sha256": config["tokenizer_sha256"],
             "labels": data.labels, "input_manifest_sha256": file_sha256(config["input_manifest"]),
             "test_sentence_keys": [list(row["key"]) for row in data.rows["test"]],
             "test_prediction_ids": indices, "inference_seconds": seconds}
    write_json(output / "inference_trace.json", trace)
    archive = output / "submission.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.write(output / "prediction.json", arcname="prediction.json")
    contract = load_test_contract(ROOT / "data/raw/Phase_3_RR", config["download_manifest"])
    validation = validate_submission(output / "prediction.json", archive, contract)
    write_json(output / "submission_validation.json", validation)
    record = {"candidate_local_complete": True, "architecture": config["architecture"], "method": config["method"],
              "selected_checkpoint": str(output / "best.pt"), "selected_checkpoint_sha256": trace["selected_checkpoint_sha256"],
              "dev_macro_f1": metrics["macro_f1"], "sentence_only_dev_macro_f1": sentence_only["macro_f1"],
              "context_check_scope": "Same selected weights with/without inference context, not matched training conditions",
              "zip": str(archive), "zip_sha256": validation["zip_sha256"],
              "inference_trace_sha256": file_sha256(output / "inference_trace.json"),
              "config_sha256": file_sha256(config_path), "submitted_to_codabench": False}
    write_json(output / "candidate_completed.json", record)
    return record


def cache_teacher(teacher_config_path, student_config_path, device):
    teacher_config = strict_loads(Path(teacher_config_path).read_bytes())
    student_config = strict_loads(Path(student_config_path).read_bytes())
    verify_bindings(teacher_config)
    verify_bindings(student_config)
    teacher_output = Path(teacher_config["output"])
    if not (teacher_output / "candidate_completed.json").exists():
        raise ValueError("MHA teacher is not a completed, verified RR candidate")
    for key in ("input_manifest", "context_tokens", "capacity", "tokenizer_sha256", "labels_path"):
        if teacher_config[key] != student_config[key]:
            raise ValueError(f"Teacher/student policies differ: {key}")
    target = Path(student_config["teacher_cache"])
    if target.exists():
        raise FileExistsError("Existing teacher cache must be audited, not overwritten")
    data = RRData(teacher_config["input_manifest"], teacher_config["labels_path"])
    teacher = build_model(teacher_config, device)
    selected = teacher_output / "best.pt"
    restore_checkpoint(selected, teacher, config_hash=file_sha256(teacher_config_path))
    teacher.requires_grad_(False)
    teacher.eval()
    student_total = sum(p.numel() for p in load_backbone(student_config["checkpoint"]).parameters()) + 13 * (student_config["model_config"]["d_model"] + 1)
    teacher_total = sum(p.numel() for p in teacher.parameters())
    if teacher_total <= student_total:
        raise ValueError("Actual teacher is not larger than the student")
    logits, elapsed = predict(teacher, data.rows["train"], teacher_config, device, return_logits=True)
    probabilities = torch.softmax(logits / student_config["temperature"], dim=-1).numpy()
    if probabilities.shape != (16845, 13) or not np.isfinite(probabilities).all():
        raise ValueError("Teacher did not cover all training sentences")
    target.parent.mkdir(parents=True, exist_ok=True)
    probability_path = target.with_suffix(".npy")
    with probability_path.open("xb") as stream:
        np.save(stream, probabilities, allow_pickle=False)
    manifest = {"probabilities": str(probability_path), "keys": [list(row["key"]) for row in data.rows["train"]],
        "labels": data.labels, "temperature": student_config["temperature"],
        "context_tokens": student_config["context_tokens"], "capacity": student_config["capacity"],
        "tokenizer_sha256": student_config["tokenizer_sha256"],
        "input_manifest_sha256": file_sha256(student_config["input_manifest"]),
        "teacher_total_parameters": teacher_total, "student_total_parameters": student_total,
        "training_sentences_only": True, "teacher_frozen_eval": True, "inference_seconds": elapsed,
        "bindings": {str(selected): file_sha256(selected), str(teacher_config_path): file_sha256(teacher_config_path),
                     str(probability_path): file_sha256(probability_path)}}
    write_json(target, manifest)
    return {"teacher_cache": str(target), "sha256": file_sha256(target), "sentences": 16845}
