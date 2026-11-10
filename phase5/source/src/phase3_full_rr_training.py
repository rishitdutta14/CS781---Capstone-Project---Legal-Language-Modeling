"""Two-rank RR fine-tuning; global sentence means, no padded/repeated examples."""

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.phase3_data import RRData, epoch_batches, predictions_from_rows
from src.phase3_full_rr import (FullRRClassifier, adapter_state, atomic_torch_save,
    full_collate, full_decoder_hidden, load_full_base, restore_adapter_state)
from src.phase3_model import frozen_sha256, parameter_summary
from src.phase3_rebuilt_training import make_optimizer
from src.phase3_submission import file_sha256, load_test_contract, strict_loads, validate_submission
from src.phase3_training import (ROOT, append_json, loss_per_sentence, metric_details,
    now, restore_rng, rng_state, seed_all, verify_bindings, write_json)


def sentence_loss_scale(global_sentences, world_size):
    if global_sentences < 1 or world_size < 1:
        raise ValueError("Global sentence count/world size must be positive")
    # DDP averages gradients over ranks. Unequal rank counts are NOT separate
    # equally weighted batch means: their summed losses share one denominator.
    return world_size / global_sentences


def save_checkpoint(path, model, optimizer, scheduler, progress, config):
    for state in optimizer.state.values():
        if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Nonfinite adapter optimizer state")
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng_state(cuda=True))
    if dist.get_rank() == 0:
        atomic_torch_save(path, {"schema": "full_rr_lora_only_v1", "trainable": adapter_state(model),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng_by_rank": states, "progress": progress,
            "config_sha256": config["config_sha256"], "frozen_base_sha256": config["checkpoint_sha256"]})
    dist.barrier()


def restore_checkpoint(path, model, optimizer, scheduler, config, resume_rng=False):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved["schema"] != "full_rr_lora_only_v1" or saved["config_sha256"] != config["config_sha256"]
            or saved["frozen_base_sha256"] != config["checkpoint_sha256"]):
        raise ValueError("Adapter checkpoint/base/config binding mismatch")
    restore_adapter_state(model, saved["trainable"])
    if optimizer is not None:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
    if resume_rng:
        if len(saved["rng_by_rank"]) != dist.get_world_size():
            raise ValueError("Recovery requires the original two-rank layout")
        restore_rng(saved["rng_by_rank"][dist.get_rank()])
    return saved["progress"]


@torch.no_grad()
def predict(model, rows, config, device, no_context=False):
    model.eval()
    rank, world = dist.get_rank(), dist.get_world_size()
    indices = list(range(rank, len(rows), world))
    results, started = [], time.perf_counter()
    for first in range(0, len(indices), config["microbatch"]):
        selected = indices[first:first + config["microbatch"]]
        batch = full_collate([rows[i] for i in selected], config["context_tokens"], config["capacity"], device)
        if no_context:
            # Keep target chunking fixed. Remove only encoder context, not
            # silently change target segmentation when doing this ablation.
            batch["source_ids"] = torch.full((batch["decoder_ids"].shape[0], 1), 2, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite RR inference logits")
        results.extend(zip(selected, logits.argmax(-1).cpu().tolist()))
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - started, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    gathered = [None] * world
    dist.all_gather_object(gathered, results)
    merged = sorted(item for part in gathered for item in part)
    if [i for i, _ in merged] != list(range(len(rows))):
        raise ValueError("Distributed inference omitted, duplicated or reordered sentences")
    return [label for _, label in merged], float(elapsed)


def evaluate(model, data, config, device, output, no_context=False):
    predictions, elapsed = predict(model, data.rows["dev"], config, device, no_context)
    details = None
    if dist.get_rank() == 0:
        output = Path(output)
        if output.exists():
            raise FileExistsError("Preserve the previous evaluation attempt")
        output.mkdir(parents=True)
        rows = data.rows["dev"]
        prediction = output / "prediction.json"
        write_json(prediction, predictions_from_rows(rows, predictions, data.labels))
        for domain in ("CL", "IT"):
            subset = [row for row in rows if row["doc"]["domain"] == domain]
            write_json(output / (domain + "_truth.json"), predictions_from_rows(subset, [row["label"] for row in subset], data.labels))
        command = [sys.executable, str(ROOT / "official/evaluation_scripts/evaluate_phase3.py"),
            "--truth-cl", str(output / "CL_truth.json"), "--truth-it", str(output / "IT_truth.json"),
            "--prediction", str(prediction), "--labels", config["labels_path"]]
        import os
        official = subprocess.run(command, capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd=ROOT)
        with (output / "official_output.txt").open("x") as stream:
            stream.write(official.stdout + official.stderr)
        if official.returncode:
            raise RuntimeError("Official Phase3 development evaluator failed")
        details = metric_details(rows, predictions, data.labels)
        printed = {line.split(":")[0]: float(line.split(":")[1])
                   for line in official.stdout.splitlines() if line.startswith("Macro-")}
        for key, metric in (("Macro-F1", "macro_f1"), ("Macro-Precision", "macro_precision"), ("Macro-Recall", "macro_recall")):
            if abs(printed[key] - details[metric]) > .00000051:
                raise ValueError("Diagnostics differ from the unchanged official evaluator")
        details["domains"] = {}
        for domain in ("CL", "IT"):
            positions = [i for i, row in enumerate(rows) if row["doc"]["domain"] == domain]
            details["domains"][domain] = metric_details([rows[i] for i in positions], [predictions[i] for i in positions], data.labels)
        details.update(inference_seconds=elapsed, no_context=no_context, official_command=command,
                       official_returncode=official.returncode, evaluation_output=str(output))
        write_json(output / "metrics.json", details)
    shared = [details]
    dist.broadcast_object_list(shared, src=0)
    return shared[0]


def train(config_path, phase, device):
    config = strict_loads(Path(config_path).read_bytes())
    # load_full_base below verifies the3GB checkpoint itself. Verify all other
    # bindings here rather than redundantly stream the same file twice/rank.
    verify_bindings({**config, "bindings": {name: digest for name, digest in config["bindings"].items()
                                            if name != config["checkpoint"]}})
    config["config_sha256"] = file_sha256(config_path)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError("This experiment requires exactly two GPUs")
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(config["seed"], cuda=True)
    data = RRData(config["input_manifest"], config["labels_path"])
    base = load_full_base(config["checkpoint"], config["checkpoint_sha256"], config["tokenizer_sha256"])
    base = base.to(device).eval()
    source = torch.arange(4, 12, device=device).unsqueeze(0)
    target = torch.arange(12, 28, device=device).unsqueeze(0)
    with torch.no_grad():
        reference = base(source, target, source != 0, target != 0)
    model = FullRRClassifier(base, config["rank"], config["alpha"]).to(device).eval()
    with torch.no_grad():
        restored = base.head(full_decoder_hidden(base, source, target))
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)
    write_json(Path(config["output"]) / f"real_preflight_{phase}_rank{rank}.json", {
        "passed": True, "strict251M_original_restore": True,
        "initial_LoRA_original_full_forward_identity_exact": True,
        "tokenizer_sha256": config["tokenizer_sha256"], "base_sha256": config["checkpoint_sha256"],
        "FP32_probe_max_abs_logit_difference": float((restored-reference).abs().max())})
    del reference, restored, source, target
    frozen_initial = frozen_sha256(model)
    output = Path(config["output"])
    summary = {**parameter_summary(model), "frozen_initial_sha256": frozen_initial}
    if rank == 0:
        path = output / "parameters.json"
        if path.exists() and strict_loads(path.read_bytes()) != summary:
            raise ValueError("Trainability changed on resume")
        if not path.exists():
            write_json(path, summary)
    wrapped = DDP(model, device_ids=[0], output_device=0, broadcast_buffers=False, gradient_as_bucket_view=True)
    seed_all(config["seed"] + rank, cuda=True)
    steps_per_epoch = math.ceil(len(data.rows["train"]) / config["global_batch_size"])
    optimizer, scheduler = make_optimizer(model, config, steps_per_epoch)
    progress = {"epoch": 0, "next_update": 0, "step": 0, "sentences_seen": 0,
        "best_macro_f1": -1., "bad_epochs": 0, "epoch_sentences": 0,
        "epoch_loss_sum": 0., "epoch_indices_seen": [], "training_seconds": 0.}
    recovery = output / "recovery.pt"
    if (output / "training_completed.json").exists():
        raise ValueError("Training already completed; do not rerun it")
    if recovery.exists():
        if phase == "pilot" and (output / "pilot_completed.json").exists():
            raise ValueError("Pilot already passed; resume production")
        progress = restore_checkpoint(recovery, model, optimizer, scheduler, config, resume_rng=True)
    elif phase == "production":
        raise ValueError("Production must resume its committed pilot")
    weights = torch.tensor(config["class_weights"], device=device)
    started, times = time.perf_counter(), []
    session = now()
    if rank == 0:
        append_json(output / "sessions.jsonl", {"phase": phase, "session": session, "resume_step": progress["step"],
            "resume_epoch": progress["epoch"], "resume_next_update": progress["next_update"]})
    while progress["epoch"] < config["max_epochs"] and not progress.get("protocol_complete", False):
        batches = epoch_batches(len(data.rows["train"]), config["global_batch_size"], config["seed"], progress["epoch"])
        if progress["epoch_indices_seen"] != sum(batches[:progress["next_update"]], []):
            raise ValueError("Recovery cursor does not match actually consumed examples")
        wrapped.train()
        for step in range(progress["next_update"], len(batches)):
            global_indices = batches[step]
            local_indices = global_indices[rank::world]
            observed = [None] * world
            dist.all_gather_object(observed, local_indices)
            if sorted(i for part in observed for i in part) != sorted(global_indices):
                raise ValueError("Distributed training duplicated or dropped real examples")
            optimizer.zero_grad(set_to_none=True)
            tick, local_loss = time.perf_counter(), 0.
            microbatches = [local_indices[i:i + config["microbatch"]] for i in range(0, len(local_indices), config["microbatch"])] or [[]]
            for number, indices in enumerate(microbatches):
                rows = [data.rows["train"][i] for i in indices] if indices else [data.rows["train"][0]]
                sync = nullcontext() if number == len(microbatches) - 1 else wrapped.no_sync()
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = wrapped(**full_collate(rows, config["context_tokens"], config["capacity"], device))
                    gold = torch.tensor([row["label"] for row in rows], device=device)
                    losses, _, _ = loss_per_sentence(logits, gold, weights)
                    if not torch.isfinite(losses).all():
                        raise ValueError("Nonfinite training objective")
                    total = losses.sum() if indices else losses.sum() * 0
                    (total * sentence_loss_scale(len(global_indices), world)).backward()
                    if indices:
                        local_loss += float(total.detach())
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                  config["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            totals = torch.tensor([local_loss, len(local_indices)], dtype=torch.float64, device=device)
            dist.all_reduce(totals)
            if int(totals[1]) != len(global_indices):
                raise ValueError("Actual distributed sentence denominator changed")
            torch.cuda.synchronize(device)
            seconds = torch.tensor(time.perf_counter() - tick, device=device)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            seconds = float(seconds)
            times.append(seconds)
            progress["step"] += 1
            progress["next_update"] = step + 1
            progress["sentences_seen"] += len(global_indices)
            progress["epoch_sentences"] += len(global_indices)
            progress["epoch_loss_sum"] += float(totals[0])
            progress["epoch_indices_seen"].extend(global_indices)
            progress["training_seconds"] += seconds
            append_json(output / f"rank{rank}_coverage.jsonl", {"session": session, "step": progress["step"],
                "real_indices": local_indices, "zero_loss_dummy_rows": int(not local_indices)})
            if rank == 0:
                record = {"session": session, "epoch": progress["epoch"] + 1, "step": progress["step"],
                    "next_update": progress["next_update"], "loss": float(totals[0]) / len(global_indices),
                    "sentences_this_update": len(global_indices), "train_indices": global_indices,
                    "sentences_seen": progress["sentences_seen"], "gradient_norm": float(norm),
                    "learning_rates": scheduler.get_last_lr(), "seconds": seconds}
                append_json(output / "training.jsonl", record)
                write_json(output / "progress.json", {"status": "training", "phase": phase, **record})
            if progress["step"] % config["checkpoint_every_updates"] == 0:
                save_checkpoint(recovery, model, optimizer, scheduler, progress, config)
            if phase == "pilot" and progress["step"] >= config["pilot_updates"]:
                if frozen_sha256(model) != frozen_initial:
                    raise ValueError("Pilot changed frozen pretrained parameters")
                save_checkpoint(recovery, model, optimizer, scheduler, progress, config)
                diagnostic = data.rows["dev"][:64] + data.rows["dev"][-64:]
                predictions, infer = predict(model, diagnostic, config, device)
                memory = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.int64)
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                if rank == 0:
                    median = float(np.median(times[2:] or times))
                    pilot = {"passed": True, "retained_for_production": True, "steps_per_epoch": steps_per_epoch,
                        "progress": {k:v for k,v in progress.items() if k != "epoch_indices_seen"},
                        "median_update_seconds_excluding_two_warmups": median,
                        "training_seconds_per_epoch_estimate": median * steps_per_epoch,
                        "maximum_training_hours_estimate_without_eval": median * steps_per_epoch * config["max_epochs"] / 3600,
                        "peak_allocated_bytes_max_over_ranks": int(memory), "diagnostic_inference_seconds": infer,
                        "bounded_dev_128_diagnostic_not_selection": metric_details(diagnostic, predictions, data.labels),
                        "recovery_sha256": file_sha256(recovery), "config_sha256": config["config_sha256"],
                        "wall_seconds": time.perf_counter() - started}
                    write_json(output / "pilot_completed.json", pilot)
                    write_json(output / "progress.json", {"status": "pilot_complete_waiting_for_production_resume", "step": progress["step"]})
                dist.barrier()
                return
        if (progress["epoch_indices_seen"] != sum(batches, [])
                or sorted(progress["epoch_indices_seen"]) != list(range(len(data.rows["train"])))):
            raise ValueError("Epoch did not consume all and only the real training sentences")
        if frozen_sha256(model) != frozen_initial:
            raise ValueError("Frozen pretrained backbone changed")
        epoch = progress["epoch"] + 1
        metrics = evaluate(model, data, config, device, output / "dev" / f"epoch_{epoch:03d}")
        if rank == 0:
            append_json(output / "epochs.jsonl", {"epoch": epoch, "train_loss": progress["epoch_loss_sum"] / progress["epoch_sentences"],
                "train_sentences": progress["epoch_sentences"], "dev_macro_f1": metrics["macro_f1"],
                "actual_example_order_sha256": hashlib.sha256(json.dumps(progress["epoch_indices_seen"]).encode()).hexdigest(),
                "each_real_example_exactly_once": True})
        improved = metrics["macro_f1"] > progress["best_macro_f1"]
        progress["bad_epochs"] = 0 if improved else progress["bad_epochs"] + 1
        if improved:
            progress["best_macro_f1"], progress["best_epoch"] = metrics["macro_f1"], epoch
        progress.update(epoch=epoch, next_update=0, epoch_indices_seen=[], epoch_sentences=0, epoch_loss_sum=0.)
        progress["protocol_complete"] = epoch >= config["max_epochs"] or (epoch >= config["min_epochs"] and progress["bad_epochs"] >= config["patience"])
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, progress, config)
        save_checkpoint(recovery, model, optimizer, scheduler, progress, config)
    save_checkpoint(output / "final.pt", model, optimizer, scheduler, progress, config)
    if rank == 0:
        write_json(output / "training_completed.json", {"passed": True, "progress": progress,
            "frozen_initial_sha256": frozen_initial, "frozen_final_sha256": frozen_sha256(model),
            "checkpoint_schema": "Adapter/head and optimizer/scheduler/RNG; original frozen Phase1 checkpoint is referenced, not copied"})
    selected = restore_checkpoint(output / "best.pt", model, None, None, config)
    metrics = evaluate(model, data, config, device, output / "selected_dev")
    if abs(metrics["macro_f1"] - selected["best_macro_f1"]) > 1e-12:
        raise ValueError("Selected checkpoint did not reproduce its development metric")
    no_context = evaluate(model, data, config, device, output / "selected_dev_sentence_only", no_context=True)
    predictions, seconds = predict(model, data.rows["test"], config, device)
    if rank == 0:
        write_json(output / "prediction.json", predictions_from_rows(data.rows["test"], predictions, data.labels))
        trace = {"selected_checkpoint_sha256": file_sha256(output / "best.pt"), "config_sha256": config["config_sha256"],
            "frozen_base_sha256": config["checkpoint_sha256"], "tokenizer_sha256": config["tokenizer_sha256"],
            "labels": data.labels, "test_sentence_keys": [list(row["key"]) for row in data.rows["test"]],
            "test_prediction_ids": predictions, "inference_seconds": seconds}
        write_json(output / "inference_trace.json", trace)
        archive = output / "submission.zip"
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
            stream.write(output / "prediction.json", arcname="prediction.json")
        validation = validate_submission(output / "prediction.json", archive,
            load_test_contract(ROOT / "data/raw/Phase_3_RR", config["download_manifest"]))
        write_json(output / "submission_validation.json", validation)
        write_json(output / "candidate_completed.json", {"candidate_local_complete": True,
            "dev_macro_f1": metrics["macro_f1"], "sentence_only_dev_macro_f1": no_context["macro_f1"],
            "zip": str(archive), "zip_sha256": validation["zip_sha256"], "selected_epoch": selected["best_epoch"],
            "independent_final_audit_still_required": True, "uploaded": False})
    dist.barrier()
