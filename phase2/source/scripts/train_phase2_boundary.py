#!/usr/bin/env python3
"""Two-GPU BIO plus learned exact-boundary NER, without changing attention."""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_phase2_bio import save_checkpoint, save_json
from src.phase2_experiment import BACKBONES, file_sha, set_loss_normalizers, synchronize_fresh_config
from src.phase2_boundary_experiment import (evaluate_boundary_model as evaluate_model,
    infer_boundary_document, load_boundary_backbone as load_backbone,
    load_boundary_documents as load_documents)
from src.phase2_boundaries import boundary_loss_sum
from src.phase2_objectives import rank_group, token_loss_sum
from src.phase2_stopping import EarlyStopping
from src.phase2_submission import validate_archive
from src.phase2_telemetry import local_memory, training_memory_by_rank
from src.phase2_adaptation import METHODS, apply_adaptation, frozen_parameter_sha256
from src.phase2_distillation import attach_training_cache, distillation_loss_sum



def step_lr(step, total, maximum, schedule, warmup):
    if step < warmup:
        return maximum * (step + 1) / warmup
    if schedule == "constant":
        return maximum
    progress = min(1, (step - warmup) / max(1, total - warmup - 1))
    return maximum * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)


def train_group(ddp, optimizer, group, rows, rank, args, weights, verify_gradients=False):
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(rows[i]["normalizer"] for i in group)
    boundary_denominator = sum(len(rows[i]["boundary_targets"]) for i in group)
    kd_weight = getattr(args, "kd_weight", 0.)
    use_kd = kd_weight > 0
    kd_denominator = sum(rows[i]["kd_normalizer"] for i in group) if use_kd else 0
    local = rank_group(group, rank, 2)
    numerator = torch.zeros(3 if use_kd else 2, device="cuda", dtype=torch.float64)
    ddp.train()
    for number, index in enumerate(local):
        row = rows[group[0] if index is None else index]
        ids = torch.tensor([row["ids"]], device="cuda")
        target = torch.tensor([row["targets"]], device="cuda")
        if index is None:
            target.fill_(-100)
        sync = nullcontext() if number == len(local)-1 else ddp.no_sync()
        with sync:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = ddp(ids, row["candidate_table"])
                loss_sum, _ = token_loss_sum(outputs["bio"], target, args.loss, weights, args.focal_gamma)
                boundary_sum, _ = boundary_loss_sum(outputs["boundaries"], row["candidate_table"],
                                                    [] if index is None else row["boundary_targets"])
                loss = 2 * (loss_sum / denominator + args.boundary_weight * boundary_sum / max(1, boundary_denominator))
                if use_kd:
                    kd_mask = row["kd_mask"].clone()
                    if index is None:
                        kd_mask.fill_(False)
                    kd_sum, _ = distillation_loss_sum(outputs["bio"][0], row["kd_probabilities"],
                                                      kd_mask, args.kd_temperature)
                    loss = loss + 2 * kd_weight * kd_sum / max(1, kd_denominator)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
        numerator[0] += loss_sum.detach().double()
        numerator[1] += boundary_sum.detach().double()
        if use_kd:
            numerator[2] += kd_sum.detach().double()
    if verify_gradients:
        missing = [name for name, p in ddp.module.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise ValueError(f"intended trainable parameters missing gradients: {missing}")
        unexpected = [name for name, p in ddp.module.named_parameters() if not p.requires_grad and p.grad is not None]
        if unexpected:
            raise ValueError(f"frozen parameters received gradients: {unexpected}")
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    dist.all_reduce(numerator)
    losses = (float(numerator[0]) / denominator, float(numerator[1]) / max(1, boundary_denominator))
    if use_kd:
        losses += (float(numerator[2]) / max(1, kd_denominator),)
    return losses, float(norm)


def should_pause_after_epoch(completed_epochs, stopped, maximum_epochs, pause_after_epochs):
    return bool(pause_after_epochs and completed_epochs >= pause_after_epochs
                and completed_epochs < maximum_epochs and not stopped)


def main(args):
    if not 1 <= args.epochs <= 60 or args.patience < 0 or args.accumulation < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("invalid campaign limits/settings")
    if not math.isfinite(args.boundary_weight) or args.boundary_weight <= 0:
        raise ValueError("boundary weight must be finite and positive")
    method = getattr(args, "adaptation", "full_fine_tuning")
    kd_weight = getattr(args, "kd_weight", 0.)
    teacher_cache = getattr(args, "teacher_cache", None)
    if not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError("KD weight must be finite and nonnegative")
    if (method == "knowledge_distillation") != (kd_weight > 0 and teacher_cache is not None):
        raise ValueError("KD requires its own adaptation declaration, positive weight, and train-only cache")
    if method != "knowledge_distillation" and (kd_weight or teacher_cache):
        raise ValueError("non-KD experiments must not silently use teacher targets")
    if getattr(args, "pause_after_epochs", 0) < 0:
        raise ValueError("pause epoch must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DDP training run")
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or torch.cuda.device_count() != 2:
        raise ValueError("exactly two DDP ranks are required")
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported")
    torch.cuda.set_per_process_memory_fraction(20 * 2**30 / torch.cuda.get_device_properties(rank).total_memory, rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 5 * 2**30:
        raise RuntimeError("checkpoint destination needs 5 GiB free")
    if (output / "COMPLETED.json").exists():
        raise ValueError("completed experiments must never be overwritten")
    manifest, tags, docs, rows, histogram = load_documents(args.data_dir)
    weights = set_loss_normalizers(rows, histogram, args.loss)
    model, info = load_backbone(args.architecture, len(tags), manifest["config"]["tokenizer_sha256"])
    if method == "lorma_plus" and args.architecture != "dense":
        raise ValueError("planned LoRMA+ mapping is the rebuilt dense decoder")
    adaptation = apply_adaptation(model, method, getattr(args, "lorma_rank", 8),
                                  getattr(args, "lorma_alpha", 8.), getattr(args, "partial_layers", 2))
    kd_info = None
    if method == "knowledge_distillation":
        kd_info = attach_training_cache(teacher_cache, manifest, tags, docs["train"], rows, args.kd_temperature)
        if kd_info["student_parameters"] != adaptation["total_parameters"]:
            raise ValueError("cache student size differs from the actual NER model")
    info["parameters"] = adaptation["total_parameters"]
    if args.dropout is not None:
        if not 0 <= args.dropout < 1:
            raise ValueError("invalid dropout")
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = args.dropout
    ddp = DDP(model, device_ids=[rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    effective_batch = 2 * args.accumulation
    steps_per_epoch = math.ceil(len(rows) / effective_batch)
    code_paths = [Path(__file__), ROOT / "scripts/train_phase2_bio.py",
                  ROOT / "src/phase2_experiment.py", ROOT / "src/phase2_objectives.py",
                  ROOT / "src/phase2_decoding.py", ROOT / "src/phase2_bio.py",
                  ROOT / "src/phase2_data.py", ROOT / "src/phase2_stopping.py",
                  ROOT / "src/phase2_submission.py", ROOT / "src/phase1_v2_models.py",
                  ROOT / "src/phase2_boundaries.py", ROOT / "src/phase2_boundary_experiment.py",
                  ROOT / "src/phase2_telemetry.py",
                  ROOT / "src/phase2_adaptation.py", ROOT / "src/phase2_distillation.py",
                  ROOT / "assignment3_impl/q2/model_arch.py", ROOT / "assignment3_impl/q2/attention.py"]
    config = {**info, "architecture": args.architecture, "output_head": "BIO_plus_learned_character_boundary_candidates",
              "adaptation": method, "adaptation_details": adaptation, "attention_changed": False,
              "knowledge_distillation": kd_info, "kd_weight": kd_weight,
              "kd_objective": "T^2 KL(teacher entity types || student entity types), global mean over owned aligned tokens" if kd_info else None,
              "both_encoder_decoder_stacks_active": args.architecture == "encoder_decoder",
              "seed": args.seed, "maximum_epochs": args.epochs, "patience": args.patience,
              "loss": args.loss, "focal_gamma": args.focal_gamma,
              "boundary_loss": "candidate cross-entropy; global mean across exactly two owned endpoints per gold entity",
              "boundary_weight": args.boundary_weight,
              "boundary_candidate_cap": None,
              "boundary_features": "Raw UTF-8 bytes around each candidate within its boundary token plus adjacent whitespace; relative positions only, no absolute coordinates or gold labels as model inputs",
              "boundary_parameter_initialization": "random; no external weights",
              "boundary_endpoints_per_epoch": sum(len(row["boundary_targets"]) for row in rows),
              "training_class_counts": dict(zip(tags, histogram.tolist())),
              "class_weights": None if weights is None else dict(zip(tags, weights.tolist())),
              "weight_formula": "min(10,max(1,sqrt(training_O_count/training_tag_count)))" if weights is not None else None,
              "loss_denominator": "global sum of target class weights" if weights is not None else "global real supervised token count",
              "learning_rate": args.lr, "schedule": args.schedule, "warmup_steps": args.warmup_steps,
              "weight_decay": args.weight_decay, "dropout_override": args.dropout,
              "decoding": args.decoding, "tags": tags,
              "data_dir": str(args.data_dir.resolve()), "data_manifest_sha256": file_sha(args.data_dir / "manifest.json"),
              "data_preflight_sha256": file_sha(args.data_dir / "preflight.json"),
              "document_ids": manifest["document_ids"], "training_windows": len(rows),
              "supervised_tokens_per_epoch": int(histogram.sum()),
              "all_training_documents_each_epoch": True, "effective_batch_windows": effective_batch,
              "last_group_policy": "zero-loss synchronization placeholders only, no dropped or duplicated supervision",
              "context_limit": 1600, "overlap": manifest["config"]["overlap_tokens"],
              "precision": "bfloat16",
              "code_sha256": {str(p.relative_to(ROOT)): file_sha(p) for p in code_paths},
              "test_used_for_selection": False, "source_initialization": "original Phase 1 checkpoint; fresh NER head and optimizer",
              "model_code_llm_permission": "User explicitly confirmed permission covers Phase 2",
              "software": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}}
    stopping = EarlyStopping(args.patience)
    history, optimizer_steps, first_epoch, best_epoch, previous_elapsed = [], 0, 0, 0, 0.
    if args.resume:
        if not (output / "latest_checkpoint.pt").exists():
            raise ValueError("no latest checkpoint to resume")
        saved = torch.load(output / "latest_checkpoint.pt", map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("resume configuration/code differs; do not call a different experiment a resume")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        first_epoch, optimizer_steps = saved["completed_epochs"], saved["optimizer_steps"]
        history, best_epoch = saved["history"], saved["best_epoch"]
        stopping.best, stopping.stale_epochs = saved["stopping_best"], saved["stale_epochs"]
        previous_elapsed = saved["elapsed_seconds"]
        rng = saved["rng_by_rank"][rank]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], rank)
        del saved
    synchronize_fresh_config(output / "config.json", args.resume)
    if rank == 0:
        save_json(output / "config.json", config)
        for path in code_paths:
            destination = output / "source_snapshot" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(path, destination)
        print(json.dumps({"architecture": args.architecture, "loss": args.loss, "parameters": info["parameters"],
                          "adaptation": method, "trainable_parameters": adaptation["trainable_parameters"],
                          "training_windows": len(rows), "starting_epoch": first_epoch+1,
                          "maximum_epochs": args.epochs, "decoding": args.decoding}), flush=True)
    started = time.monotonic()
    stopped_early = False
    for epoch in range(first_epoch, args.epochs):
        if args.resume and stopping.patience and stopping.stale_epochs >= stopping.patience:
            stopped_early = True
            break
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        epoch_started = time.monotonic()
        loss_numerator, loss_denominator, endpoint_numerator, endpoint_denominator = 0., 0., 0., 0.
        kd_numerator, kd_normalizer = 0., 0
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(order), effective_batch):
            group = order[start:start+effective_batch]
            lr = step_lr(optimizer_steps, steps_per_epoch*args.epochs, args.lr, args.schedule, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            losses, norm = train_group(ddp, optimizer, group, rows, rank, args, weights, optimizer_steps == 0)
            bio_loss, endpoint_loss = losses[:2]
            loss = bio_loss + args.boundary_weight * endpoint_loss
            if kd_info:
                count = sum(rows[i]["kd_normalizer"] for i in group)
                kd_numerator += losses[2] * count
                kd_normalizer += count
                loss += kd_weight * losses[2]
            optimizer_steps += 1
            normalizer = sum(rows[i]["normalizer"] for i in group)
            loss_numerator += bio_loss * normalizer
            loss_denominator += normalizer
            endpoints = sum(len(rows[i]["boundary_targets"]) for i in group)
            endpoint_numerator += endpoint_loss * endpoints
            endpoint_denominator += endpoints
            if rank == 0 and (optimizer_steps % 10 == 0 or start == 0):
                save_json(output / "status.json", {"status": "training", "epoch": epoch+1,
                          "optimizer_steps": optimizer_steps, "loss": loss, "lr": lr,
                          "windows_completed_this_epoch": min(start+effective_batch, len(order)),
                          "elapsed_seconds": previous_elapsed + time.monotonic()-started})
                print(f"epoch {epoch+1}/{args.epochs} step {optimizer_steps}: loss={loss:.4f} lr={lr:.3g}", flush=True)
        dist.barrier()
        training_memory = training_memory_by_rank()
        stop = False
        if rank == 0:
            scores, predictions, truth = evaluate_model(model, docs["validation"], tags, manifest["labels"], args.decoding)
            improved, stop = stopping.update(scores["strict_macro_f1"])
            if improved:
                best_epoch = epoch+1
                save_checkpoint(output / "best_model.pt", {"model": model.state_dict(), "config": config,
                               "epoch": best_epoch, "validation_strict_macro_f1": stopping.best})
                save_json(output / "best_validation_prediction.json", predictions)
            history.append({"epoch": epoch+1,
                            "training_loss": loss_numerator/loss_denominator + args.boundary_weight*endpoint_numerator/max(1,endpoint_denominator) + kd_weight*kd_numerator/max(1,kd_normalizer),
                            "BIO_loss": loss_numerator/loss_denominator,
                            "endpoint_loss": endpoint_numerator/max(1,endpoint_denominator),
                            "KD_loss": kd_numerator/max(1,kd_normalizer) if kd_info else None,
                            "kd_tokens": kd_normalizer if kd_info else None,
                            "validation": scores, "selected_as_best": improved,
                            "stale_epochs": stopping.stale_epochs, "last_learning_rate": lr,
                            "epoch_seconds": time.monotonic()-epoch_started,
                            "training_memory_by_rank": training_memory,
                            "rank0_peak_including_validation": local_memory()})
            save_json(output / "metrics.json", {"epochs": history, "best_validation_strict_macro_f1": stopping.best})
            save_json(output / "validation_truth.json", truth)
            print(f"epoch {epoch+1} strict macro-F1={scores['strict_macro_f1']:.6f}; best={stopping.best:.6f}", flush=True)
        # Persist each rank's dropout RNG for genuine epoch-boundary resumption.
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(rank)}
        gathered = [None, None]
        dist.all_gather_object(gathered, rng)
        if frozen_parameter_sha256(model) != adaptation["frozen_initial_sha256"]:
            raise ValueError("frozen adaptation parameters changed during training")
        if rank == 0:
            save_checkpoint(output / "latest_checkpoint.pt", {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": config, "completed_epochs": epoch+1,
                "optimizer_steps": optimizer_steps, "history": history, "best_epoch": best_epoch,
                "stopping_best": stopping.best, "stale_epochs": stopping.stale_epochs,
                "rng_by_rank": gathered, "elapsed_seconds": previous_elapsed+time.monotonic()-started})
        flag = torch.tensor(int(stop), device="cuda")
        dist.broadcast(flag, src=0)
        dist.barrier()
        if should_pause_after_epoch(epoch+1, flag.item(), args.epochs, getattr(args, "pause_after_epochs", 0)):
            if rank == 0:
                paused = {"status": "paused_after_pilot_epoch", "completed_epochs": epoch+1,
                    "maximum_epochs": args.epochs, "adaptation": method,
                    "optimizer_steps": optimizer_steps, "epoch_seconds": history[-1]["epoch_seconds"],
                    "training_memory_by_rank": training_memory,
                    "frozen_parameters_unchanged": True,
                    "config_sha256": file_sha(output / "config.json"),
                    "latest_checkpoint_sha256": file_sha(output / "latest_checkpoint.pt"),
                    "resume_required": True, "real_data_attempt_charged": True}
                save_json(output / "PAUSED.json", paused)
                save_json(output / "status.json", paused)
                print(json.dumps(paused), flush=True)
            dist.barrier()
            dist.destroy_process_group()
            return
        if flag.item():
            stopped_early = True
            break
    if rank == 0:
        saved = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        del saved
        model.eval()
        predictions = {}
        for key in manifest["document_ids"]["test"]:
            doc = json.loads((args.data_dir / "documents/test" / f"{key}.json").read_text())
            predictions[key], _ = infer_boundary_document(model, doc, tags, args.decoding)
        save_json(output / "prediction.json", predictions)
        with zipfile.ZipFile(output / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(output / "prediction.json", arcname="prediction.json")
        validation = validate_archive(output / "submission.zip", manifest["labels"])
        result = subprocess.run([sys.executable, str(ROOT / "official/evaluation_scripts/evaluate_phase2.py"),
                    "--truth", str(output / "validation_truth.json"),
                    "--prediction", str(output / "best_validation_prediction.json"),
                    "--labels", str(ROOT / "data/raw/Phase_2_LNER/Train/labels.txt")],
                    capture_output=True, text=True, check=True)
        (output / "official_validation_evaluator.txt").write_text(result.stdout)
        if f"Strict Macro-F1:        {stopping.best:.6f}" not in result.stdout:
            raise ValueError("official validation score differs")
        if file_sha(info["phase1_checkpoint"]) != info["phase1_checkpoint_sha256"]:
            raise ValueError("source Phase 1 checkpoint changed")
        completed = {"status": "completed", "architecture": args.architecture, "loss": args.loss,
                     "adaptation": method, "trainable_parameters": adaptation["trainable_parameters"],
                     "frozen_parameters_unchanged": True,
                     "epochs": len(history), "maximum_epochs": args.epochs, "best_epoch": best_epoch,
                     "early_stopped": stopped_early, "best_validation_strict_macro_f1": stopping.best,
                     "optimizer_steps": optimizer_steps, "submission_validation": validation,
                     "submission_zip": str(output / "submission.zip"), "zip_sha256": file_sha(output / "submission.zip"),
                     "elapsed_seconds": previous_elapsed+time.monotonic()-started,
                     "resumed": args.resume, "source_checkpoint_unchanged": True, "submission_uploaded": False}
        save_json(output / "COMPLETED.json", completed)
        save_json(output / "status.json", completed)
        print(json.dumps(completed, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=BACKBONES, required=True)
    parser.add_argument("--adaptation", choices=METHODS, default="full_fine_tuning")
    parser.add_argument("--lorma-rank", type=int, default=8)
    parser.add_argument("--lorma-alpha", type=float, default=8.)
    parser.add_argument("--partial-layers", type=int, default=2)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--kd-temperature", type=float, default=2.)
    parser.add_argument("--kd-weight", type=float, default=0.)
    parser.add_argument("--pause-after-epochs", type=int, default=0,
                        help="execution-only pilot gate; unchanged-config resume removes this flag")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss", choices=("ce", "weighted_ce", "focal"), default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.)
    parser.add_argument("--decoding", choices=("argmax", "legal_bio"), default="legal_bio")
    parser.add_argument("--boundary-weight", type=float, default=1.)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=781)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            save_json(args.output_dir / "failure.json", {"status": "failed", "error": str(exc),
                       "resume_checkpoint_exists": (args.output_dir / "latest_checkpoint.pt").exists()})
        raise
