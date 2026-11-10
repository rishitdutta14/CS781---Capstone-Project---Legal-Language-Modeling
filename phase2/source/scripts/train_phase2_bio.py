#!/usr/bin/env python3
"""Two-GPU BIO baseline; pilot and full training are separate processes."""

import argparse
import hashlib
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase1_v2_models import V2Config, build_model
from src.phase2_bio import (DenseBIO, EncoderDecoderBIO, bio_tags, decode_tags, preferred_window,
                           strict_scores, supervision)
from src.phase2_stopping import EarlyStopping


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    if shutil.disk_usage(path.parent).free < 4 * 2**30:
        raise RuntimeError("less than 4 GiB free at the checkpoint destination")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_data(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    preflight = json.loads((directory / "preflight.json").read_text())
    if preflight["status"] != "passed" or not preflight["gold_window_roundtrip_exact"]:
        raise ValueError("data preflight did not pass")
    config = manifest["config"]
    if (config["context_limit"], config["overlap_tokens"], config["added_special_tokens"]) != (1600, 128, []):
        raise ValueError("data settings differ from the approved baseline")
    for path, digest in manifest["inputs_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"input hash changed: {path}")
    tags = bio_tags(manifest["labels"])
    documents, rows, counts = {}, [], {"supervised_tokens": 0, "positive_tokens": 0}
    for split in ("train", "validation"):
        documents[split] = []
        for key in manifest["document_ids"][split]:
            path = directory / "documents" / split / f"{key}.json"
            doc = json.loads(path.read_text())
            if doc["document_id"] != key or doc["split"] != split:
                raise ValueError("prepared document ID/split mismatch")
            targets, _ = supervision(doc, tags)
            documents[split].append(doc)
            if split == "train":
                for w, labels in zip(doc["windows"], targets):
                    a, b = w["token_start"], w["token_end"]
                    n = sum(t != -100 for t in labels)
                    if n == 0:
                        raise ValueError("window has no supervised real tokens")
                    rows.append({"document_id": key, "window_number": w["window_number"],
                                 "ids": doc["token_ids"][a:b], "targets": labels, "count": n})
                    counts["supervised_tokens"] += n
                    counts["positive_tokens"] += sum(t > 0 for t in labels)
    if len(rows) % 2:
        raise ValueError("two-rank exact-once training currently requires an even window count")
    return manifest, tags, documents, rows, counts


@torch.no_grad()
def infer_document(model, doc, tags):
    offsets = doc["token_offsets"]
    if not offsets:
        return []
    owners = [preferred_window(doc, t) for t in range(len(offsets))]
    predictions = [None] * len(offsets)
    for number, window in enumerate(doc["windows"]):
        a, b = window["token_start"], window["token_end"]
        ids = torch.tensor([doc["token_ids"][a:b]], device="cuda", dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            choices = model(ids).argmax(-1)[0].cpu().tolist()
        for local, tag in enumerate(choices):
            if owners[a+local] == number:
                predictions[a+local] = tag
    if any(t is None for t in predictions):
        raise ValueError("document prediction omitted token positions")
    return decode_tags(predictions, offsets, tags, len(doc["text"]))


def evaluate(model, docs, tags, labels):
    model.eval()
    started = time.monotonic()
    predictions = {doc["document_id"]: infer_document(model, doc, tags) for doc in docs}
    gold = {doc["document_id"]: [[e["start"], e["end"], e["label"]] for e in doc["entities"]] for doc in docs}
    scores = strict_scores(gold, predictions, labels)
    scores["elapsed_seconds"] = time.monotonic()-started
    return scores, predictions, gold


def train_group(ddp, optimizer, group, rows, local_rank, world, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(rows[i]["count"] for i in group)
    local = group[local_rank::world]
    numerator = torch.zeros((), device="cuda", dtype=torch.float64)
    ddp.train()
    for number, index in enumerate(local):
        row = rows[index]
        ids = torch.tensor([row["ids"]], device="cuda", dtype=torch.long)
        target = torch.tensor([row["targets"]], device="cuda", dtype=torch.long)
        sync = nullcontext() if number == len(local)-1 else ddp.no_sync()
        with sync:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = ddp(ids)
                loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target.reshape(-1),
                                           ignore_index=-100, reduction="sum")
                # DDP averages gradients, so scale to the true global token mean.
                loss = loss_sum * world / denominator
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite BIO training loss")
            loss.backward()
        numerator += loss_sum.detach().double()
    norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    dist.all_reduce(numerator)
    return float(numerator)/denominator, float(norm)


def setup_model(checkpoint_path, tags, architecture="dense"):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved["architecture"] != architecture:
        raise ValueError("checkpoint architecture differs from the requested baseline")
    cfg = V2Config(**saved["model_config"])
    if cfg.max_seq_len != 1600:
        raise ValueError("checkpoint context differs from approved 1600")
    backbone = build_model(cfg)
    backbone.load_state_dict(saved["model"], strict=True)
    if any(not torch.equal(tensor.cpu(), saved["model"][name]) for name, tensor in backbone.state_dict().items()):
        raise ValueError("Phase 1 checkpoint loading changed weights")
    if architecture == "dense":
        model = DenseBIO(backbone, len(tags), checkpoint_blocks=True)
    elif architecture == "encoder_decoder":
        model = EncoderDecoderBIO(backbone, len(tags), checkpoint_blocks=True)
    else:
        raise ValueError("unsupported BIO backbone")
    del saved, backbone
    parameters = sum(p.numel() for p in model.parameters())
    if not 50_000_000 <= parameters <= 300_000_000:
        raise ValueError(f"parameter count outside required range: {parameters}")
    return model.cuda(), cfg, parameters


def main(args):
    if args.epochs < 1 or args.patience < 0:
        raise ValueError("epochs must be positive and patience nonnegative")
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("this run must see exactly GPUs 6 and 7 via two-rank torchrun")
    torch.cuda.set_device(rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("baseline requires BF16 support")
    torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/torch.cuda.get_device_properties(rank).total_memory, rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 5*2**30:
        raise RuntimeError("checkpoint destination needs at least 5 GiB free")
    manifest, tags, docs, rows, supervision_counts = load_data(args.data_dir)
    gold_projection = {}
    for split, documents in docs.items():
        projected, truth = {}, {}
        for document in documents:
            _, target = supervision(document, tags)
            key = document["document_id"]
            projected[key] = decode_tags(target, document["token_offsets"], tags, len(document["text"]))
            truth[key] = [[e["start"], e["end"], e["label"]] for e in document["entities"]]
        gold_projection[split] = strict_scores(truth, projected, manifest["labels"])
    model, cfg, parameters = setup_model(args.checkpoint, tags, args.architecture)
    ddp = DDP(model, device_ids=[rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=0.01)
    effective_batch = 2*args.accumulation
    steps_per_epoch = math.ceil(len(rows)/effective_batch)
    config = {"architecture": args.architecture, "adaptation": "full_fine_tuning", "output_head": "plain_BIO_25_tags",
              "backbone_attention": ("unchanged_A3_causal_MHA" if args.architecture == "dense" else
                                     "unchanged_bidirectional_encoder_MHA_and_causal_decoder_MHA_with_full_encoder_cross_attention"),
              "model_input": ("window text token IDs" if args.architecture == "dense" else
                              "same window text token IDs to both encoder and decoder, without shifting; no gold tags as inputs"),
              "context_policy": "same 1600-token text window; each encoder/decoder stream individually within its original limit",
              "all_backbone_stacks_active": True, "parameter_count": parameters,
              "phase1_checkpoint": str(args.checkpoint.resolve()), "phase1_checkpoint_sha256": sha(args.checkpoint),
              "phase1_model_config": vars(cfg), "backbone_loaded_exactly": True,
              "tags": tags, "data_dir": str(args.data_dir.resolve()), "data_manifest_sha256": sha(args.data_dir / "manifest.json"),
              "data_preflight_sha256": sha(args.data_dir / "preflight.json"), "tokenizer_sha256": manifest["config"]["tokenizer_sha256"],
              "split_ids": manifest["document_ids"], "training_windows": len(rows), "supervision_counts": supervision_counts,
              "seed": args.seed, "epochs": args.epochs, "learning_rate": args.lr, "lr_schedule": "constant",
              "initialization": "fresh Phase 2 run from original Phase 1 weights; new BIO classifier and optimizer; no Phase 2 resume",
              "early_stopping": {"patience_epochs": args.patience, "metric": "validation strict macro-F1",
                                 "minimum_delta": 0.0, "comparison": "strictly greater", "enabled": args.patience > 0},
              "optimizer": "AdamW(default betas, eps; weight_decay=0.01)", "gradient_clip_norm": 1.0,
              "precision": "bfloat16", "activation_checkpointing": True, "world_size": world,
              "microbatch_per_gpu": 1, "gradient_accumulation_per_gpu": args.accumulation,
              "effective_batch_windows": effective_batch, "optimizer_steps_per_epoch": steps_per_epoch,
              
              "allocator_limit_gib_per_gpu": args.memory_gib, "checkpoint_destination": str(output),
              "loss": "unweighted token-mean BIO cross-entropy; no LM objective",
              "training_ownership": "one containing window per entity; max minimum edge distance at entity midpoint; negatives per-token max minimum edge distance; earlier wins ties",
              "ignored_supervision": "non-owning appearances, cut-off entities, zero-width synthetic tokens; no padding with microbatch 1",
              "inference_ownership": "one window per token chosen by max minimum edge distance; earlier wins ties; no gold information",
              "bio_repair": "orphan/different-type I starts new entity; zero-width offsets skipped; exact duplicate spans removed",
              "boundary_refinement": False, "rule_based_boundary_correction": False,
              "known_boundary_limitation": "2572/8393 gold entities do not align with whole-token boundaries; exact spans not rounded or repaired",
              "gold_BIO_projection_scores_NOT_model_performance": gold_projection,
              "checkpoint_selection": "best validation strict macro-F1 across all 10 validation documents",
              "test_labels_used": False, "model_code_llm_permission": "User explicitly confirmed permission covers Phase 2",
              "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (
                  Path(__file__), ROOT / "src/phase2_bio.py", ROOT / "src/phase2_data.py",
                  ROOT / "src/phase2_stopping.py",
                  ROOT / "src/phase1_v2_models.py", ROOT / "assignment3_impl/q2/model_arch.py",
                  ROOT / "assignment3_impl/q2/attention.py")},
              "software": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}}
    if rank == 0:
        if args.mode == "train" and (output / "config.json").exists():
            raise ValueError("existing training run must not be overwritten; use a fresh output directory")
        save_json(output / ("pilot_config.json" if args.mode == "pilot" else "config.json"), config)
        print(json.dumps({"mode": args.mode, "parameters": parameters, "windows": len(rows),
                          "epochs": args.epochs, "steps_per_epoch": steps_per_epoch,
                          "effective_batch": effective_batch}, indent=2), flush=True)
    if args.mode == "pilot":
        # Maximum-length training windows only. Updates are disposable: full
        # training starts in a fresh process from the original Phase 1 weights.
        longest = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]), reverse=True)
        times, losses = [], []
        torch.cuda.reset_peak_memory_stats()
        for step in range(4):
            group = longest[step*effective_batch:(step+1)*effective_batch]
            dist.barrier()
            torch.cuda.synchronize()
            started = time.monotonic()
            loss, norm = train_group(ddp, optimizer, group, rows, rank, world, args.lr)
            torch.cuda.synchronize()
            elapsed = time.monotonic()-started
            times.append(elapsed)
            losses.append(loss)
            if rank == 0:
                print(f"pilot step {step+1}/4: loss={loss:.4f}, seconds={elapsed:.2f}, gradient_norm={norm:.3f}", flush=True)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.float64)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        dist.barrier()
        if rank == 0:
            # One full validation document for an actual inference timing check.
            scores, predictions, gold = evaluate(model, docs["validation"][:1], tags, manifest["labels"])
            seconds = sum(times[1:])/len(times[1:])
            val_windows = sum(len(d["windows"]) for d in docs["validation"])
            sampled_windows = len(docs["validation"][0]["windows"])
            val_estimate = scores["elapsed_seconds"] * val_windows / sampled_windows
            projected = args.epochs * (steps_per_epoch * seconds + val_estimate)
            save_json(output / "pilot.json", {"passed": True, "step_seconds": times, "losses": losses,
                                              "mean_steady_step_seconds": seconds, "peak_allocated_gib": float(peak)/2**30,
                                              "validation_sample_windows": sampled_windows,
                                              "validation_sample_seconds": scores["elapsed_seconds"],
                                              "estimated_full_training_and_epoch_validation_seconds": projected,
                                              "estimate_caveat": "Maximum-length pilot on shared hardware; excludes startup, checkpoint I/O and final test inference; estimate not a guarantee.",
                                              "pilot_updates_discarded": True, "full_training_will_reload_phase1_checkpoint": True})
            print(f"Pilot passed. Estimated {args.epochs}-epoch training plus validation: {projected/60:.1f} minutes", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return
    if not (output / "pilot.json").exists() or not json.loads((output / "pilot.json").read_text())["passed"]:
        raise ValueError("successful speed/memory pilot is required")
    history, best, optimizer_steps = [], -1.0, 0
    stopping = EarlyStopping(args.patience)
    stopped_early = False
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        order = list(range(len(rows)))
        random.Random(args.seed+epoch).shuffle(order)
        epoch_started = time.monotonic()
        losses = []
        for start in range(0, len(order), effective_batch):
            group = order[start:start+effective_batch]
            loss, norm = train_group(ddp, optimizer, group, rows, rank, world, args.lr)
            optimizer_steps += 1
            losses.append(loss)
            if rank == 0 and (optimizer_steps % 10 == 0 or start == 0):
                save_json(output / "status.json", {"status": "training", "epoch": epoch+1,
                          "epochs": args.epochs, "optimizer_steps": optimizer_steps,
                          "windows_completed_this_epoch": min(start+effective_batch, len(rows)),
                          "loss": loss, "elapsed_seconds": time.monotonic()-started})
                print(f"epoch {epoch+1}/{args.epochs} step {optimizer_steps}: loss={loss:.4f}", flush=True)
        dist.barrier()
        if rank == 0:
            scores, predictions, gold = evaluate(model, docs["validation"], tags, manifest["labels"])
            value = scores["strict_macro_f1"]
            improved, should_stop = stopping.update(value)
            row = {"epoch": epoch+1, "mean_training_group_loss": sum(losses)/len(losses),
                   "epoch_seconds": time.monotonic()-epoch_started, "validation": scores,
                   "selected_as_best": improved, "stale_epochs": stopping.stale_epochs}
            history.append(row)
            if improved:
                best = value
                best_epoch = epoch+1
                save_checkpoint(output / "best_model.pt", {"model": model.state_dict(), "config": config,
                                                          "epoch": epoch+1, "validation_strict_macro_f1": best})
                save_json(output / "best_validation_prediction.json", predictions)
            save_checkpoint(output / "latest_checkpoint.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                                              "config": config, "completed_epochs": epoch+1,
                                                              "optimizer_steps": optimizer_steps, "best_strict_macro_f1": best})
            save_json(output / "metrics.json", {"epochs": history, "best_validation_strict_macro_f1": best})
            save_json(output / "validation_truth.json", gold)
            print(f"epoch {epoch+1} validation strict macro-F1={value:.6f}; best={best:.6f}", flush=True)
        stop_flag = torch.tensor(int(should_stop) if rank == 0 else 0, device="cuda")
        dist.broadcast(stop_flag, src=0)
        dist.barrier()
        if bool(stop_flag.item()):
            stopped_early = True
            if rank == 0:
                print(f"Early stopping after epoch {epoch+1}: {args.patience} consecutive epochs without improvement.", flush=True)
            break
    if rank == 0:
        state = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        del state
        predictions = {}
        for key in manifest["document_ids"]["test"]:
            path = args.data_dir / "documents/test" / f"{key}.json"
            doc = json.loads(path.read_text())
            predictions[key] = infer_document(model, doc, tags)
        if set(predictions) != set(manifest["document_ids"]["test"]):
            raise ValueError("test prediction IDs differ")
        prediction_path = output / "prediction.json"
        save_json(prediction_path, predictions)
        with zipfile.ZipFile(output / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(prediction_path, arcname="prediction.json")
        result = subprocess.run([sys.executable, str(ROOT / "official/evaluation_scripts/evaluate_phase2.py"),
                                 "--truth", str(output / "validation_truth.json"),
                                 "--prediction", str(output / "best_validation_prediction.json"),
                                 "--labels", str(ROOT / "data/raw/Phase_2_LNER/Train/labels.txt")],
                                capture_output=True, text=True, check=True)
        (output / "official_validation_evaluator.txt").write_text(result.stdout)
        if f"Strict Macro-F1:        {best:.6f}" not in result.stdout:
            raise ValueError("internal and official validation scoring differ")
        if sha(args.checkpoint) != config["phase1_checkpoint_sha256"]:
            raise ValueError("source checkpoint changed")
        completed = {"status": "completed", "epochs": len(history), "maximum_epochs": args.epochs,
                     "early_stopped": stopped_early, "best_epoch": best_epoch, "optimizer_steps": optimizer_steps,
                     "elapsed_seconds": time.monotonic()-started, "best_validation_strict_macro_f1": best,
                     "test_ids": len(predictions), "submission_zip": str(output / "submission.zip"),
                     "plain_whole_token_BIO": True, "source_checkpoint_unchanged": True,
                     "submission_uploaded": False}
        save_json(output / "COMPLETED.json", completed)
        save_json(output / "status.json", completed)
        print(json.dumps(completed, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "train"), required=True)
    parser.add_argument("--architecture", choices=("dense", "encoder_decoder"), default="dense")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/phase2_preflight_20261011_r2")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/phase1_v2/dense/best_model.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--memory-gib", type=float, default=20)
    parser.add_argument("--seed", type=int, default=781)
    main(parser.parse_args())
