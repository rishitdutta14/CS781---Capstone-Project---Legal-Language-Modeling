#!/usr/bin/env python3
"""Train one of the four Phase 1 attention variants."""

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase1_common import (
    MODEL_SPECS,
    OUTPUT_ROOT,
    SHARD_DIR,
    SOURCES,
    atomic_json,
    build_model,
    config_record,
    parameter_count,
)


class LegalTokenShards:
    def __init__(self, root=SHARD_DIR):
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run build_phase1_shards.py first: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("document_overlap") != 0:
            raise ValueError("shard manifest does not prove a disjoint split")
        if self.manifest.get("test_data_used") is not False:
            raise ValueError("shard manifest does not exclude test data")
        self.arrays = {
            split: {
                source: np.memmap(
                    root / f"{split}_{source}.bin", dtype=np.uint16, mode="r"
                )
                for source in SOURCES
            }
            for split in ("train", "val")
        }

    def batch(self, split, batch_size, seq_len, generator, offset=0):
        if batch_size % len(SOURCES):
            raise ValueError("batch size must be divisible by the number of sources")
        rows = []
        for row in range(batch_size):
            source = SOURCES[(offset + row) % len(SOURCES)]
            values = self.arrays[split][source]
            high = len(values) - seq_len - 1
            start = int(torch.randint(high, (1,), generator=generator).item())
            rows.append(np.array(values[start : start + seq_len + 1], dtype=np.int64))
        tokens = torch.from_numpy(np.stack(rows))
        return tokens[:, :-1], tokens[:, 1:]


def initialize_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


def learning_rate(step, total_steps, maximum, minimum, warmup):
    if step < warmup:
        return maximum * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def validation_loss(model, corpus, cfg, device):
    model.eval()
    generator = torch.Generator().manual_seed(cfg["seed"] + 10_000)
    losses = []
    for batch_index in range(cfg["eval_batches"]):
        x, y = corpus.batch(
            "val",
            cfg["eval_batch_size"],
            cfg["train_seq_len"],
            generator,
            batch_index * cfg["eval_batch_size"],
        )
        x, y = x.to(device), y.to(device)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses)


def train(model_name, run_dir=None, max_steps=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full Phase 1 training")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each service must see exactly one allocated GPU")

    run_dir = Path(run_dir or OUTPUT_ROOT / model_name)
    checkpoint_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)
    done_path = run_dir / "TRAINING_COMPLETED.json"
    if done_path.is_file():
        print(f"Training already complete: {done_path}", flush=True)
        return json.loads(done_path.read_text(encoding="utf-8"))

    cfg = config_record(model_name)
    tokens_per_step = (
        cfg["micro_batch_size"]
        * cfg["train_seq_len"]
        * cfg["gradient_accumulation_steps"]
    )
    total_steps = math.ceil(cfg["train_tokens"] / tokens_per_step)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        total_steps = max_steps
        cfg["train_tokens"] = max_steps * tokens_per_step
        cfg["eval_every"] = max_steps
        cfg["eval_batches"] = 2
    cfg["tokens_per_optimizer_step"] = tokens_per_step
    cfg["total_steps"] = total_steps
    device = torch.device("cuda")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_model(model_name).to(device)
    model.apply(initialize_weights)
    count = parameter_count(model)
    if not 50_000_000 <= count <= 300_000_000:
        raise ValueError(f"parameter count outside allowed range: {count:,}")
    cfg["parameter_count"] = count
    cfg["gpu"] = torch.cuda.get_device_name(0)
    atomic_json(run_dir / "config.json", cfg)

    corpus = LegalTokenShards()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=cfg["weight_decay"],
        fused=True,
    )
    data_generator = torch.Generator().manual_seed(cfg["seed"] + 1)
    latest_path = checkpoint_dir / "latest.pt"
    metrics_path = run_dir / "metrics.json"
    metrics = {
        "model": model_name,
        "parameter_count": count,
        "train": [],
        "validation": [],
        "best_validation_loss": None,
        "best_step": None,
    }
    start_step = 0
    best_loss = float("inf")
    elapsed_before = 0.0

    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        data_generator.set_state(checkpoint["data_generator_state"])
        start_step = checkpoint["step"]
        best_loss = checkpoint["best_loss"]
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"Resuming from optimizer step {start_step:,}", flush=True)
    else:
        print(
            f"Randomly initialized {model_name}: {count / 1e6:.2f}M parameters",
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    recent_loss = 0.0
    recent_steps = 0
    model.train()

    for step in range(start_step, total_steps):
        lr = learning_rate(
            step,
            total_steps,
            cfg["learning_rate"],
            cfg["min_learning_rate"],
            cfg["warmup_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(cfg["gradient_accumulation_steps"]):
            offset = (
                step * cfg["gradient_accumulation_steps"] * cfg["micro_batch_size"]
                + micro_step * cfg["micro_batch_size"]
            )
            x, y = corpus.batch(
                "train",
                cfg["micro_batch_size"],
                cfg["train_seq_len"],
                data_generator,
                offset,
            )
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                loss = loss / cfg["gradient_accumulation_steps"]
            loss.backward()
            step_loss += float(loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["gradient_clip"]
        )
        optimizer.step()
        recent_loss += step_loss
        recent_steps += 1
        completed_step = step + 1

        if completed_step % 50 == 0 or completed_step == total_steps:
            elapsed = elapsed_before + time.time() - started
            tokens_seen = completed_step * tokens_per_step
            throughput = max(0, tokens_seen - start_step * tokens_per_step) / max(
                1e-6, time.time() - started
            )
            row = {
                "step": completed_step,
                "tokens": tokens_seen,
                "loss": recent_loss / recent_steps,
                "learning_rate": lr,
                "gradient_norm": float(grad_norm),
                "elapsed_seconds": elapsed,
                "tokens_per_second_since_resume": throughput,
            }
            metrics["train"].append(row)
            recent_loss = 0.0
            recent_steps = 0
            atomic_json(metrics_path, metrics)
            print(
                f"step {completed_step:,}/{total_steps:,} "
                f"loss {row['loss']:.4f} lr {lr:.2e} "
                f"{throughput:,.0f} tok/s",
                flush=True,
            )

        evaluate = completed_step % cfg["eval_every"] == 0 or completed_step == total_steps
        if evaluate:
            val_loss = validation_loss(model, corpus, cfg, device)
            val_row = {
                "step": completed_step,
                "tokens": completed_step * tokens_per_step,
                "loss": val_loss,
                "perplexity": math.exp(min(val_loss, 20)),
            }
            metrics["validation"].append(val_row)
            if val_loss < best_loss:
                best_loss = val_loss
                metrics["best_validation_loss"] = val_loss
                metrics["best_step"] = completed_step
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "step": completed_step,
                        "validation_loss": val_loss,
                        "config": cfg,
                    },
                    best_dir / "model.pt",
                )

            elapsed = elapsed_before + time.time() - started
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": completed_step,
                "best_loss": best_loss,
                "data_generator_state": data_generator.get_state(),
                "elapsed_seconds": elapsed,
                "config": cfg,
            }
            atomic_torch_save(checkpoint, latest_path)
            atomic_json(metrics_path, metrics)
            print(
                f"validation step {completed_step:,}: loss {val_loss:.4f}, "
                f"ppl {val_row['perplexity']:.2f}",
                flush=True,
            )

    elapsed = elapsed_before + time.time() - started
    result = {
        "model": model_name,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "steps": total_steps,
        "training_tokens": total_steps * tokens_per_step,
        "best_step": metrics["best_step"],
        "best_validation_loss": metrics["best_validation_loss"],
        "elapsed_seconds": elapsed,
        "average_tokens_per_second": total_steps * tokens_per_step / elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint": str((best_dir / "model.pt").resolve()),
    }
    atomic_json(done_path, result)
    print(f"Training complete: {done_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    train(args.model, args.run_dir, args.max_steps)


if __name__ == "__main__":
    main()
