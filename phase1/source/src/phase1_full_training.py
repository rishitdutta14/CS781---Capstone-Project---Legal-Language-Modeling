"""Token-weighted full-pass batching, optimizer steps and recoverable checkpoints."""

import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.phase1_full_corpus_prepare import atomic_json, file_sha256
from src.phase1_full_pass import collate


def sample_features(dataset):
    """All samples, aligned with dataset global IDs. No source selection."""
    lengths, starts, docs, sources, eos = [], [], [], [], []
    document_base = 0
    source_names = sorted({p["source"] for p in dataset.parts})
    for part in dataset.parts:
        sample = np.load(part["sample_file"]["path"], mmap_mode="r")
        document = np.load(part["document_file"]["path"], mmap_mode="r")
        lengths.append(np.asarray(sample[:, 2], dtype=np.int32))
        starts.append(np.asarray(sample[:, 1] > 0, dtype=np.int8))
        docs.append(np.asarray(sample[:, 0] + document_base, dtype=np.int32))
        sources.append(np.full(len(sample), source_names.index(part["source"]), dtype=np.int8))
        eos.append(np.asarray(sample[:, 1] + sample[:, 2] == document[sample[:, 0], 1], dtype=np.int8))
        document_base += len(document)
    return {"lengths": np.concatenate(lengths), "has_prefix": np.concatenate(starts),
            "document": np.concatenate(docs), "source": np.concatenate(sources),
            "has_eos": np.concatenate(eos), "source_names": source_names,
            "document_count": document_base}


def batch_order(features, global_batch_size, seed):
    """Shuffle within length/prefix buckets and shuffle batches, retaining tails.

    Bucketing is a scheduling optimization, not a mixing ratio or data filter.
    Each global ID appears exactly once; rank partitioning is done later.
    """
    if global_batch_size < 1:
        raise ValueError("Global batch size must be positive")
    lengths, prefix = features["lengths"], features["has_prefix"]
    if len(lengths) != len(prefix) or np.any(lengths < 1) or np.any(lengths > 800):
        raise ValueError("Invalid complete-pass feature table")
    keys = prefix.astype(np.int32) * 16 + (lengths - 1) // 64
    rng = np.random.default_rng(seed)
    ordered, blocks, offset = [], [], 0
    for key in np.unique(keys):
        ids = np.flatnonzero(keys == key).astype(np.int32)
        rng.shuffle(ids)
        ordered.append(ids)
        blocks.extend((offset + start, min(global_batch_size, len(ids) - start))
                      for start in range(0, len(ids), global_batch_size))
        offset += len(ids)
    order = np.concatenate(ordered)
    blocks = np.asarray(blocks, dtype=np.int64)
    rng.shuffle(blocks)
    if not np.array_equal(np.sort(order), np.arange(len(lengths), dtype=np.int32)) or int(blocks[:, 1].sum()) != len(lengths):
        raise ValueError("Complete-pass permutation lost or duplicated samples")
    return order, blocks


def local_microbatches(ids, rank, world_size, microbatch_size):
    if not 0 <= rank < world_size or microbatch_size < 1:
        raise ValueError("Invalid distributed batch layout")
    stride = world_size * microbatch_size
    for offset in range(0, len(ids), stride):
        start = offset + rank * microbatch_size
        yield ids[start:min(start + microbatch_size, offset + stride)]


def move_batch(examples, device):
    source, decoder, labels, sm, tm = collate(examples)
    # Avoid a dense causal mask when a bucket has no padding, enabling the
    # equivalent optimized is_causal path. All-valid is checked on CPU.
    sm = None if bool(sm.all()) else sm
    tm = None if bool(tm.all()) else tm
    return (source.to(device), decoder.to(device), labels.to(device),
            None if sm is None else sm.to(device), None if tm is None else tm.to(device))


def summed_loss(model, batch, precision=None):
    source, decoder, labels, sm, tm = batch
    autocast = (torch.autocast(source.device.type, dtype=precision)
                if precision is not None else contextlib.nullcontext())
    with autocast:
        logits = model(source, decoder, sm, tm)
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                               ignore_index=-100, reduction="sum")


def optimizer_step(model, optimizer, dataset, ids, features, device,
                   rank=0, world_size=1, microbatch_size=4, precision=None, clip=1.0):
    denominator = int(features["lengths"][ids].sum())
    if denominator < 1:
        raise ValueError("Global batch has no supervised targets")
    optimizer.zero_grad(set_to_none=True)
    batches = list(local_microbatches(ids, rank, world_size, microbatch_size))
    numerator = torch.zeros((), device=device, dtype=torch.float64)
    actual_local_targets = 0
    data_seconds = 0.0
    for number, local_ids in enumerate(batches):
        data_started = time.perf_counter()
        examples = [dataset[int(index)] for index in local_ids]
        batch = move_batch(examples, device)
        data_seconds += time.perf_counter() - data_started
        actual_local_targets += int((batch[2] != -100).sum())
        sync = (model.no_sync() if world_size > 1 and number + 1 < len(batches)
                else contextlib.nullcontext())
        with sync:
            loss = summed_loss(model, batch, precision)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            # DDP averages gradients across ranks. This scaling yields SUM of
            # all valid-token gradients / total valid tokens, including tails.
            (loss * world_size / denominator).backward()
            numerator += loss.detach().double()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    actual = torch.tensor(actual_local_targets, device=device, dtype=torch.int64)
    if world_size > 1:
        dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
        dist.all_reduce(actual, op=dist.ReduceOp.SUM)
    if int(actual) != denominator:
        raise ValueError("Actual distributed labels disagree with target plan")
    optimizer.step()
    return {"loss": float(numerator / denominator), "valid_targets": denominator,
            "gradient_norm": float(norm), "samples": len(ids),
            "local_data_preparation_and_transfer_seconds": data_seconds}


def state_digest(value):
    """Hash every tensor and scalar in a nested optimizer/model state."""
    digest = hashlib.sha256()
    def walk(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                walk(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(len(item)).encode())
            for child in item:
                walk(child)
        else:
            digest.update(repr(item).encode())
    walk(value)
    return digest.hexdigest()


def initialize_weights(module):
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        torch.nn.init.normal_(module.weight, std=0.02)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


def learning_rate(targets, total_targets, peak=3e-4, floor=3e-5, warmup_fraction=0.01):
    warmup = max(1, int(total_targets * warmup_fraction))
    if targets <= warmup:
        return peak * max(1, targets) / warmup
    progress = min(1.0, (targets - warmup) / max(1, total_targets - warmup))
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def capture_rng(device=None):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device is not None else None}


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer, progress, bindings, rank_rng_states,
                    scheduler_state=None, scaler_state=None, extra_state=None):
    path = Path(path)
    if path.exists() or path.with_suffix(".metadata.json").exists():
        raise FileExistsError("Preserve historical checkpoints")
    temporary = path.with_suffix(f".partial.{os.getpid()}.pt")
    with temporary.open("wb") as handle:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "progress": progress, "bindings": bindings, "rank_rng_states": rank_rng_states,
                    "scheduler": scheduler_state, "scaler": scaler_state,
                    "extra_state": extra_state}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    metadata = {"path": str(path.resolve()), "sha256": file_sha256(path),
                "bytes": path.stat().st_size, "progress": progress, "bindings": bindings,
                "rank_count": len(rank_rng_states), "atomic_write_and_file_fsync": True}
    atomic_json(path.with_suffix(".metadata.json"), metadata)
    return metadata


def load_checkpoint(path, model, optimizer, expected_bindings, rank=0, device=None):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    if metadata["bindings"] != expected_bindings or file_sha256(path) != metadata["sha256"]:
        raise ValueError("Checkpoint identity/hash/config mismatch")
    # Only trusted, locally generated, hash-bound training files are accepted.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["bindings"] != expected_bindings or state["progress"] != metadata["progress"]:
        raise ValueError("Checkpoint internal metadata mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rank_rng_states"][rank], device)
    return state
