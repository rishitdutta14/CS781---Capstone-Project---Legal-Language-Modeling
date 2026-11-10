"""Production coverage and full held-out scoring for the audited single pass."""

import contextlib
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.phase1_full_training import move_batch, local_microbatches


def batch_workload(features, ids, world, microbatch):
    """Exact input/padding slot counts for the validated per-rank collator."""
    counts = dict(source_input_slots=0, decoder_input_slots=0,
                  source_padding_slots=0, decoder_padding_slots=0,
                  dummy_source_BOS_inputs=0, dummy_decoder_BOS_inputs=0,
                  masked_label_slots=0)
    for rank in range(world):
        for local in local_microbatches(ids, rank, world, microbatch):
            if not len(local):
                counts["source_input_slots"] += 1
                counts["decoder_input_slots"] += 1
                counts["dummy_source_BOS_inputs"] += 1
                counts["dummy_decoder_BOS_inputs"] += 1
                counts["masked_label_slots"] += 1
                continue
            source_lengths = np.where(features["has_prefix"][local], 800, 1)
            target_lengths = features["lengths"][local]
            source_slots = len(local) * int(source_lengths.max())
            target_slots = len(local) * int(target_lengths.max())
            counts["source_input_slots"] += source_slots
            counts["decoder_input_slots"] += target_slots
            counts["source_padding_slots"] += source_slots - int(source_lengths.sum())
            counts["decoder_padding_slots"] += target_slots - int(target_lengths.sum())
            counts["masked_label_slots"] += target_slots - int(target_lengths.sum())
    return counts


class Coverage:
    def __init__(self, features, document_lengths):
        self.features = features
        self.document_lengths = np.asarray(document_lengths, dtype=np.int64)
        self.seen = np.zeros(len(features["lengths"]), dtype=np.uint8)
        self.document_targets = np.zeros(len(document_lengths), dtype=np.int64)
        self.document_eos = np.zeros(len(document_lengths), dtype=np.uint8)
        self.source_targets = np.zeros(len(features["source_names"]), dtype=np.int64)
        self.samples, self.targets, self.eos = 0, 0, 0
        self.context_tokens, self.source_bos_inputs = 0, 0
        self.workload = {name: 0 for name in batch_workload(features, np.array([], dtype=int), 1, 1)}

    def check_next(self, ids):
        if len(np.unique(ids)) != len(ids) or np.any(self.seen[ids]):
            raise ValueError("Attempted duplicate supervised sample")

    def commit(self, ids, actual_valid_targets, workload=None):
        self.check_next(ids)
        lengths = self.features["lengths"][ids]
        if int(lengths.sum()) != actual_valid_targets:
            raise ValueError("Committed optimizer labels differ from planned targets")
        self.seen[ids] = 1
        np.add.at(self.document_targets, self.features["document"][ids], lengths)
        np.add.at(self.document_eos, self.features["document"][ids], self.features["has_eos"][ids])
        np.add.at(self.source_targets, self.features["source"][ids], lengths)
        self.samples += len(ids)
        self.targets += actual_valid_targets
        self.eos += int(self.features["has_eos"][ids].sum())
        prefix_count = int(self.features["has_prefix"][ids].sum())
        self.context_tokens += 800 * prefix_count
        self.source_bos_inputs += len(ids) - prefix_count
        if workload is not None:
            for name, value in workload.items():
                self.workload[name] += value

    def summary(self):
        return {"committed_samples": self.samples, "committed_targets": self.targets,
                "committed_content_targets": self.targets - self.eos, "committed_EOS_targets": self.eos,
                "repeated_source_context_positions": self.context_tokens,
                "source_BOS_input_positions": self.source_bos_inputs,
                "decoder_input_positions": self.targets,
                "decoder_BOS_input_positions": self.samples,
                "decoder_previous_target_input_positions": self.targets - self.samples,
                "documents_with_any_committed_target": int(np.count_nonzero(self.document_targets)),
                "source_target_counts": dict(zip(self.features["source_names"], map(int, self.source_targets))),
                "input_workload_including_padding_and_dummy_rows": self.workload.copy()}

    def state(self):
        return {"seen": self.seen, "document_targets": self.document_targets,
                "document_eos": self.document_eos, "source_targets": self.source_targets,
                "summary": self.summary()}

    def restore(self, state):
        for name in ("seen", "document_targets", "document_eos", "source_targets"):
            expected = getattr(self, name)
            value = state[name]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError("Checkpoint coverage dimensions/dtypes changed")
            setattr(self, name, value.copy())
        summary = state["summary"]
        self.samples, self.targets = summary["committed_samples"], summary["committed_targets"]
        self.eos = summary["committed_EOS_targets"]
        self.context_tokens = summary["repeated_source_context_positions"]
        self.source_bos_inputs = summary["source_BOS_input_positions"]
        self.workload = summary["input_workload_including_padding_and_dummy_rows"].copy()
        if (int(self.seen.sum()) != self.samples or int(self.document_targets.sum()) != self.targets
                or int(self.document_eos.sum()) != self.eos or int(self.source_targets.sum()) != self.targets):
            raise ValueError("Checkpoint coverage counters do not reconcile")

    def verify_complete(self, expected):
        if not np.all(self.seen == 1) or not np.array_equal(self.document_targets, self.document_lengths):
            raise ValueError("Training did not cover every sample/document target exactly once")
        if not np.all(self.document_eos == 1) or self.targets != expected["distinct_supervised_targets"]:
            raise ValueError("Terminal EOS/target coverage incomplete")
        if self.samples != expected["samples"] or self.eos != expected["documents"]:
            raise ValueError("Terminal sample/document counts disagree")
        if self.context_tokens != expected["source_context_tokens"] or self.source_bos_inputs != expected["source_BOS_inputs"]:
            raise ValueError("Repeated context/special-input accounting disagrees")
        measured = self.workload["decoder_input_slots"] > 0
        if measured:
            if self.workload["decoder_input_slots"] - self.workload["decoder_padding_slots"] - self.workload["dummy_decoder_BOS_inputs"] != self.targets:
                raise ValueError("Decoder input/padding workload does not reconcile")
            if self.workload["source_input_slots"] - self.workload["source_padding_slots"] - self.workload["dummy_source_BOS_inputs"] != self.context_tokens + self.source_bos_inputs:
                raise ValueError("Source input/padding workload does not reconcile")
        return {"passed": True, "every_sample_once": True, "every_document_target_once": True,
                "every_terminal_EOS_once": True, "input_workload_accounting_available": measured, **self.summary()}


@torch.no_grad()
def validation(model, dataset, ids, source_names, device, rank=0, world=1,
               microbatch=8, precision=None, progress_callback=None):
    was_training = model.training
    model.eval()
    source_lookup = {name: i for i, name in enumerate(source_names)}
    stats = torch.zeros((len(source_names), 2), dtype=torch.float64, device=device)
    actual_examples = torch.zeros((), dtype=torch.int64, device=device)
    for offset in range(0, len(ids), world * microbatch):
        local_ids = ids[offset + rank * microbatch:offset + (rank + 1) * microbatch]
        if len(local_ids):
            examples = [dataset[int(index)] for index in local_ids]
            batch = move_batch(examples, device)
            source, decoder, labels, sm, tm = batch
            autocast = (torch.autocast(source.device.type, dtype=precision)
                        if precision is not None else contextlib.nullcontext())
            with autocast:
                logits = model(source, decoder, sm, tm)
                per_token = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none", ignore_index=-100)
            values = torch.stack((per_token.double().sum(1), (labels != -100).sum(1).double()), dim=1)
            categories = torch.tensor([source_lookup[row["document_id"].split(":", 1)[0]] for row in examples], device=device)
            stats.index_add_(0, categories, values)
            actual_examples += len(examples)
        if progress_callback is not None and rank == 0 and offset % (world * microbatch * 1000) == 0:
            progress_callback(min(offset + world * microbatch, len(ids)), len(ids))
    if world > 1:
        dist.all_reduce(stats)
        dist.all_reduce(actual_examples)
    if int(actual_examples) != len(ids) or not bool(torch.isfinite(stats).all()) or stats[:, 1].sum() <= 0:
        raise ValueError("Actual held-out examples/loss do not reconcile")
    counts = stats.cpu().numpy()
    total_loss, total_tokens = counts.sum(axis=0)
    result = {"examples": len(ids), "valid_targets": int(total_tokens),
              "cross_entropy": float(total_loss / total_tokens),
              "perplexity": math.exp(float(total_loss / total_tokens)),
              "by_source": {name: {"valid_targets": int(tokens), "cross_entropy": float(loss / tokens) if tokens else None}
                            for name, (loss, tokens) in zip(source_names, counts)}}
    model.train(was_training)
    return result
