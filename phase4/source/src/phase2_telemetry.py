"""Read-only resource measurements; these never enter the loss or selection."""

import resource

import torch
import torch.distributed as dist


def local_memory():
    return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cpu_process_lifetime_peak_rss_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def training_memory_by_rank():
    measurements = [None] * dist.get_world_size()
    dist.all_gather_object(measurements, local_memory())
    return measurements


def summarize_memory(history):
    measured = [row for row in history if "training_memory_by_rank" in row]
    if not measured:
        return {"recorded": False, "reason": "This historical trainer did not persist per-rank allocator peaks. No peak is inferred from a configured cap or a later nvidia-smi sample."}
    ranks = len(measured[0]["training_memory_by_rank"])
    if any(len(row["training_memory_by_rank"]) != ranks for row in measured):
        raise ValueError("rank count changed within memory history")
    return {"recorded": True, "epochs_measured": len(measured),
            "training_peaks_by_rank": [{key: max(row["training_memory_by_rank"][rank][key] for row in measured)
                                         for key in measured[0]["training_memory_by_rank"][rank]} for rank in range(ranks)],
            "rank0_peak_including_validation": {key: max(row["rank0_peak_including_validation"][key] for row in measured)
                                               for key in measured[0]["rank0_peak_including_validation"]},
            "scope": "CUDA allocator training peaks per epoch/rank; rank0 also includes validation. CPU RSS is each process lifetime maximum, not simultaneous aggregate RAM. Driver/NCCL allocations outside the PyTorch allocator are excluded."}
