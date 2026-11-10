#!/usr/bin/env python3
"""Generate exactly 128 continuation words from a trained Phase 1 model."""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assignment3_impl.q2.model_arch import DecoderLM, LMConfig
from src.phase1_common import MODEL_SPECS, OUTPUT_ROOT, TEST_PATH, TOKENIZER_PATH, atomic_json
from src.tokenization.a1_byte_bpe import ByteBPE


def safe_decode(tokenizer, token_ids):
    try:
        return tokenizer.decode(token_ids)
    except UnicodeDecodeError:
        data = bytearray()
        for token_id in token_ids:
            token = tokenizer.itos[int(token_id)]
            if token not in tokenizer.RESERVED:
                data.extend(bytes.fromhex(token))
        return data.decode("utf-8", errors="replace").lstrip(" ")


def sample_top_p(
    logits,
    temperature,
    top_p,
    generator,
    force_word_start=None,
    word_start_mask=None,
):
    logits = logits.float() / temperature
    logits[:, :4] = float("-inf")
    if force_word_start is not None and force_word_start.any():
        if word_start_mask is None:
            raise ValueError("word_start_mask is required for constrained sampling")
        constrained = logits[force_word_start]
        constrained[:, ~word_start_mask] = float("-inf")
        logits[force_word_start] = constrained
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative - probabilities > top_p
    sorted_logits.masked_fill_(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_logits)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompt_ids, generator, word_start_mask):
    device = next(model.parameters()).device
    prompts = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    prompt_len = prompts.size(1)
    max_new_tokens = min(560, model.cfg.max_seq_len - prompt_len)
    if max_new_tokens < 128:
        raise ValueError(f"only {max_new_tokens} generation slots remain")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, caches = model.forward_with_cache(
            prompts, start_pos=0, last_token_only=True
        )
    generated = [[] for _ in prompt_ids]
    forced_steps = [0 for _ in prompt_ids]

    for step in range(max_new_tokens):
        remaining_slots = max_new_tokens - step
        word_counts = [
            len(safe_decode(tokenizer, ids).split()) for ids in generated
        ]
        force_word_start = torch.tensor(
            [128 - count >= remaining_slots for count in word_counts],
            dtype=torch.bool,
            device=device,
        )
        next_token = sample_top_p(
            logits[:, -1],
            0.1,
            0.95,
            generator,
            force_word_start,
            word_start_mask,
        )
        for row, token_id in enumerate(next_token[:, 0].tolist()):
            generated[row].append(token_id)
            forced_steps[row] += int(force_word_start[row])

        if step >= 127 and (step % 4 == 3 or step + 1 == max_new_tokens):
            enough = all(
                len(safe_decode(tokenizer, ids).split()) >= 128
                for ids in generated
            )
            if enough:
                break

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, caches = model.forward_with_cache(
                next_token,
                caches,
                start_pos=prompt_len + step,
                last_token_only=True,
            )

    texts = []
    for ids in generated:
        words = " ".join(safe_decode(tokenizer, ids).split()).split()
        if len(words) < 128:
            raise RuntimeError(
                f"generation produced only {len(words)} words from {len(ids)} tokens"
            )
        texts.append(" ".join(words[:128]))
    return texts, [len(ids) for ids in generated], forced_steps


def make_word_start_mask(tokenizer, device):
    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    for token_id, piece in enumerate(tokenizer.itos):
        if token_id < 4:
            continue
        try:
            text = bytes.fromhex(piece).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if text.startswith(" ") and text.strip():
            mask[token_id] = True
    if not mask.any():
        raise RuntimeError("tokenizer has no valid word-start tokens")
    return mask


def generate(model_name, run_dir=None):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("generation service must see exactly one CUDA GPU")
    run_dir = Path(run_dir or OUTPUT_ROOT / model_name)
    prediction_path = run_dir / "prediction.json"
    done_path = run_dir / "GENERATION_COMPLETED.json"
    if prediction_path.is_file() and done_path.is_file():
        print(f"Generation already complete: {prediction_path}", flush=True)
        return json.loads(done_path.read_text(encoding="utf-8"))

    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model/model.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("training configuration or best checkpoint is missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = LMConfig(**config["model"])
    model = DecoderLM(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.cuda().eval()
    tokenizer = ByteBPE.load(TOKENIZER_PATH)
    word_start_mask = make_word_start_mask(tokenizer, torch.device("cuda"))

    test_data = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(test_data, dict) or len(test_data) != 1_000:
        raise ValueError("official test JSON must contain exactly 1,000 records")

    groups = defaultdict(list)
    for example_id, record in test_data.items():
        text = record.get("text") if isinstance(record, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"test record {example_id!r} has empty text")
        ids = tokenizer.encode_ids(text)
        if len(ids) + 128 > cfg.max_seq_len:
            raise ValueError(f"test record {example_id!r} cannot fit the context")
        groups[len(ids)].append((str(example_id), ids))

    generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 20_000)
    predictions = {}
    generated_lengths = []
    forced_steps = []
    started = time.time()
    completed = 0
    for prompt_len in sorted(groups):
        group = groups[prompt_len]
        for start in range(0, len(group), 8):
            batch = group[start : start + 8]
            texts, lengths, batch_forced_steps = generate_batch(
                model,
                tokenizer,
                [ids for _, ids in batch],
                generator,
                word_start_mask,
            )
            for (example_id, _), text in zip(batch, texts):
                predictions[example_id] = {"text": text}
            generated_lengths.extend(lengths)
            forced_steps.extend(batch_forced_steps)
            completed += len(batch)
            if completed % 100 == 0 or completed == len(test_data):
                print(f"generated {completed}/1000", flush=True)

    ordered = {str(key): predictions[str(key)] for key in test_data}
    atomic_json(prediction_path, ordered)
    result = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "examples": len(ordered),
        "target_words": 128,
        "temperature": 0.1,
        "top_p": 0.95,
        "eos_and_other_special_tokens_suppressed": True,
        "word_boundary_repairs": sum(count > 0 for count in forced_steps),
        "forced_word_start_tokens": sum(forced_steps),
        "maximum_forced_word_start_tokens": max(forced_steps),
        "minimum_generated_tokens": min(generated_lengths),
        "maximum_generated_tokens": max(generated_lengths),
        "mean_generated_tokens": sum(generated_lengths) / len(generated_lengths),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(done_path, result)
    print(f"Generation complete: {prediction_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    generate(args.model, args.run_dir)


if __name__ == "__main__":
    main()
