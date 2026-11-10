"""Shared configuration and small utilities for the Phase 1 runs."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from assignment3_impl.q2.model_arch import DecoderLM, LMConfig


ROOT = Path(__file__).resolve().parents[1]
SHARD_DIR = ROOT / "data/processed/phase1_16k_100m"
TOKENIZER_PATH = ROOT / "tokenizers/capstone_legal_bpe_16k.json"
TEST_PATH = ROOT / "data/raw/Phase_1_Pretraining/Test/test.json"
OUTPUT_ROOT = ROOT / "outputs/phase1"
SOURCES = ("acts", "high_courts", "supreme_court", "tribunals")

MODEL_SPECS = {
    "mha": {"attention_type": "standard", "n_kv_heads": 8, "kv_latent_dim": None},
    "gqa": {"attention_type": "standard", "n_kv_heads": 2, "kv_latent_dim": None},
    "mqa": {"attention_type": "standard", "n_kv_heads": 1, "kv_latent_dim": None},
    "mla": {"attention_type": "mla", "n_kv_heads": None, "kv_latent_dim": 128},
}


def model_config(name):
    if name not in MODEL_SPECS:
        raise ValueError(f"unknown model: {name}")
    spec = MODEL_SPECS[name]
    return LMConfig(
        vocab_size=16_000,
        d_model=640,
        n_layers=10,
        n_heads=8,
        n_kv_heads=spec["n_kv_heads"],
        d_ff=2_560,
        max_seq_len=1_600,
        dropout=0.1,
        pos_emb_type="sinusoidal",
        attention_scale=None,
        norm_first=True,
        attention_type=spec["attention_type"],
        kv_latent_dim=spec["kv_latent_dim"],
    )


def build_model(name):
    return DecoderLM(model_config(name))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def config_record(name):
    cfg = model_config(name)
    return {
        "model_name": name,
        "model": asdict(cfg),
        "tokenizer": str(TOKENIZER_PATH.resolve()),
        "shards": str(SHARD_DIR.resolve()),
        "test_file": str(TEST_PATH.resolve()),
        "seed": 781,
        "train_tokens": 200_000_000,
        "train_seq_len": 1_024,
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "weight_decay": 0.1,
        "warmup_steps": 500,
        "eval_every": 500,
        "eval_batches": 50,
        "eval_batch_size": 4,
        "gradient_clip": 1.0,
        "precision": "bfloat16",
        "official_data_only": True,
        "random_initialization": True,
    }


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
