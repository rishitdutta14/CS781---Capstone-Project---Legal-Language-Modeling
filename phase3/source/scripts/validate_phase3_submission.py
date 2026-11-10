"""Validate a local RR candidate without creating or uploading a submission."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase3_submission import file_sha256, load_test_contract, strict_loads, validate_submission


def run(args):
    if args.report and args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing validation report")
    bindings = {"selected_checkpoint": args.checkpoint, "tokenizer": args.tokenizer,
                "inference_config": args.inference_config}
    if any(bindings.values()) and not all(bindings.values()):
        raise ValueError("Provide checkpoint, tokenizer and inference config together, or none")
    contract = load_test_contract(args.raw_dir, args.download_manifest)
    result = validate_submission(args.prediction, args.zip, contract)
    result["validator_sha256"] = file_sha256(ROOT / "src/phase3_submission.py")
    result["cli_sha256"] = file_sha256(Path(__file__))
    result["artifact_bindings"] = {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in bindings.items() if path is not None}
    if args.inference_config:
        if not isinstance(strict_loads(args.inference_config.read_bytes()), dict):
            raise ValueError("Inference config must be a JSON object")
    result["binding_scope"] = (
        "Hashes identify supplied files; actual checkpoint use and ordered inference "
        "must be verified separately by the model/export audit")
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/Phase_3_RR")
    parser.add_argument("--download-manifest", type=Path,
                        default=ROOT / "reports/phase3/data_setup_20261016_r1/download_manifest.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--inference-config", type=Path)
    parser.add_argument("--report", type=Path)
    run(parser.parse_args())
