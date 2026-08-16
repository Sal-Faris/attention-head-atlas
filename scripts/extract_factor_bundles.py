"""Extract exact compact QK and OV factor bundles from a TransformerLens model."""

from __future__ import annotations

import argparse
import subprocess
from importlib.metadata import version
from pathlib import Path

from head_atlas.factor_io import (
    extract_factors_from_transformer_lens,
    extract_processed_factors_from_safetensors,
    save_factor_bundle,
    verify_factorized_actions,
)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--snapshot", type=Path)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--checkpoint-value", type=int)
    checkpoint.add_argument("--revision")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.snapshot is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("QK", "OV"):
            operators, snapshot_metadata = extract_processed_factors_from_safetensors(
                args.snapshot, kind
            )
            output = args.output_dir / f"{kind.lower()}_factors.npz"
            save_factor_bundle(
                output,
                operators,
                {
                    **snapshot_metadata,
                    "model": args.model,
                    "snapshot": str(args.snapshot.resolve()),
                    "checkpoint_value": args.checkpoint_value,
                    "model_revision": args.revision,
                    "git_commit": git_commit(),
                },
            )
            print(f"saved {len(operators)} {kind} factorized operators to {output}")
        return

    try:
        import torch
        from transformer_lens import HookedTransformer
    except ImportError as error:
        raise SystemExit('Install model dependencies with: pip install -e ".[models]"') from error

    load_options = {
        "device": args.device,
        "checkpoint_value": args.checkpoint_value,
        "revision": args.revision,
    }
    if args.cache_dir is not None:
        load_options["cache_dir"] = str(args.cache_dir)
    model = HookedTransformer.from_pretrained(args.model, **load_options)

    common_metadata = {
        "model": args.model,
        "checkpoint_value": args.checkpoint_value,
        "model_revision": args.revision,
        "device": args.device,
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "transformer_lens": version("transformer-lens"),
        "n_layers": int(model.cfg.n_layers),
        "n_heads": int(model.cfg.n_heads),
        "d_model": int(model.cfg.d_model),
        "d_head": int(model.cfg.d_head),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("QK", "OV"):
        operators = extract_factors_from_transformer_lens(model, kind)
        extraction_error = verify_factorized_actions(model, operators)
        if extraction_error["maximum_relative_error"] > 1e-5:
            raise RuntimeError(
                f"{kind} factor extraction check failed: "
                f"maximum relative error {extraction_error['maximum_relative_error']}"
            )
        output = args.output_dir / f"{kind.lower()}_factors.npz"
        save_factor_bundle(output, operators, {**common_metadata, **extraction_error})
        print(f"saved {len(operators)} {kind} factorized operators to {output}")


if __name__ == "__main__":
    main()
