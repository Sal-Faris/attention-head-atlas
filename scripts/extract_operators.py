"""Extract QK or OV operators from a TransformerLens model.

Usage:
    python scripts/extract_operators.py --model gpt2-small --kind OV \
        --output artifacts/gpt2-small/ov_operators.npz
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from head_atlas.model_io import extract_from_transformer_lens, save_operator_bundle


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--kind", choices=("OV", "QK"), default="OV")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformer_lens import HookedTransformer
        import torch
        import transformer_lens
    except ImportError as error:
        raise SystemExit('Install model dependencies with: pip install -e ".[models]"') from error

    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    operators = extract_from_transformer_lens(model, args.kind)
    save_operator_bundle(
        args.output,
        operators,
        {
            "model": args.model,
            "kind": args.kind,
            "device": args.device,
            "git_commit": git_commit(),
            "torch": torch.__version__,
            "transformer_lens": transformer_lens.__version__,
            "n_layers": int(model.cfg.n_layers),
            "n_heads": int(model.cfg.n_heads),
            "d_model": int(model.cfg.d_model),
            "d_head": int(model.cfg.d_head),
        },
    )
    print(f"saved {len(operators)} {args.kind} operators to {args.output}")


if __name__ == "__main__":
    main()

