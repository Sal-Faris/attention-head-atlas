"""Compute normalized Frobenius distances from compact factor bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import blockwise_factorized_frobenius_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--scratch-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    operators, source_metadata = load_factor_bundle(args.input)
    distances = blockwise_factorized_frobenius_distances(
        operators,
        block_size=args.block_size,
        scratch_directory=args.scratch_directory,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        distances=distances,
        layers=np.asarray([operator.layer for operator in operators], dtype=np.int64),
        heads=np.asarray([operator.head for operator in operators], dtype=np.int64),
        kinds=np.asarray([operator.kind for operator in operators]),
        metric=np.asarray("normalized_frobenius"),
        source_metadata_json=np.asarray(json.dumps(source_metadata, sort_keys=True)),
    )
    print(f"saved {distances.shape[0]} x {distances.shape[1]} distances to {args.output}")


if __name__ == "__main__":
    main()
