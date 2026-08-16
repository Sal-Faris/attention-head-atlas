"""Compute projector-based distances between leading operator subspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

from head_atlas.distances import chordal_subspace_distances, weighted_product_distances
from head_atlas.model_io import load_operator_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--power-iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranks = sorted(set(args.ranks))
    if not ranks or ranks[0] < 1:
        raise ValueError("ranks must be positive")
    if args.power_iterations < 0:
        raise ValueError("power iterations must be nonnegative")

    operators, source_metadata = load_operator_bundle(args.input)
    dimension = operators[0].matrix.shape[0]
    maximum_rank = ranks[-1]
    if maximum_rank >= dimension:
        raise ValueError("maximum rank must be smaller than the operator dimension")

    left_bases = []
    right_bases = []
    singular_values = []
    for index, operator in enumerate(operators):
        left, values, right_transpose = randomized_svd(
            np.asarray(operator.matrix),
            n_components=maximum_rank,
            n_iter=args.power_iterations,
            random_state=args.seed + index,
        )
        left_bases.append(left)
        right_bases.append(right_transpose.T)
        singular_values.append(values)
        if (index + 1) % 12 == 0 or index + 1 == len(operators):
            print(f"decomposed {index + 1}/{len(operators)} operators", flush=True)

    left_array = np.stack(left_bases)
    right_array = np.stack(right_bases)
    payload: dict[str, np.ndarray] = {
        "layers": np.asarray([operator.layer for operator in operators], dtype=np.int64),
        "heads": np.asarray([operator.head for operator in operators], dtype=np.int64),
        "kinds": np.asarray([operator.kind for operator in operators]),
        "ranks": np.asarray(ranks, dtype=np.int64),
        "singular_values": np.stack(singular_values),
    }
    for rank in ranks:
        left_distances = chordal_subspace_distances(left_array[:, :, :rank])
        right_distances = chordal_subspace_distances(right_array[:, :, :rank])
        payload[f"left_rank_{rank}"] = left_distances
        payload[f"right_rank_{rank}"] = right_distances
        payload[f"joint_rank_{rank}"] = weighted_product_distances(
            [left_distances, right_distances]
        )
        print(f"computed left/right/joint rank-{rank} distances", flush=True)

    metadata = {
        "metric": "normalized_chordal_projector_distance",
        "algorithm": "sklearn.randomized_svd",
        "power_iterations": args.power_iterations,
        "seed": args.seed,
        "left_semantics": "input/read or query-side leading singular subspace",
        "right_semantics": "output/write or key-side leading singular subspace",
        "source_metadata": source_metadata,
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved projector distances to {args.output}")


if __name__ == "__main__":
    main()
