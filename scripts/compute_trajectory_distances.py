"""Compute population distances across every checkpoint in a factor manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import blockwise_factorized_frobenius_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--scratch-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for kind in ("QK", "OV"):
        operators = []
        checkpoint_labels = []
        checkpoint_values = []
        source_hashes = []
        reference_locations = None
        for record in manifest["records"]:
            factor_record = record["factors"][kind]
            checkpoint_operators, metadata = load_factor_bundle(factor_record["path"])
            locations = [
                (operator.layer, operator.head) for operator in checkpoint_operators
            ]
            if reference_locations is None:
                reference_locations = locations
            elif locations != reference_locations:
                raise ValueError(
                    f"head ordering changed at {record['revision']} for {kind}"
                )
            if metadata["snapshot_commit"] != record["snapshot_commit"]:
                raise ValueError(f"snapshot provenance mismatch at {record['revision']}")
            step = int(record["revision"].removeprefix("step"))
            operators.extend(checkpoint_operators)
            checkpoint_labels.extend([record["revision"]] * len(checkpoint_operators))
            checkpoint_values.extend([step] * len(checkpoint_operators))
            source_hashes.append(factor_record["sha256"])

        print(f"computing {kind} distances for {len(operators)} observations", flush=True)
        distances = blockwise_factorized_frobenius_distances(
            operators,
            block_size=args.block_size,
            scratch_directory=args.scratch_directory,
        )
        output = args.output_root / f"{kind.lower()}_trajectory_distances.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            distances=distances,
            checkpoints=np.asarray(checkpoint_labels),
            checkpoint_values=np.asarray(checkpoint_values, dtype=np.int64),
            layers=np.asarray([operator.layer for operator in operators], dtype=np.int64),
            heads=np.asarray([operator.head for operator in operators], dtype=np.int64),
            kinds=np.asarray([operator.kind for operator in operators]),
            metric=np.asarray("normalized_frobenius"),
            qk_position_view=np.asarray(
                "pre_rotary_zero_relative_rotation" if kind == "QK" else "not_applicable"
            ),
            source_factor_sha256=np.asarray(source_hashes),
            source_manifest=np.asarray(str(args.manifest)),
        )
        print(f"saved {distances.shape} {kind} distances to {output}", flush=True)


if __name__ == "__main__":
    main()
