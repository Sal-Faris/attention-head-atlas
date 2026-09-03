"""Measure how Pythia QK geometry changes under relative RoPE offsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import (
    blockwise_factorized_frobenius_distances,
    rotate_qk_relative,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pythia70m_deduped_pilot.json")
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/rotary_qk_robustness.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/rotary_qk_robustness.png"),
    )
    parser.add_argument("--scratch-directory", type=Path)
    return parser.parse_args()


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), 1)]


def nearest_neighbors(distances: np.ndarray) -> np.ndarray:
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    return np.argmin(masked, axis=1)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    offsets = [int(offset) for offset in config["qk_rotary_relative_offsets"]]
    records = {}
    for checkpoint in config["qk_rotary_audit_checkpoints"]:
        operators, metadata = load_factor_bundle(
            args.artifact_root / checkpoint / "qk_factors.npz"
        )
        rotary_fraction = float(metadata["rotary_pct"])
        base = float(metadata["rope_theta"])
        checkpoint_records = []
        zero_distances = None
        zero_neighbors = None
        for offset in offsets:
            rotated = [
                rotate_qk_relative(
                    operator,
                    offset,
                    rotary_fraction=rotary_fraction,
                    base=base,
                )
                for operator in operators
            ]
            distances = blockwise_factorized_frobenius_distances(
                rotated,
                block_size=8,
                scratch_directory=args.scratch_directory,
            )
            if zero_distances is None:
                zero_distances = distances
                zero_neighbors = nearest_neighbors(distances)
            zero_upper = upper_triangle(zero_distances)
            offset_upper = upper_triangle(distances)
            checkpoint_records.append(
                {
                    "relative_offset": offset,
                    "pearson_distance_correlation_with_offset_zero": float(
                        np.corrcoef(zero_upper, offset_upper)[0, 1]
                    ),
                    "spearman_distance_correlation_with_offset_zero": float(
                        spearmanr(zero_upper, offset_upper).statistic
                    ),
                    "nearest_neighbor_preservation": float(
                        np.mean(nearest_neighbors(distances) == zero_neighbors)
                    ),
                    "mean_absolute_distance_change": float(
                        np.mean(np.abs(offset_upper - zero_upper))
                    ),
                    "maximum_absolute_distance_change": float(
                        np.max(np.abs(offset_upper - zero_upper))
                    ),
                }
            )
        records[checkpoint] = checkpoint_records
        print(f"audited {checkpoint} at {len(offsets)} relative offsets", flush=True)

    report = {
        "analysis_status": "preregistered QK positional-robustness audit",
        "operator_interpretation": "content QK with explicit relative GPT-NeoX RoPE rotation",
        "views": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    metrics = (
        ("spearman_distance_correlation_with_offset_zero", "Distance-rank correlation"),
        ("nearest_neighbor_preservation", "Nearest-neighbor preservation"),
        ("mean_absolute_distance_change", "Mean absolute distance change"),
    )
    for checkpoint, checkpoint_records in records.items():
        x = np.asarray([record["relative_offset"] for record in checkpoint_records])
        for axis, (metric, label) in zip(axes, metrics, strict=True):
            axis.plot(
                x + 1,
                [record[metric] for record in checkpoint_records],
                marker="o",
                label=checkpoint,
            )
            axis.set_xscale("log")
            axis.set_xlabel("Relative token offset + 1 (log scale)")
            axis.set_ylabel(label)
    axes[0].set_title("Global QK geometry")
    axes[1].set_title("Local QK neighborhoods")
    axes[2].set_title("Absolute geometric change")
    axes[0].legend(fontsize=8)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)
    print(f"saved rotary audit to {args.output} and figure to {args.figure}")


if __name__ == "__main__":
    main()
