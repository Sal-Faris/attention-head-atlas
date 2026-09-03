"""Summarize population structure and head continuity across checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/trajectory_geometry.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/trajectory_geometry.png"),
    )
    return parser.parse_args()


def load_view(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return (
            np.asarray(bundle["distances"], dtype=np.float64),
            np.asarray(bundle["checkpoint_values"], dtype=np.int64),
        )


def summarize_view(distances: np.ndarray, checkpoint_values: np.ndarray) -> dict:
    steps = np.unique(checkpoint_values)
    checkpoint_indices = [np.flatnonzero(checkpoint_values == step) for step in steps]
    within = []
    for step, indices in zip(steps, checkpoint_indices, strict=True):
        block = distances[np.ix_(indices, indices)]
        upper = block[np.triu_indices(len(indices), 1)]
        within.append(
            {
                "checkpoint": int(step),
                "mean_pairwise_distance": float(np.mean(upper)),
                "standard_deviation_pairwise_distance": float(np.std(upper)),
            }
        )

    adjacent = []
    for first_step, second_step, first, second in zip(
        steps[:-1], steps[1:], checkpoint_indices[:-1], checkpoint_indices[1:], strict=True
    ):
        block = distances[np.ix_(first, second)]
        matched = np.diag(block)
        off_diagonal = block[~np.eye(len(block), dtype=bool)]
        nearest = np.argmin(block, axis=1)
        adjacent.append(
            {
                "first_checkpoint": int(first_step),
                "second_checkpoint": int(second_step),
                "mean_same_head_distance": float(np.mean(matched)),
                "mean_different_head_distance": float(np.mean(off_diagonal)),
                "same_to_different_distance_ratio": float(
                    np.mean(matched) / np.mean(off_diagonal)
                ),
                "nearest_neighbor_head_identity_accuracy": float(
                    np.mean(nearest == np.arange(len(first)))
                ),
            }
        )
    return {
        "checkpoint_values": steps.tolist(),
        "within_checkpoint": within,
        "adjacent_checkpoint_continuity": adjacent,
    }


def main() -> None:
    args = parse_args()
    views = {}
    for kind in ("QK", "OV"):
        distances, checkpoint_values = load_view(
            args.artifact_root / f"{kind.lower()}_trajectory_distances.npz"
        )
        views[kind] = summarize_view(distances, checkpoint_values)

    report = {
        "analysis_status": "descriptive trajectory geometry",
        "distance": "normalized Frobenius chord distance",
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for kind in ("QK", "OV"):
        within = views[kind]["within_checkpoint"]
        steps = np.asarray([record["checkpoint"] for record in within])
        mean = np.asarray([record["mean_pairwise_distance"] for record in within])
        deviation = np.asarray(
            [record["standard_deviation_pairwise_distance"] for record in within]
        )
        axes[0].plot(steps + 1, mean, marker="o", label=kind)
        axes[0].fill_between(steps + 1, mean - deviation, mean + deviation, alpha=0.18)
        adjacent = views[kind]["adjacent_checkpoint_continuity"]
        destination = np.asarray([record["second_checkpoint"] for record in adjacent])
        drift = np.asarray([record["mean_same_head_distance"] for record in adjacent])
        background = np.asarray(
            [record["mean_different_head_distance"] for record in adjacent]
        )
        axes[1].plot(destination + 1, drift, marker="o", label=f"{kind} same head")
        axes[1].plot(
            destination + 1,
            background,
            linestyle="--",
            label=f"{kind} different heads",
        )
    axes[0].set_xscale("log")
    axes[0].set_title("Population geometry departs from random orthogonality")
    axes[0].set_xlabel("Training step + 1 (log scale)")
    axes[0].set_ylabel("Within-checkpoint distance")
    axes[0].axhline(np.sqrt(2), color="gray", linewidth=1, linestyle=":")
    axes[0].legend()
    axes[1].set_xscale("log")
    axes[1].set_title("Individual head trajectories remain identifiable")
    axes[1].set_xlabel("Destination training step + 1 (log scale)")
    axes[1].set_ylabel("Adjacent-checkpoint distance")
    axes[1].legend(fontsize=8)
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)
    print(f"saved trajectory report to {args.output} and figure to {args.figure}")


if __name__ == "__main__":
    main()
