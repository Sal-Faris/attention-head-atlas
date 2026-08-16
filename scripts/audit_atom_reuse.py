"""Measure whether dictionary atoms are shared across independent head trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.dictionary import head_trajectory_groups


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
        default=Path("results/pythia-70m-deduped/atom_reuse.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/atom_reuse.png"),
    )
    return parser.parse_args()


def concentration(values: np.ndarray) -> tuple[float, float]:
    nonnegative = np.abs(np.asarray(values, dtype=np.float64))
    total = float(np.sum(nonnegative))
    if total <= 1e-12:
        return 0.0, 1.0
    shares = nonnegative / total
    return float(1.0 / np.sum(shares**2)), float(np.max(shares))


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def audit_artifact(
    artifact: dict[str, np.ndarray],
    minimum_effective: float,
    maximum_share: float,
    minimum_effective_layers: float,
    maximum_layer_share: float,
) -> dict:
    codes = np.asarray(artifact["codes"], dtype=np.float64)
    discovery = np.asarray(artifact["discovery_mask"], dtype=bool)
    groups = head_trajectory_groups(
        artifact["layers"][discovery], artifact["heads"][discovery]
    )
    unique_groups = np.unique(groups)
    trajectory_usage = np.stack(
        [
            np.mean(np.abs(codes[discovery][groups == group]), axis=0)
            for group in unique_groups
        ]
    )
    discovery_layers = np.asarray(artifact["layers"])[discovery]
    unique_layers = np.unique(discovery_layers)
    layer_usage = np.stack(
        [
            np.mean(np.abs(codes[discovery][discovery_layers == layer]), axis=0)
            for layer in unique_layers
        ]
    )
    final_step = int(np.max(artifact["checkpoint_values"]))
    final_mask = artifact["checkpoint_values"] == final_step
    final_usage = np.abs(codes[final_mask])
    records = []
    for atom in range(codes.shape[1]):
        effective_trajectories, top_trajectory_share = concentration(
            trajectory_usage[:, atom]
        )
        effective_final_heads, top_final_head_share = concentration(final_usage[:, atom])
        effective_layers, top_layer_share = concentration(layer_usage[:, atom])
        records.append(
            {
                "atom": atom,
                "effective_trajectory_participation": effective_trajectories,
                "largest_trajectory_share": top_trajectory_share,
                "effective_final_head_participation": effective_final_heads,
                "largest_final_head_share": top_final_head_share,
                "active_final_head_count": int(np.sum(final_usage[:, atom] > 1e-12)),
                "effective_layer_participation": effective_layers,
                "largest_layer_share": top_layer_share,
                "passes_reuse_rule": bool(
                    effective_trajectories >= minimum_effective
                    and top_trajectory_share <= maximum_share
                ),
                "passes_cross_layer_reuse_rule": bool(
                    effective_trajectories >= minimum_effective
                    and top_trajectory_share <= maximum_share
                    and effective_layers >= minimum_effective_layers
                    and top_layer_share <= maximum_layer_share
                ),
            }
        )
    return {
        "atom_count": codes.shape[1],
        "reusable_atom_count": sum(record["passes_reuse_rule"] for record in records),
        "cross_layer_reusable_atom_count": sum(
            record["passes_cross_layer_reuse_rule"] for record in records
        ),
        "atoms": records,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    minimum_effective = float(config["reuse_minimum_effective_trajectories"])
    maximum_share = float(config["reuse_maximum_single_trajectory_share"])
    minimum_effective_layers = float(config["reuse_minimum_effective_layers"])
    maximum_layer_share = float(config["reuse_maximum_single_layer_share"])
    results = {}
    for view in ("QK", "OV", "JOINT"):
        results[view] = {}
        for profile in ("optimal", "compact", "residual"):
            if profile == "compact":
                stem = f"{view.lower()}_compact_dictionary.npz"
            elif profile == "residual":
                stem = f"{view.lower()}_residual_compact_dictionary.npz"
            else:
                stem = f"{view.lower()}_dictionary.npz"
            results[view][profile] = audit_artifact(
                load_artifact(args.artifact_root / stem),
                minimum_effective,
                maximum_share,
                minimum_effective_layers,
                maximum_layer_share,
            )
            print(
                f"{view} {profile}: "
                f"{results[view][profile]['reusable_atom_count']} reusable, "
                f"{results[view][profile]['cross_layer_reusable_atom_count']} cross-layer",
                flush=True,
            )

    report = {
        "analysis_status": "post-selection atom reuse audit",
        "reuse_rule": {
            "minimum_effective_trajectories": minimum_effective,
            "maximum_single_trajectory_share": maximum_share,
            "minimum_effective_layers": minimum_effective_layers,
            "maximum_single_layer_share": maximum_layer_share,
        },
        "views": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, view in zip(axes, ("QK", "OV", "JOINT"), strict=True):
        for profile, marker in (("optimal", "o"), ("compact", "s")):
            values = sorted(
                record["effective_trajectory_participation"]
                for record in results[view][profile]["atoms"]
            )
            quantiles = (np.arange(len(values)) + 0.5) / len(values)
            axis.plot(values, quantiles, marker=marker, label=profile)
        axis.axvline(minimum_effective, color="gray", linestyle=":", linewidth=1)
        axis.set_title(view)
        axis.set_xlabel("Effective head trajectories per atom")
        axis.set_ylabel("Cumulative atom fraction")
        axis.legend(fontsize=8)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)
    print(f"saved reuse audit to {args.output} and figure to {args.figure}")


if __name__ == "__main__":
    main()
