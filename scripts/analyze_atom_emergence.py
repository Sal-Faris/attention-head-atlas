"""Test selected atom usage for ordered emergence across Pythia training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


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
        default=Path("results/pythia-70m-deduped/atom_emergence.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/atom_emergence.png"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile", choices=("optimal", "compact", "residual"), default="optimal"
    )
    return parser.parse_args()


def false_discovery_rates(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def maximum_consecutive(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def permutation_correlations(
    strengths: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    time_ranks = rankdata(np.arange(len(strengths)), method="average")
    time_ranks -= np.mean(time_ranks)
    strength_ranks = np.apply_along_axis(rankdata, 0, strengths)
    strength_ranks -= np.mean(strength_ranks, axis=0, keepdims=True)
    denominator = np.linalg.norm(time_ranks) * np.linalg.norm(strength_ranks, axis=0)
    observed = time_ranks @ strength_ranks / denominator
    permuted_times = np.stack([rng.permutation(time_ranks) for _ in range(repetitions)])
    null_correlations = permuted_times @ strength_ranks / denominator[None, :]
    p_values = (1 + np.sum(np.abs(null_correlations) >= np.abs(observed), axis=0)) / (
        repetitions + 1
    )
    return observed, p_values


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    repetitions = int(config["time_permutation_repetitions"])
    false_discovery_rate = float(config["emergence_false_discovery_rate"])
    minimum_fraction = float(config["recurrence_minimum_head_fraction"])
    minimum_multiplier = float(
        config["recurrence_minimum_initialization_strength_multiplier"]
    )
    minimum_adjacent = int(config["recurrence_minimum_adjacent_checkpoints"])
    rng = np.random.default_rng(args.seed)
    report_views = {}
    heatmaps = {}

    for view_name in ("QK", "OV", "JOINT"):
        if args.profile == "compact":
            artifact_stem = f"{view_name.lower()}_compact_dictionary.npz"
        elif args.profile == "residual":
            artifact_stem = f"{view_name.lower()}_residual_compact_dictionary.npz"
        else:
            artifact_stem = f"{view_name.lower()}_dictionary.npz"
        artifact_path = args.artifact_root / artifact_stem
        with np.load(artifact_path, allow_pickle=False) as artifact:
            codes = np.asarray(artifact["codes"], dtype=np.float64)
            checkpoint_values = np.asarray(artifact["checkpoint_values"], dtype=np.int64)
        steps = np.unique(checkpoint_values)
        strengths = np.stack(
            [np.mean(np.abs(codes[checkpoint_values == step]), axis=0) for step in steps]
        )
        prevalence = np.stack(
            [
                np.mean(np.abs(codes[checkpoint_values == step]) > 1e-12, axis=0)
                for step in steps
            ]
        )
        shares = strengths / np.maximum(np.sum(strengths, axis=1, keepdims=True), 1e-12)
        correlations, p_values = permutation_correlations(shares, repetitions, rng)
        q_values = false_discovery_rates(p_values)
        initialization_strength = strengths[0]
        recurrence_mask = (prevalence >= minimum_fraction) & (
            strengths >= minimum_multiplier * np.maximum(initialization_strength, 1e-12)
        )

        atoms = []
        for atom in range(codes.shape[1]):
            consecutive = maximum_consecutive(recurrence_mask[:, atom])
            atoms.append(
                {
                    "atom": atom,
                    "spearman_training_correlation": float(correlations[atom]),
                    "time_permutation_p_value": float(p_values[atom]),
                    "benjamini_hochberg_q_value": float(q_values[atom]),
                    "significant_monotonic_trajectory": bool(
                        q_values[atom] <= false_discovery_rate
                    ),
                    "initialization_strength": float(initialization_strength[atom]),
                    "final_strength": float(strengths[-1, atom]),
                    "final_to_initialization_ratio": float(
                        strengths[-1, atom] / max(initialization_strength[atom], 1e-12)
                    ),
                    "maximum_adjacent_recurrent_checkpoints": consecutive,
                    "passes_recurrence_rule": consecutive >= minimum_adjacent,
                    "strength_by_checkpoint": strengths[:, atom].tolist(),
                    "coefficient_share_by_checkpoint": shares[:, atom].tolist(),
                    "active_head_fraction_by_checkpoint": prevalence[:, atom].tolist(),
                }
            )
        significant_count = sum(
            atom["significant_monotonic_trajectory"] for atom in atoms
        )
        recurrent_count = sum(atom["passes_recurrence_rule"] for atom in atoms)
        report_views[view_name] = {
            "checkpoint_values": steps.tolist(),
            "atom_count": codes.shape[1],
            "significant_monotonic_atom_count": significant_count,
            "recurrent_atom_count": recurrent_count,
            "passes_protocol_emergence_gate": bool(significant_count or recurrent_count),
            "atoms": atoms,
        }
        emergence_center = np.sum(
            shares * np.arange(len(steps))[:, None], axis=0
        ) / np.maximum(np.sum(shares, axis=0), 1e-12)
        order = np.argsort(emergence_center)
        heatmaps[view_name] = (steps, shares[:, order].T, order)
        print(
            f"{view_name}: {significant_count} monotonic, {recurrent_count} recurrent atoms",
            flush=True,
        )

    report = {
        "analysis_status": "preregistered post-selection emergence audit",
        "profile": args.profile,
        "coefficient_measure": (
            "atom share of checkpoint mean absolute OMP coefficients; normalized "
            "within checkpoint before temporal testing"
        ),
        "time_null": "checkpoint-order permutation",
        "time_permutation_repetitions": repetitions,
        "false_discovery_rate": false_discovery_rate,
        "recurrence_rule": {
            "minimum_active_head_fraction": minimum_fraction,
            "minimum_initialization_strength_multiplier": minimum_multiplier,
            "minimum_adjacent_checkpoints": minimum_adjacent,
        },
        "views": report_views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    figure, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    for axis, view_name in zip(axes, ("QK", "OV", "JOINT"), strict=True):
        steps, shares, order = heatmaps[view_name]
        image = axis.imshow(shares, aspect="auto", interpolation="nearest", cmap="viridis")
        axis.set_title(f"{view_name}: atom coefficient share across training")
        axis.set_xlabel("Training checkpoint")
        axis.set_ylabel("Atoms ordered by temporal center")
        axis.set_xticks(np.arange(len(steps)), [str(step) for step in steps], rotation=30)
        axis.set_yticks(np.arange(len(order)), [str(atom) for atom in order], fontsize=6)
        figure.colorbar(image, ax=axis, label="Share of mean |coefficient|")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)
    print(f"saved emergence report to {args.output} and figure to {args.figure}")


if __name__ == "__main__":
    main()
