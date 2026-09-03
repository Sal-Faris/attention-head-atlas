"""Plot held-out reuse and emergent dimensions of unrestricted QK modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/pythia-70m-deduped/unrestricted_qk_mode_dimensions_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/unrestricted_qk_mode_dimensions_v1.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"head_parity": "#3465a4", "layer_parity": "#d36b32"}
    labels = {"head_parity": "Unseen heads", "layer_parity": "Unseen layers"}

    for scheme in ("head_parity", "layer_parity"):
        section = report["schemes"][scheme]
        counts = np.asarray(sorted(map(int, section["curves"])))
        full = np.asarray(
            [section["curves"][str(count)]["unrestricted_full_rank_variance_recovered"] for count in counts]
        )
        rank64 = np.asarray(
            [section["curves"][str(count)]["truncated_variance_recovered"]["64"] for count in counts]
        )
        axes[0, 0].plot(counts, 100 * full, "o-", color=colors[scheme], label=labels[scheme])
        axes[0, 0].plot(counts, 100 * rank64, "o--", color=colors[scheme], alpha=0.65)
    axes[0, 0].set(
        title="Transfer of unrestricted population modes",
        xlabel="Population modes learned from training weights",
        ylabel="Complete QK variance recovered (%)",
        xticks=[1, 2, 4, 8, 16],
    )
    axes[0, 0].legend(frameon=False)
    axes[0, 0].text(15.5, 5.0, "solid: full mode\ndashed: rank-64 truncation", ha="right", fontsize=8)

    for scheme in ("head_parity", "layer_parity"):
        curve = report["schemes"][scheme]["curves"]["4"]
        ranks = np.asarray(sorted(map(int, curve["truncated_variance_recovered"])))
        recovered = np.asarray(
            [curve["truncated_variance_recovered"][str(rank)] for rank in ranks]
        )
        fraction = recovered / curve["unrestricted_full_rank_variance_recovered"]
        axes[0, 1].plot(ranks, 100 * fraction, "o-", color=colors[scheme], label=labels[scheme])
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set(
        title="How much dimension the four-mode span needs",
        xlabel="Rank retained within each unrestricted mode",
        ylabel="Fraction of full-mode transfer retained (%)",
        xticks=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    axes[0, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0, 1].legend(frameon=False)

    indices = np.arange(1, 9)
    for scheme in ("head_parity", "layer_parity"):
        dimensions = report["schemes"][scheme]["mean_mode_dimensions_across_folds"][:8]
        rank90 = [item["rank_90_percent_energy"] for item in dimensions]
        axes[1, 0].plot(indices, rank90, "o-", color=colors[scheme], label=labels[scheme])
    axes[1, 0].set(
        title="Dimensions emerge as a hierarchy, not one size",
        xlabel="Population mode",
        ylabel="Rank needed for 90% mode energy",
        xticks=indices,
    )
    axes[1, 0].legend(frameon=False)

    for scheme in ("head_parity", "layer_parity"):
        stability = report["schemes"][scheme]["disjoint_training_mode_stability"]
        counts = np.asarray(sorted(map(int, stability)))
        overlap = np.asarray([stability[str(count)]["mode_span_overlap_fraction"] for count in counts])
        axes[1, 1].plot(counts, 100 * overlap, "o-", color=colors[scheme], label=labels[scheme])
    axes[1, 1].set(
        title="Reproducibility across disjoint training sets",
        xlabel="Population-mode span dimension",
        ylabel="Shared subspace fraction (%)",
        xticks=[1, 2, 4, 8, 16],
    )
    axes[1, 1].legend(frameon=False)
    figure.suptitle("Unrestricted QK structure is real, medium-rank, and multiscale", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
