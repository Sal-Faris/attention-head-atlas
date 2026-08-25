"""Plot held-out compression and population use of shared rank-one QK motifs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/pythia-70m-deduped/shared_rank_one_qk_motifs_v1.json"),
    )
    parser.add_argument(
        "--motifs",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/shared_rank_one_qk_motifs_v1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/shared_rank_one_qk_motifs_v1.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.results.read_text(encoding="utf-8"))
    with np.load(args.motifs, allow_pickle=False) as bundle:
        coefficients = np.asarray(bundle["coefficients"])
        layers = np.asarray(bundle["layers"])
        heads = np.asarray(bundle["heads"])

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    colors = {"head_parity": "#3264a8", "layer_parity": "#d26a34"}
    labels = {"head_parity": "Unseen heads", "layer_parity": "Unseen layers"}
    for scheme in ("head_parity", "layer_parity"):
        section = report["reports"][scheme]
        counts = np.asarray(sorted(map(int, section["by_motif_count"])))
        recovered = np.asarray(
            [
                section["by_motif_count"][str(count)][
                    "rank_one_motif_full_variance_recovered"
                ]
                for count in counts
            ]
        )
        axes[0].plot(counts, 100 * recovered, "o-", color=colors[scheme], label=labels[scheme])
        matched = section["by_motif_count"][str(max(counts))][
            "parameter_matched_dense_pca_full_variance_recovered"
        ]
        axes[0].scatter(
            [max(counts)], [100 * matched], marker="x", s=80, color=colors[scheme]
        )
    axes[0].set(
        title="Held-out full-matrix reconstruction",
        xlabel="Shared rank-one motifs",
        ylabel="Variance recovered (%)",
        xticks=[4, 8, 16, 32],
    )
    axes[0].legend(frameon=False)
    axes[0].text(
        31,
        2.05,
        "× = equal shared-parameter\nfull-matrix dictionary",
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )

    absolute = np.abs(coefficients)
    effective_heads = (absolute.sum(axis=0) ** 2) / np.maximum(
        np.sum(absolute**2, axis=0), 1e-12
    )
    axes[1].hist(effective_heads, bins=np.arange(0, 26, 2), color="#5b9b76", edgecolor="white")
    axes[1].axvline(np.median(effective_heads), color="black", linestyle="--", linewidth=1)
    axes[1].set(
        title="How broadly each motif is reused",
        xlabel="Effective number of heads",
        ylabel="Motif count",
    )

    weighted_layer = (absolute.T @ layers) / np.maximum(absolute.sum(axis=0), 1e-12)
    motif_order = np.argsort(weighted_layer)
    normalized = absolute / np.maximum(absolute.max(axis=0, keepdims=True), 1e-12)
    image = axes[2].imshow(normalized[:, motif_order], aspect="auto", cmap="magma", vmin=0, vmax=1)
    boundaries = np.flatnonzero(np.diff(layers)) + 0.5
    for boundary in boundaries:
        axes[2].axhline(boundary, color="white", linewidth=0.5, alpha=0.7)
    axes[2].set(
        title="Motif use across the model",
        xlabel="Motifs (ordered by mean layer)",
        ylabel="Heads (ordered by layer)",
        yticks=np.arange(len(layers))[::4],
        yticklabels=[f"L{layer}H{head}" for layer, head in zip(layers, heads, strict=True)][::4],
    )
    figure.colorbar(image, ax=axes[2], label="Relative |coefficient|", fraction=0.046)
    figure.suptitle("Shared rank-one QK motifs: real but local and incomplete", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
