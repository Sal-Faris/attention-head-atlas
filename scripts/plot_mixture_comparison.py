"""Plot cross-validated hard-cluster, sparse-mixture, and PCA reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    components = result["component_counts"]
    model_styles = {
        "kmeans": ("Hard cluster (one centroid)", "o", "0.35"),
        "dictionary_2": ("Sparse mixture (2 atoms)", "s", "tab:blue"),
        "dictionary_4": ("Sparse mixture (4 atoms)", "^", "tab:orange"),
        "dictionary_8": ("Sparse mixture (8 atoms)", "D", "tab:green"),
        "pca": ("Dense linear factors (PCA)", "x", "tab:red"),
    }

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, view_name in zip(axes, ("QK", "OV", "JOINT"), strict=True):
        view = result["views"][view_name]
        for model, (label, marker, color) in model_styles.items():
            values = [view[str(count)]["variance_recovered"][model] for count in components]
            axis.plot(
                components,
                values,
                marker=marker,
                color=color,
                linewidth=2,
                label=label,
            )
        axis.axhline(0.0, color="0.7", linewidth=0.8)
        axis.set_xticks(components)
        axis.set_xlabel("Learned atoms / clusters / components")
        axis.set_ylabel("Held-out operator variance recovered")
        axis.set_title(view_name)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Sparse combinations generalize better than discrete head clusters\n"
        "Six-fold head-level cross-validation; same exact operator geometry",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"saved mixture comparison to {args.output}")


if __name__ == "__main__":
    main()
