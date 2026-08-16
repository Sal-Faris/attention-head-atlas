"""Plot matched motif stability across initialization and bootstrap resampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    views = ["QK", "OV", "JOINT"]
    components = sorted(
        {int(value) for view in views for value in result["views"][view]}
    )
    conditions = (
        ("initialization_stability", "Different initializations", "tab:blue"),
        ("bootstrap_stability", "80% bootstrap samples", "tab:orange"),
        ("random_dictionary_baseline", "Random dictionaries", "0.65"),
    )
    labels = [f"{view}\nk={component}" for view in views for component in components]
    positions = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for offset, (key, label, color) in enumerate(conditions):
        means = []
        deviations = []
        for view in views:
            for component in components:
                summary = result["views"][view][str(component)][key]
                means.append(summary["mean"])
                deviations.append(summary["standard_deviation"])
        axis.bar(
            positions + (offset - 1) * width,
            means,
            width,
            yerr=deviations,
            capsize=3,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel("Optimally matched absolute atom cosine")
    axis.set_title(
        "Learned motif dictionaries are optimization-stable but only moderately sample-stable"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"saved dictionary stability figure to {args.output}")


if __name__ == "__main__":
    main()
