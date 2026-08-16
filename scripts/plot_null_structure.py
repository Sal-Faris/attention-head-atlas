"""Plot real population-structure summaries against spectrum-matched nulls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    audits = {"QK": load_json(args.qk_input), "OV": load_json(args.ov_input)}
    metrics = (
        ("participation_dimension", "Effective population dimension", "lower = concentrated"),
        ("top_10_variance", "Variance in first 10 axes", "higher = concentrated"),
        ("dimensions_for_90_percent", "Axes needed for 90% variance", "lower = concentrated"),
    )
    colors = {"QK": "tab:blue", "OV": "tab:orange"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    rng = np.random.default_rng(0)
    for axis, (metric, title, direction) in zip(axes, metrics, strict=True):
        for position, kind in enumerate(("QK", "OV")):
            audit = audits[kind]
            real = audit["real_structure_summary"][metric]
            null_values = np.asarray(
                [summary[metric] for summary in audit["null_structure_summaries"]]
            )
            jitter = rng.uniform(-0.09, 0.09, size=len(null_values))
            axis.scatter(
                position + jitter,
                null_values,
                color="0.68",
                alpha=0.65,
                s=28,
                label="Spectrum-matched null" if position == 0 else None,
            )
            axis.scatter(
                position,
                real,
                marker="D",
                color=colors[kind],
                edgecolors="white",
                linewidths=0.8,
                s=100,
                zorder=3,
                label="Real operators" if position == 0 else None,
            )
        axis.set_xticks([0, 1], ["QK", "OV"])
        axis.set_title(f"{title}\n({direction})")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Real GPT-2 head populations share directions absent from independent random subspaces\n"
        "20 null populations preserve every head's singular values",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"saved null-structure comparison to {args.output}")


if __name__ == "__main__":
    main()
