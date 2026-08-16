"""Plot how functional retrieval changes across operator representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frobenius-evaluation", type=Path, required=True)
    parser.add_argument("--subspace-evaluation", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_order() -> list[str]:
    return [
        "raw_frobenius",
        "singular_spectrum",
        "left_rank_8",
        "left_rank_16",
        "left_rank_32",
        "left_rank_64",
        "right_rank_8",
        "right_rank_16",
        "right_rank_32",
        "right_rank_64",
        "joint_rank_8",
        "joint_rank_16",
        "joint_rank_32",
        "joint_rank_64",
    ]


def metric_label(metric: str, kind: str) -> str:
    if metric == "raw_frobenius":
        return "Full normalized matrix"
    if metric == "singular_spectrum":
        return "Singular values only"
    side, _, rank = metric.split("_")
    side_names = {
        "QK": {"left": "Query", "right": "Key", "joint": "Query + key"},
        "OV": {"left": "Read", "right": "Write", "joint": "Read + write"},
    }
    return f"{side_names[kind][side]} subspace r={rank}"


def family_order_and_names(
    benchmark: dict[str, object],
) -> tuple[list[str], dict[str, str]]:
    families = [
        record
        for record in benchmark["families"]
        if record["use_for_primary_retrieval"]
        and record["inspection_status"] == "uninspected at benchmark freeze"
    ]
    return (
        [record["family_id"] for record in families],
        {record["family_id"]: record["display_name"] for record in families},
    )


def values_for_view(
    kind: str,
    frobenius: dict[str, object],
    subspace: dict[str, object],
    family_order: list[str],
) -> tuple[np.ndarray, list[float]]:
    rows = []
    p_values = []
    for metric in metric_order():
        if metric == "raw_frobenius":
            record = frobenius["views"][kind]
            retrieval = record["uninspected_families"]
            p_value = record["uninspected_layer_stratified_permutation"][
                "upper_tail_p_value"
            ]
        else:
            record = subspace["views"][kind][metric]
            retrieval = record["retrieval"]
            p_value = record["layer_stratified_permutation"]["upper_tail_p_value"]
        rows.append(
            [retrieval["families"][family_id]["mean_average_precision"] for family_id in family_order]
            + [retrieval["aggregate"]["mean_average_precision"]]
        )
        p_values.append(p_value)
    return np.asarray(rows), p_values


def plot_heatmap(
    axis: plt.Axes,
    kind: str,
    values: np.ndarray,
    p_values: list[float],
    family_order: list[str],
    display_names: dict[str, str],
) -> plt.AxesImage:
    image = axis.imshow(values, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(
        np.arange(len(family_order) + 1),
        [display_names[family_id] for family_id in family_order] + ["Family-balanced\naggregate"],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(metric_order())),
        [metric_label(metric, kind) for metric in metric_order()],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = f"{value:.2f}"
            if column == values.shape[1] - 1:
                label += "*" if p_values[row] <= 0.05 else ""
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value < 0.3 or value > 0.75 else "black",
            )
    axis.axhline(1.5, color="white", linewidth=1.5)
    axis.set_title(f"{kind} representations")
    return image


def main() -> None:
    args = parse_args()
    frobenius = load_json(args.frobenius_evaluation)
    subspace = load_json(args.subspace_evaluation)
    benchmark = load_json(args.benchmark)
    family_order, display_names = family_order_and_names(benchmark)

    figure, axes = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)
    image = None
    for axis, kind in zip(axes, ("QK", "OV"), strict=True):
        values, p_values = values_for_view(
            kind, frobenius, subspace, family_order
        )
        image = plot_heatmap(
            axis, kind, values, p_values, family_order, display_names
        )
    if image is None:
        raise RuntimeError("no representation panels were plotted")
    color_bar = figure.colorbar(image, ax=axes, shrink=0.75)
    color_bar.set_label("Mean average precision (1 = perfect retrieval)")
    figure.suptitle(
        "Which part of an attention operator carries published functional similarity?\n"
        "* aggregate exceeds a layer-stratified permutation null (exploratory comparison)",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"saved representation comparison to {args.output}")


if __name__ == "__main__":
    main()
