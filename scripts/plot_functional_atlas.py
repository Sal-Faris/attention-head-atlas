"""Visualize published functional families in QK, OV, and joint geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.distances import weighted_product_distances
from head_atlas.embedding import classical_mds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_distance_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        distances = np.asarray(bundle["distances"], dtype=np.float64)
        layers = np.asarray(bundle["layers"], dtype=np.int64)
        heads = np.asarray(bundle["heads"], dtype=np.int64)
        metric = str(bundle["metric"].item())
    if distances.shape != (len(layers), len(layers)) or len(heads) != len(layers):
        raise ValueError(f"inconsistent distance bundle: {path}")
    if metric != "normalized_frobenius":
        raise ValueError(f"unsupported metric in {path}: {metric}")
    return distances, layers, heads


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def family_metadata(
    benchmark: dict[str, object],
) -> tuple[list[str], dict[tuple[int, int], str], dict[str, str]]:
    family_order: list[str] = []
    head_to_family: dict[tuple[int, int], str] = {}
    display_names: dict[str, str] = {}
    for family in benchmark["families"]:
        if not family["use_for_primary_retrieval"]:
            continue
        family_id = family["family_id"]
        family_order.append(family_id)
        display_names[family_id] = family["display_name"]
        for layer, head in family["primary_heads"]:
            location = (int(layer), int(head))
            if location in head_to_family:
                raise ValueError(f"head L{layer}H{head} belongs to multiple families")
            head_to_family[location] = family_id
    return family_order, head_to_family, display_names


def plot_family_projection(
    axis: plt.Axes,
    distances: np.ndarray,
    layers: np.ndarray,
    heads: np.ndarray,
    title: str,
    head_to_family: dict[tuple[int, int], str],
    family_colors: dict[str, object],
) -> None:
    embedding = classical_mds(distances, dimensions=2)
    coordinates = embedding["coordinates"]
    variance = embedding["explained_variance_ratio"]
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        color="0.82",
        s=22,
        alpha=0.55,
        linewidths=0,
        label="Unlabelled (not negatives)",
    )
    for index, (layer, head) in enumerate(zip(layers, heads, strict=True)):
        location = (int(layer), int(head))
        family_id = head_to_family.get(location)
        if family_id is None:
            continue
        axis.scatter(
            coordinates[index, 0],
            coordinates[index, 1],
            color=family_colors[family_id],
            s=72,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        axis.annotate(
            f"L{layer}H{head}",
            coordinates[index],
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
            weight="semibold",
            zorder=4,
        )
    axis.axhline(0.0, color="0.88", linewidth=0.7, zorder=0)
    axis.axvline(0.0, color="0.88", linewidth=0.7, zorder=0)
    axis.set_xlabel(f"PCoA 1 ({100 * variance[0]:.1f}% variance)")
    axis.set_ylabel(f"PCoA 2 ({100 * variance[1]:.1f}% variance)")
    axis.set_title(f"{title} (2D shows {100 * np.sum(variance[:2]):.1f}%)")
    axis.grid(alpha=0.12)


def plot_family_retrieval(
    axis: plt.Axes,
    evaluation: dict[str, object],
    family_order: list[str],
    display_names: dict[str, str],
) -> None:
    views = ["QK", "OV", "JOINT"]
    values = np.asarray(
        [
            [
                evaluation["views"][view]["all_primary_families_descriptive"][
                    "families"
                ][family_id]["mean_average_precision"]
                for view in views
            ]
            for family_id in family_order
        ]
    )
    image = axis.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(views)), views)
    axis.set_yticks(
        np.arange(len(family_order)),
        [display_names[family_id] for family_id in family_order],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "white" if value < 0.25 or value > 0.72 else "black"
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    axis.set_title("Full-space family retrieval (mean average precision)")
    color_bar = axis.figure.colorbar(image, ax=axis, shrink=0.82)
    color_bar.set_label("mAP (1 = perfect)")


def plot_null_comparison(axis: plt.Axes, evaluation: dict[str, object]) -> None:
    views = ["QK", "OV", "JOINT"]
    observed = []
    null_means = []
    null_deviations = []
    p_values = []
    for view in views:
        result = evaluation["views"][view]["uninspected_layer_stratified_permutation"]
        observed.append(result["observed_family_balanced_map"])
        null_means.append(result["null_mean"])
        null_deviations.append(result["null_standard_deviation"])
        p_values.append(result["upper_tail_p_value"])

    positions = np.arange(len(views))
    width = 0.34
    axis.bar(positions - width / 2, observed, width, color="tab:blue", label="Observed")
    axis.bar(
        positions + width / 2,
        null_means,
        width,
        yerr=null_deviations,
        capsize=4,
        color="0.7",
        label="Layer-stratified null (mean ± SD)",
    )
    for position, value, p_value in zip(positions, observed, p_values, strict=True):
        axis.text(position - width / 2, value + 0.018, f"p={p_value:.4f}", ha="center", fontsize=7)
    axis.set_xticks(positions, views)
    axis.set_ylim(0.0, max(observed) + 0.11)
    axis.set_ylabel("Family-balanced mAP")
    axis.set_title("External labels retrieve after matching layer\n(copy pair excluded)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8)


def main() -> None:
    args = parse_args()
    qk_distances, qk_layers, qk_heads = load_distance_bundle(args.qk_input)
    ov_distances, ov_layers, ov_heads = load_distance_bundle(args.ov_input)
    if not np.array_equal(qk_layers, ov_layers) or not np.array_equal(qk_heads, ov_heads):
        raise ValueError("QK and OV bundles must contain heads in the same order")
    joint_distances = weighted_product_distances([qk_distances, ov_distances])
    benchmark = load_json(args.benchmark)
    evaluation = load_json(args.evaluation)
    family_order, head_to_family, display_names = family_metadata(benchmark)
    colors = plt.get_cmap("tab10")
    family_colors = {
        family_id: colors(index) for index, family_id in enumerate(family_order)
    }

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for axis, distances, title in zip(
        axes[0],
        (qk_distances, ov_distances, joint_distances),
        ("QK", "OV", "Equal-weight QK + OV"),
        strict=True,
    ):
        plot_family_projection(
            axis,
            distances,
            qk_layers,
            qk_heads,
            title,
            head_to_family,
            family_colors,
        )

    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=family_colors[family_id],
            label=display_names[family_id],
            markersize=7,
        )
        for family_id in family_order
    ]
    axes[1, 0].axis("off")
    axes[1, 0].legend(
        handles=legend_handles,
        loc="center",
        title="Published functional families\n(primary validated members)",
        frameon=False,
        fontsize=10,
    )
    plot_family_retrieval(axes[1, 1], evaluation, family_order, display_names)
    plot_null_comparison(axes[1, 2], evaluation)
    figure.suptitle(
        "GPT-2-small attention operators: weak global clusters, strong local functional retrieval",
        fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"saved functional atlas to {args.output}")


if __name__ == "__main__":
    main()
