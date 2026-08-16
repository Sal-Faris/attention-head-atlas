"""Plot exploratory QK and OV dendrograms from labeled distance bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import squareform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_distance_bundle(path: Path) -> tuple[np.ndarray, list[str], np.ndarray, str]:
    with np.load(path, allow_pickle=False) as bundle:
        distances = np.asarray(bundle["distances"], dtype=np.float64)
        layers = np.asarray(bundle["layers"], dtype=np.int64)
        heads = np.asarray(bundle["heads"], dtype=np.int64)
        kinds = np.asarray(bundle["kinds"])
        metric = str(bundle["metric"].item())

    item_count = distances.shape[0]
    if distances.shape != (item_count, item_count):
        raise ValueError(f"distance matrix in {path} must be square")
    if len(layers) != item_count or len(heads) != item_count or len(kinds) != item_count:
        raise ValueError(f"labels in {path} do not match the distance matrix")
    if not np.isfinite(distances).all():
        raise ValueError(f"distance matrix in {path} contains non-finite values")
    if not np.allclose(distances, distances.T, rtol=0.0, atol=1e-10):
        raise ValueError(f"distance matrix in {path} must be symmetric")
    if not np.allclose(np.diag(distances), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"distance matrix in {path} must have a zero diagonal")
    if metric != "normalized_frobenius":
        raise ValueError(f"unsupported metric in {path}: {metric}")
    if np.unique(kinds).size != 1:
        raise ValueError(f"operator kinds in {path} must be uniform")

    labels = [f"L{layer}H{head}" for layer, head in zip(layers, heads, strict=True)]
    return distances, labels, layers, str(kinds[0])


def plot_dendrogram(
    axis: plt.Axes,
    distances: np.ndarray,
    labels: list[str],
    layers: np.ndarray,
    kind: str,
    layer_colors: list[tuple[float, float, float, float]],
) -> float:
    condensed = squareform(distances, checks=False)
    hierarchy = linkage(condensed, method="average", optimal_ordering=True)
    correlation, _ = cophenet(hierarchy, condensed)
    dendrogram(
        hierarchy,
        labels=labels,
        leaf_font_size=5,
        color_threshold=0,
        above_threshold_color="0.25",
        ax=axis,
    )

    layer_by_label = dict(zip(labels, layers, strict=True))
    for tick_label in axis.get_xticklabels():
        tick_label.set_color(layer_colors[layer_by_label[tick_label.get_text()]])

    axis.set_title(
        f"{kind} exploratory average-linkage dendrogram "
        f"(cophenetic correlation = {correlation:.3f})"
    )
    axis.set_ylabel("Merge distance")
    axis.set_xlabel("Head (label color indicates layer)")
    axis.grid(axis="y", alpha=0.2)
    return float(correlation)


def main() -> None:
    args = parse_args()
    qk_distances, qk_labels, qk_layers, qk_kind = load_distance_bundle(args.qk_input)
    ov_distances, ov_labels, ov_layers, ov_kind = load_distance_bundle(args.ov_input)

    maximum_layer = int(max(np.max(qk_layers), np.max(ov_layers)))
    color_map = plt.get_cmap("tab20")
    layer_colors = [color_map(index / max(maximum_layer, 1)) for index in range(maximum_layer + 1)]

    figure, axes = plt.subplots(2, 1, figsize=(24, 14), constrained_layout=True)
    qk_correlation = plot_dendrogram(
        axes[0], qk_distances, qk_labels, qk_layers, qk_kind, layer_colors
    )
    ov_correlation = plot_dendrogram(
        axes[1], ov_distances, ov_labels, ov_layers, ov_kind, layer_colors
    )

    legend_handles = [
        Line2D([0], [0], color=layer_colors[layer], marker="o", linestyle="", label=f"Layer {layer}")
        for layer in range(maximum_layer + 1)
    ]
    figure.legend(handles=legend_handles, loc="outside lower center", ncol=6, title="GPT-2 layer")
    figure.suptitle(
        "GPT-2-small attention-head operator geometry — provisional, no cluster cut",
        fontsize=16,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(f"saved dendrogram prototype to {args.output}")
    print(f"cophenetic correlations: QK={qk_correlation:.6f}, OV={ov_correlation:.6f}")


if __name__ == "__main__":
    main()
