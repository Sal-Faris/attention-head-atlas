"""Plot PCoA projections and variance spectra for QK and OV distances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from scipy.stats import spearmanr

from head_atlas.embedding import classical_mds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--qk-statistics", type=Path, required=True)
    parser.add_argument("--ov-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    return parser.parse_args()


def load_distance_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
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
        raise ValueError(f"metadata in {path} does not match the distance matrix")
    if metric != "normalized_frobenius":
        raise ValueError(f"unsupported metric in {path}: {metric}")
    if np.unique(kinds).size != 1:
        raise ValueError(f"operator kinds in {path} must be uniform")

    return distances, layers, heads, str(kinds[0])


def load_diagnostics(
    path: Path,
    expected_layers: np.ndarray,
    expected_heads: np.ndarray,
    expected_kind: str,
) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    records: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        location = (int(row["layer"]), int(row["head"]))
        if location in records:
            raise ValueError(f"duplicate head L{location[0]}H{location[1]} in {path}")
        if row["kind"] != expected_kind:
            raise ValueError(f"unexpected operator kind in {path}: {row['kind']}")
        records[location] = row

    ordered_rows = []
    for layer, head in zip(expected_layers, expected_heads, strict=True):
        location = (int(layer), int(head))
        if location not in records:
            raise ValueError(f"head L{location[0]}H{location[1]} is missing from {path}")
        ordered_rows.append(records[location])
    if len(records) != len(ordered_rows):
        raise ValueError(f"{path} contains heads absent from the distance bundle")

    return {
        "effective_rank": np.asarray(
            [float(row["effective_rank"]) for row in ordered_rows], dtype=np.float64
        ),
        "top_1_energy": np.asarray(
            [float(row["top_1_energy"]) for row in ordered_rows], dtype=np.float64
        ),
    }


def plot_embedding(
    axis: plt.Axes,
    coordinates: np.ndarray,
    explained_variance: np.ndarray,
    layers: np.ndarray,
    heads: np.ndarray,
    kind: str,
    color_map: plt.Colormap,
    color_norm: BoundaryNorm,
) -> None:
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=layers,
        cmap=color_map,
        norm=color_norm,
        s=42,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.35,
    )
    for coordinate, layer, head in zip(coordinates, layers, heads, strict=True):
        axis.annotate(
            f"L{layer}H{head}",
            coordinate,
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=4.5,
            alpha=0.65,
        )

    axis.axhline(0.0, color="0.8", linewidth=0.7, zorder=0)
    axis.axvline(0.0, color="0.8", linewidth=0.7, zorder=0)
    axis.set_xlabel(f"PCoA 1 ({100 * explained_variance[0]:.1f}% of variance)")
    axis.set_ylabel(f"PCoA 2 ({100 * explained_variance[1]:.1f}% of variance)")
    axis.set_title(f"{kind}: two-dimensional projection")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.15)


def plot_variance_spectrum(
    axis: plt.Axes,
    explained_variance: np.ndarray,
    kind: str,
) -> int:
    positive_variance = explained_variance[explained_variance > 0]
    cumulative = np.cumsum(positive_variance)
    dimensions = np.arange(1, len(cumulative) + 1)
    dimensions_90 = int(np.searchsorted(cumulative, 0.9) + 1)

    axis.plot(dimensions, cumulative, color="tab:blue", linewidth=2)
    axis.scatter([2], [cumulative[1]], color="tab:orange", zorder=3, label="2D projection")
    axis.axhline(0.9, color="0.35", linestyle="--", linewidth=1, label="90% threshold")
    axis.axvline(dimensions_90, color="0.35", linestyle=":", linewidth=1)
    axis.set_xlim(1, len(cumulative))
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Number of PCoA dimensions retained")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_title(f"{kind}: {dimensions_90} dimensions needed for 90%")
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    return dimensions_90


def plot_diagnostic_embedding(
    figure: plt.Figure,
    axis: plt.Axes,
    coordinates: np.ndarray,
    explained_variance: np.ndarray,
    layers: np.ndarray,
    heads: np.ndarray,
    values: np.ndarray,
    kind: str,
    diagnostic_label: str,
) -> None:
    points = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values,
        cmap="viridis",
        s=46,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.35,
    )
    for coordinate, layer, head in zip(coordinates, layers, heads, strict=True):
        axis.annotate(
            f"L{layer}H{head}",
            coordinate,
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=4.5,
            alpha=0.6,
        )

    color_bar = figure.colorbar(points, ax=axis, shrink=0.82)
    color_bar.set_label(diagnostic_label)
    axis.axhline(0.0, color="0.8", linewidth=0.7, zorder=0)
    axis.axvline(0.0, color="0.8", linewidth=0.7, zorder=0)
    axis.set_xlabel(f"PCoA 1 ({100 * explained_variance[0]:.1f}%)")
    axis.set_ylabel(f"PCoA 2 ({100 * explained_variance[1]:.1f}%)")
    axis.set_title(f"{kind}: coloured by {diagnostic_label.lower()}")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.15)


def print_diagnostic_correlations(
    coordinates: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    kind: str,
) -> None:
    for name, values in diagnostics.items():
        correlations = [
            float(spearmanr(values, coordinates[:, component]).statistic)
            for component in range(2)
        ]
        print(
            f"{kind} {name} Spearman correlations with PCoA 1/2: "
            f"{correlations[0]:.3f}, {correlations[1]:.3f}"
        )


def main() -> None:
    args = parse_args()
    qk_distances, qk_layers, qk_heads, qk_kind = load_distance_bundle(args.qk_input)
    ov_distances, ov_layers, ov_heads, ov_kind = load_distance_bundle(args.ov_input)
    qk_result = classical_mds(qk_distances, dimensions=2)
    ov_result = classical_mds(ov_distances, dimensions=2)
    qk_diagnostics = load_diagnostics(
        args.qk_statistics, qk_layers, qk_heads, qk_kind
    )
    ov_diagnostics = load_diagnostics(
        args.ov_statistics, ov_layers, ov_heads, ov_kind
    )

    maximum_layer = int(max(np.max(qk_layers), np.max(ov_layers)))
    color_map = plt.get_cmap("turbo", maximum_layer + 1)
    color_norm = BoundaryNorm(np.arange(-0.5, maximum_layer + 1.5), color_map.N)

    figure, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    plot_embedding(
        axes[0, 0],
        qk_result["coordinates"],
        qk_result["explained_variance_ratio"],
        qk_layers,
        qk_heads,
        qk_kind,
        color_map,
        color_norm,
    )
    plot_embedding(
        axes[0, 1],
        ov_result["coordinates"],
        ov_result["explained_variance_ratio"],
        ov_layers,
        ov_heads,
        ov_kind,
        color_map,
        color_norm,
    )
    qk_dimensions_90 = plot_variance_spectrum(
        axes[1, 0], qk_result["explained_variance_ratio"], qk_kind
    )
    ov_dimensions_90 = plot_variance_spectrum(
        axes[1, 1], ov_result["explained_variance_ratio"], ov_kind
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=color_norm, cmap=color_map)
    color_bar = figure.colorbar(
        scalar_mappable,
        ax=axes[0, :],
        ticks=np.arange(maximum_layer + 1),
        location="right",
        shrink=0.9,
    )
    color_bar.set_label("GPT-2 layer")
    figure.suptitle(
        "GPT-2-small attention-head geometry — PCoA of normalized Frobenius distances",
        fontsize=15,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)

    diagnostic_figure, diagnostic_axes = plt.subplots(
        2, 2, figsize=(15, 12), constrained_layout=True
    )
    diagnostic_specs = (
        ("effective_rank", "Effective rank"),
        ("top_1_energy", "Top singular-value energy"),
    )
    for column, (diagnostic_name, diagnostic_label) in enumerate(diagnostic_specs):
        plot_diagnostic_embedding(
            diagnostic_figure,
            diagnostic_axes[0, column],
            qk_result["coordinates"],
            qk_result["explained_variance_ratio"],
            qk_layers,
            qk_heads,
            qk_diagnostics[diagnostic_name],
            qk_kind,
            diagnostic_label,
        )
        plot_diagnostic_embedding(
            diagnostic_figure,
            diagnostic_axes[1, column],
            ov_result["coordinates"],
            ov_result["explained_variance_ratio"],
            ov_layers,
            ov_heads,
            ov_diagnostics[diagnostic_name],
            ov_kind,
            diagnostic_label,
        )
    diagnostic_figure.suptitle(
        "Do PCoA patterns track individual operator spectra?",
        fontsize=15,
    )
    args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_figure.savefig(args.diagnostic_output, dpi=200)
    plt.close(diagnostic_figure)

    print(f"saved PCoA prototype to {args.output}")
    print(f"saved diagnostic-coloured projections to {args.diagnostic_output}")
    print(
        f"2D variance: QK={100 * np.sum(qk_result['explained_variance_ratio'][:2]):.2f}%, "
        f"OV={100 * np.sum(ov_result['explained_variance_ratio'][:2]):.2f}%"
    )
    print(f"dimensions for 90% variance: QK={qk_dimensions_90}, OV={ov_dimensions_90}")
    print(
        "negative eigenvalue mass: "
        f"QK={qk_result['negative_eigenvalue_mass_ratio']:.3e}, "
        f"OV={ov_result['negative_eigenvalue_mass_ratio']:.3e}"
    )
    print_diagnostic_correlations(qk_result["coordinates"], qk_diagnostics, qk_kind)
    print_diagnostic_correlations(ov_result["coordinates"], ov_diagnostics, ov_kind)


if __name__ == "__main__":
    main()
