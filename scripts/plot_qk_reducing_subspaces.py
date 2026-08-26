"""Plot the end-to-end QK reducing-subspace experiment.

The figure separates exploratory multiresolution selection from the frozen
real/null comparison and makes the strongest temporal null visually explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

DEFAULT_INPUT = Path("results/pythia-70m-deduped/qk_reducing_subspaces_v1.json")
DEFAULT_OUTPUT = Path("results/pythia-70m-deduped/qk_reducing_subspaces_v1.png")
NULL_NAMES = (
    "independent_spectrum_haar",
    "within_layer_side_trajectory_pairing",
    "smooth_singular_frame_trajectory",
)
NULL_LABELS = ("Spectrum-Haar", "Layer-paired sides", "Smooth frames")
NULL_COLORS = ("#0072B2", "#E69F00", "#7A5195")
SPLIT_NAMES = ("primary", "late_sensitivity")
SPLIT_LABELS = ("Primary", "Late sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing {context}.{key}")
    return mapping[key]


def validate_report(report: dict[str, Any]) -> None:
    """Fail early when a report cannot support the required panels."""

    if report.get("schema") != "qk-reducing-subspaces-v1":
        raise ValueError("expected qk-reducing-subspaces-v1 report")
    protocol = _require(report, "protocol", "report")
    grid = _require(protocol, "grid", "protocol")
    if not isinstance(grid, list) or not grid:
        raise ValueError("protocol.grid must be a nonempty list")
    real = _require(report, "real", "report")
    primary = _require(_require(real, "splits", "real"), "primary", "real.splits")
    population = _require(primary, "population", "real.splits.primary")
    if any(key not in population for key in grid):
        raise ValueError("primary population does not contain every grid configuration")
    for null_name in NULL_NAMES:
        _require(_require(report, "nulls", "report"), null_name, "nulls")


def _config_label(key: str, report: dict[str, Any]) -> str:
    population = report["real"]["splits"]["primary"]["population"][key]
    # Stable IDs have the form d064_p032_q032. Parsing the ID keeps plotting
    # independent of per-head ordering and avoids hardcoding the frozen grid.
    try:
        dimension, output_rank, input_rank = (int(part[1:]) for part in key.split("_"))
    except (ValueError, IndexError) as error:
        raise ValueError(f"invalid configuration id: {key}") from error
    if "validation" not in population or "confirmation" not in population:
        raise ValueError(f"configuration {key} has incomplete phases")
    return f"d={dimension}\np={output_rank}, q={input_rank}"


def _population_metric(
    report: dict[str, Any], config: str, phase: str, metric: str
) -> tuple[float, float]:
    summary = report["real"]["splits"]["primary"]["population"][config][phase][metric]
    return float(summary["mean"]), float(summary["standard_deviation"])


def _comparison(report: dict[str, Any], null_name: str, split: str, metric: str) -> dict[str, Any]:
    return report["nulls"][null_name]["comparisons"][split][metric]


def decision_matrix(report: dict[str, Any], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Return split-level and strict-both-splits decision gates.

    A gate passes only for a positive real-minus-null effect and a finite-null
    upper-tail p-value no greater than ``alpha``. The last row is the strict
    conjunction across both frozen temporal splits.
    """

    passed = np.zeros((3, len(NULL_NAMES)), dtype=bool)
    p_values = np.full((2, len(NULL_NAMES)), np.nan, dtype=np.float64)
    metric = "confirmation_gain_over_random"
    for row, split in enumerate(SPLIT_NAMES):
        for column, null_name in enumerate(NULL_NAMES):
            item = _comparison(report, null_name, split, metric)
            effect = float(item["observed_minus_null_mean"])
            p_value = float(item["upper_tail_p_value"])
            passed[row, column] = effect > 0.0 and p_value <= alpha
            p_values[row, column] = p_value
    passed[2] = np.all(passed[:2], axis=0)
    return passed, p_values


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _panel_multiresolution(axis: plt.Axes, report: dict[str, Any]) -> None:
    grid = report["protocol"]["grid"]
    x = np.arange(len(grid), dtype=float)
    validation = np.asarray(
        [_population_metric(report, key, "validation", "gain_over_random")[0] for key in grid]
    )
    confirmation = np.asarray(
        [_population_metric(report, key, "confirmation", "gain_over_random")[0] for key in grid]
    )
    fixed = report["protocol"]["fixed_primary"]
    selected = report["real"]["analysis_roles"]["validation_selected"]["configuration"]
    if fixed not in grid or selected not in grid:
        raise ValueError("fixed or validation-selected configuration is absent from grid")
    fixed_index, selected_index = grid.index(fixed), grid.index(selected)

    axis.axvspan(fixed_index - 0.42, fixed_index + 0.42, color="#56B4E9", alpha=0.13)
    axis.axvspan(selected_index - 0.33, selected_index + 0.33, color="#CC79A7", alpha=0.15)
    axis.plot(x, validation, "o-", color="#D55E00", linewidth=1.8, label="Validation")
    axis.plot(x, confirmation, "s-", color="#009E73", linewidth=1.8, label="Confirmation")
    axis.scatter(
        [selected_index],
        [validation[selected_index]],
        marker="*",
        color="#8E0152",
        edgecolor="white",
        linewidth=0.8,
        s=180,
        zorder=5,
        label="Validation-selected",
    )
    axis.scatter(
        [fixed_index],
        [confirmation[fixed_index]],
        marker="D",
        facecolor="none",
        edgecolor="#006D8F",
        linewidth=1.8,
        s=75,
        zorder=5,
        label="Fixed primary",
    )
    axis.axhline(0.0, color="#444444", linewidth=0.9)
    axis.set_xticks(x, [_config_label(key, report) for key in grid], fontsize=7)
    axis.set_ylabel("Gain over dimension-matched random projectors")
    axis.set_title(
        "A  Multiresolution audit (selection and confirmation kept separate)", loc="left"
    )
    axis.legend(frameon=False, ncol=2, fontsize=8)
    _style_axis(axis)


def _boxplot(
    axis: plt.Axes,
    values: list[float],
    position: float,
    color: str,
    width: float,
) -> None:
    box = axis.boxplot(
        values,
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.1},
        whiskerprops={"color": color},
        capprops={"color": color},
    )
    box["boxes"][0].set(facecolor=color, edgecolor=color, alpha=0.42)


def _panel_fixed_nulls(axis: plt.Axes, report: dict[str, Any]) -> None:
    categories: list[tuple[str, str, str]] = []
    for split, split_label in zip(SPLIT_NAMES, SPLIT_LABELS, strict=True):
        categories.extend(
            [
                (split, "confirmation_active_support_energy_fraction", f"{split_label}\nsupport"),
                (split, "confirmation_gain_over_random", f"{split_label}\ngain"),
            ]
        )
    centers = np.arange(len(categories), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(NULL_NAMES))
    for center, (split, metric, _) in zip(centers, categories, strict=True):
        observed = report["confirmatory_observed"][split]["population"][metric]["mean"]
        for offset, null_name, color in zip(offsets, NULL_NAMES, NULL_COLORS, strict=True):
            values = _comparison(report, null_name, split, metric)["null_population_means"]
            _boxplot(axis, [float(value) for value in values], center + offset, color, 0.17)
        axis.scatter(
            center,
            observed,
            marker="*",
            s=125,
            color="#111111",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=label)
        for color, label in zip(NULL_COLORS, NULL_LABELS, strict=True)
    ]
    handles.append(
        plt.Line2D([0], [0], marker="*", linestyle="", color="#111111", label="Real mean")
    )
    axis.axhline(0.0, color="#444444", linewidth=0.9)
    axis.set_xticks(centers, [label for _, _, label in categories])
    axis.set_ylabel("Fraction / gain (frozen primary configuration)")
    axis.set_title("B  Fixed-primary confirmation against matched nulls", loc="left")
    axis.legend(handles=handles, frameon=False, fontsize=8, ncol=2)
    _style_axis(axis)


def _smooth_adjusted_by_layer(report: dict[str, Any], split: str) -> dict[int, list[float]]:
    real_items = report["confirmatory_observed"][split]["per_head"]
    repetitions = report["nulls"][NULL_NAMES[2]]["repetitions"]
    null_by_key: dict[str, list[float]] = {}
    for repetition in repetitions:
        for item in repetition[split]["per_head"]:
            null_by_key.setdefault(item["key"], []).append(
                float(item["confirmation_gain_over_random"])
            )
    result: dict[int, list[float]] = {}
    for item in real_items:
        key = item["key"]
        if key not in null_by_key:
            raise ValueError(f"smooth null is missing head {key} in {split}")
        adjusted = float(item["confirmation_gain_over_random"]) - float(np.mean(null_by_key[key]))
        result.setdefault(int(item["layer"]), []).append(adjusted)
    return result


def _panel_layer_adjusted(axis: plt.Axes, report: dict[str, Any]) -> None:
    grouped = {split: _smooth_adjusted_by_layer(report, split) for split in SPLIT_NAMES}
    layers = sorted(set().union(*(values.keys() for values in grouped.values())))
    colors = ("#0072B2", "#D55E00")
    offsets = (-0.16, 0.16)
    for split, label, color, offset in zip(SPLIT_NAMES, SPLIT_LABELS, colors, offsets, strict=True):
        for index, layer in enumerate(layers):
            values = grouped[split].get(layer, [])
            if values:
                _boxplot(axis, values, index + offset, color, 0.27)
        axis.plot([], [], color=color, linewidth=7, alpha=0.42, label=label)
    axis.axhline(0.0, color="#111111", linewidth=1.0)
    axis.set_xticks(np.arange(len(layers)), [f"L{layer}" for layer in layers])
    axis.set_ylabel("Real − mean smooth-null confirmation gain")
    axis.set_title("C  Residual evidence after the strongest temporal null", loc="left")
    axis.legend(frameon=False, ncol=2)
    _style_axis(axis)


def _stability_values(report: dict[str, Any]) -> tuple[list[str], list[list[float]]]:
    real = [
        float(item["stability_overlap"])
        for item in report["confirmatory_observed"]["primary"]["per_head"]
    ]
    values = [real]
    for null_name in NULL_NAMES:
        pooled = [
            float(item["stability_overlap"])
            for repetition in report["nulls"][null_name]["repetitions"]
            for item in repetition["primary"]["per_head"]
        ]
        values.append(pooled)
    return ["Real", *NULL_LABELS], values


def _panel_stability(axis: plt.Axes, report: dict[str, Any]) -> None:
    labels, values = _stability_values(report)
    colors = ("#222222", *NULL_COLORS)
    violin = axis.violinplot(values, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violin["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    for position, (samples, color) in enumerate(zip(values, colors, strict=True), start=1):
        quartiles = np.quantile(samples, [0.25, 0.5, 0.75])
        axis.vlines(position, quartiles[0], quartiles[2], color=color, linewidth=5)
        axis.scatter(position, quartiles[1], color="white", edgecolor=color, s=24, zorder=4)
    axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=18, ha="right")
    axis.set_ylabel("Disjoint-checkpoint reducing-pair overlap")
    axis.set_title("D  Stability (null head values pooled across repetitions)", loc="left")
    _style_axis(axis)


def _panel_gate(axis: plt.Axes, report: dict[str, Any]) -> None:
    passed, p_values = decision_matrix(report)
    image = axis.imshow(
        passed.astype(int), cmap=ListedColormap(["#BDBDBD", "#009E73"]), vmin=0, vmax=1
    )
    image.set_rasterized(True)
    for row in range(passed.shape[0]):
        for column in range(passed.shape[1]):
            if row < 2:
                p_value = p_values[row, column]
                text = f"{'PASS' if passed[row, column] else 'not passed'}\np={p_value:.3g}"
            else:
                text = "PASS" if passed[row, column] else "not passed"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if passed[row, column] else "#202020",
                fontsize=8,
                fontweight="bold" if row == 2 else "normal",
            )
    axis.set_xticks(np.arange(3), ["Haar", "Paired", "Smooth"])
    axis.set_yticks(np.arange(3), [*SPLIT_LABELS, "Strict: both"])
    axis.set_title("E  Decision gate: held-out gain", loc="left")
    axis.set_xlabel("Positive effect and finite-null p ≤ 0.05")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def create_figure(report: dict[str, Any]) -> plt.Figure:
    """Create the five-panel research figure from an in-memory report."""

    validate_report(report)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "axes.edgecolor": "#404040",
            "xtick.color": "#303030",
            "ytick.color": "#303030",
        }
    )
    figure = plt.figure(figsize=(18, 13.5), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(1.05, 1.0, 1.0))
    _panel_multiresolution(figure.add_subplot(grid[0, :2]), report)
    _panel_gate(figure.add_subplot(grid[0, 2]), report)
    _panel_fixed_nulls(figure.add_subplot(grid[1, :2]), report)
    _panel_stability(figure.add_subplot(grid[1, 2]), report)
    _panel_layer_adjusted(figure.add_subplot(grid[2, :]), report)
    figure.suptitle(
        "QK reducing compartments: apparent reuse versus temporal matched nulls",
        fontsize=16,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        -0.018,
        (
            "Weight-only analysis. Selection uses validation checkpoints; fixed-primary null "
            "tests use held-out confirmation checkpoints. Failure against smooth-frame nulls "
            "means temporal coherence is sufficient; individual heads remain candidates only."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#404040",
    )
    return figure


def plot_report(report: dict[str, Any], output: str | Path) -> Path:
    """Render and save a report, returning its output path."""

    output_path = Path(output)
    figure = create_figure(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    output = plot_report(report, args.output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
