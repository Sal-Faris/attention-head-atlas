"""Audit how much reusable intrinsic cores can compress complete head operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_singular_components
from head_atlas.intrinsic_mdl import (
    gaussian_factor_spectra,
    normalize_spectra,
    parity_splits,
    profile_reconstruction,
    rank_description,
    rank_energy_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument("--initial-revision", default="step0")
    parser.add_argument("--final-revision", default="step143000")
    parser.add_argument("--null-repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument(
        "--shared-operator-results",
        type=Path,
        default=Path("results/pythia-70m-deduped/shared_operator_compression_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/intrinsic_core_mdl_audit_v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/intrinsic_core_mdl_audit_v1.png"),
    )
    parser.add_argument(
        "--profile-figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/intrinsic_core_profiles_v1.png"),
    )
    return parser.parse_args()


def spectra_from_bundle(path: Path) -> tuple[list[object], np.ndarray, dict[str, object]]:
    operators, metadata = load_factor_bundle(path)
    spectra = np.stack([factorized_singular_components(operator)[1] for operator in operators])
    return operators, normalize_spectra(spectra), metadata


def profile_report(
    operators: list[object], spectra: np.ndarray, components: list[int]
) -> dict[str, dict[str, dict[str, float]]]:
    heads = np.asarray([operator.head for operator in operators], dtype=np.int64)
    layers = np.asarray([operator.layer for operator in operators], dtype=np.int64)
    result = {}
    for name, labels in (("held_out_heads", heads), ("held_out_layers", layers)):
        absolute = profile_reconstruction(spectra, parity_splits(labels), components)
        baseline_error = max(1.0 - absolute[0], 1e-15)
        variation = {
            component: (absolute[component] - absolute[0]) / baseline_error
            for component in components
        }
        result[name] = {
            "absolute_energy_recovered": {
                str(key): value for key, value in absolute.items()
            },
            "fraction_of_profile_variation_recovered": {
                str(key): value for key, value in variation.items()
            },
        }
    return result


def random_factor_null(
    count: int,
    width: int,
    rank: int,
    heads: np.ndarray,
    components: list[int],
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    absolute_samples: dict[int, list[float]] = {component: [] for component in components}
    variation_samples: dict[int, list[float]] = {component: [] for component in components}
    for repetition in range(repetitions):
        spectra = gaussian_factor_spectra(count, width, rank, rng)
        report = profile_reconstruction(spectra, parity_splits(heads), components)
        baseline_error = max(1.0 - report[0], 1e-15)
        for component, value in report.items():
            absolute_samples[component].append(value)
            variation_samples[component].append((value - report[0]) / baseline_error)
        print(f"Gaussian-factor null {repetition + 1}/{repetitions}", flush=True)
    return {
        str(component): {
            "absolute_energy_mean": float(np.mean(absolute_samples[component])),
            "absolute_energy_standard_deviation": float(np.std(absolute_samples[component])),
            "variation_recovered_mean": float(np.mean(variation_samples[component])),
            "variation_recovered_standard_deviation": float(
                np.std(variation_samples[component])
            ),
        }
        for component in components
    }


def synthetic_gate(rng: np.random.Generator) -> dict[str, object]:
    """Check that the conditional profile score detects a known shared core."""

    count, rank = 48, 16
    base = np.exp(-np.linspace(0.0, 2.5, rank))
    shared = np.maximum(base[None, :] * (1.0 + 0.01 * rng.standard_normal((count, rank))), 1e-8)
    shared = np.sort(shared, axis=1)[:, ::-1]
    heterogeneous = np.sort(rng.lognormal(0.0, 1.0, size=(count, rank)), axis=1)[:, ::-1]
    labels = np.tile(np.arange(8), 6)
    components = [0, 1, 2, 4]
    shared_report = profile_reconstruction(shared, parity_splits(labels), components)
    heterogeneous_report = profile_reconstruction(
        heterogeneous, parity_splits(labels), components
    )
    return {
        "shared_profile": {str(key): value for key, value in shared_report.items()},
        "heterogeneous_profile": {
            str(key): value for key, value in heterogeneous_report.items()
        },
        "gate": "mean-profile reconstruction must be higher for the shared-core population",
        "passed": bool(shared_report[0] > heterogeneous_report[0]),
    }


def mode_diagnostics(operators: list[object], spectra: np.ndarray) -> dict[str, object]:
    """Describe the leading learned directions of normalized spectral variation."""

    mean = np.mean(spectra, axis=0)
    centered = spectra - mean
    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ directions[:4].T
    layers = np.asarray([operator.layer for operator in operators], dtype=np.int64)
    heads = np.asarray([operator.head for operator in operators], dtype=np.int64)
    fractions = singular_values**2 / np.sum(singular_values**2)
    return {
        "mean_normalized_spectrum": mean.tolist(),
        "leading_directions": directions[:4].tolist(),
        "score_standard_deviations": np.std(scores, axis=0).tolist(),
        "explained_profile_variation": fractions[:8].tolist(),
        "score_layer_spearman": [
            float(spearmanr(scores[:, index], layers).statistic) for index in range(4)
        ],
        "scores": scores.tolist(),
        "layers": layers.tolist(),
        "heads": heads.tolist(),
    }


def plot_report(report: dict[str, object], output: Path) -> None:
    ranks = [int(value) for value in report["ranks"]]
    components = [int(value) for value in report["profile_components"]]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axis = axes[0, 0]
    for kind, color in (("QK", "tab:blue"), ("OV", "tab:orange")):
        values = report["views"][kind]["final"]["rank_energy"]
        axis.plot(ranks, [100 * values[str(rank)] for rank in ranks], marker="o", label=kind, color=color)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Per-head SVD rank")
    axis.set_ylabel("Mean operator energy recovered (%)")
    axis.set_title("A  Ordinary low-rank coverage", loc="left")
    axis.legend()

    axis = axes[0, 1]
    for kind, linestyle in (("QK", "-"), ("OV", "--")):
        for stage, alpha in (("initial", 0.45), ("final", 1.0)):
            values = report["views"][kind][stage]["profiles"]["held_out_heads"][
                "fraction_of_profile_variation_recovered"
            ]
            axis.plot(
                components[1:],
                [100 * values[str(component)] for component in components[1:]],
                marker="o",
                linestyle=linestyle,
                alpha=alpha,
                label=f"{kind} {stage}",
            )
    null_values = report["gaussian_factor_null"]
    axis.plot(
        components[1:],
        [100 * null_values[str(component)]["variation_recovered_mean"] for component in components[1:]],
        color="black",
        linestyle=":",
        marker="o",
        label="Gaussian-factor null",
    )
    axis.set_xlabel("PCA corrections to mean singular profile")
    axis.set_ylabel("Profile variation recovered (%)")
    axis.set_title("B  Training creates low-dimensional spectral variation", loc="left")
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    dimensions = report["description_accounting"]
    fractions = [100 * dimensions[str(rank)]["maximum_core_saving_fraction"] for rank in ranks]
    axis.plot(ranks, fractions, marker="o", color="tab:red")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Module rank")
    axis.set_ylabel("Maximum manifold-dimension saving (%)")
    axis.set_title("C  Reusing only intrinsic shape barely shortens the model", loc="left")

    axis = axes[1, 1]
    labels = ["QK", "OV"]
    conditional = [
        100
        * report["views"][kind]["final"]["profiles"]["held_out_heads"][
            "absolute_energy_recovered"
        ]["0"]
        for kind in labels
    ]
    complete = [
        100 * report["complete_operator_reference"][kind]
        for kind in labels
    ]
    positions = np.arange(len(labels))
    width = 0.36
    axis.bar(positions - width / 2, conditional, width, label="Core, true frames supplied")
    axis.bar(
        positions + width / 2,
        complete,
        width,
        label="Shared-support model, complete operator",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Held-out energy/variance recovered (%)")
    axis.set_title("D  Locating the subspaces is the hard part", loc="left")
    axis.legend(fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_profile_modes(report: dict[str, object], output: Path) -> None:
    """Plot the shapes and population scores of the leading spectral modes."""

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, kind in enumerate(("QK", "OV")):
        diagnostics = report["views"][kind]["final"]["mode_diagnostics"]
        mean = np.asarray(diagnostics["mean_normalized_spectrum"])
        directions = np.asarray(diagnostics["leading_directions"])
        scales = np.asarray(diagnostics["score_standard_deviations"])
        mode_variance = np.asarray(diagnostics["explained_profile_variation"])
        indices = np.arange(1, len(mean) + 1)

        axis = axes[0, column]
        axis.plot(indices, mean, color="black", linewidth=2, label="Population mean")
        colors = ("tab:blue", "tab:orange")
        for mode in range(2):
            for sign, linestyle in ((-1, "--"), (1, "-")):
                profile = np.maximum(mean + sign * 2 * scales[mode] * directions[mode], 1e-5)
                axis.plot(
                    indices,
                    profile,
                    color=colors[mode],
                    linestyle=linestyle,
                    label=(
                        f"PC{mode + 1} +/-2sd ({100 * mode_variance[mode]:.1f}%)"
                        if sign == 1
                        else None
                    ),
                )
        axis.set_yscale("log")
        axis.set_xlabel("Singular direction")
        axis.set_ylabel("Normalized singular value")
        axis.set_title(f"{chr(65 + column)}  {kind} learned profile shapes", loc="left")
        axis.legend(fontsize=8)

        axis = axes[1, column]
        scores = np.asarray(diagnostics["scores"])
        layers = np.asarray(diagnostics["layers"])
        scatter = axis.scatter(scores[:, 0], scores[:, 1], c=layers, cmap="viridis", s=45)
        axis.axhline(0, color="0.8", linewidth=1)
        axis.axvline(0, color="0.8", linewidth=1)
        axis.set_xlabel("Spectral PC1 score")
        axis.set_ylabel("Spectral PC2 score")
        correlation = diagnostics["score_layer_spearman"][0]
        axis.set_title(
            f"{chr(67 + column)}  {kind} heads (PC1-layer rho={correlation:.2f})",
            loc="left",
        )
        figure.colorbar(scatter, ax=axis, label="Layer")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    ranks = [1, 2, 4, 8, 16, 32, 64]
    components = [0, 1, 2, 4, 8, 16]
    views: dict[str, object] = {}
    common_metadata: dict[str, object] | None = None
    final_operators: list[object] | None = None

    for kind in ("QK", "OV"):
        stages = {}
        for label, revision in (("initial", args.initial_revision), ("final", args.final_revision)):
            path = args.artifact_root / revision / f"{kind.lower()}_factors.npz"
            operators, spectra, metadata = spectra_from_bundle(path)
            if common_metadata is None:
                common_metadata = metadata
            stages[label] = {
                "profiles": profile_report(operators, spectra, components),
                "rank_energy": {
                    str(key): value for key, value in rank_energy_curve(spectra, ranks).items()
                },
            }
            if label == "final":
                stages[label]["mode_diagnostics"] = mode_diagnostics(operators, spectra)
            if kind == "OV" and label == "final":
                final_operators = operators
        views[kind] = stages

    if final_operators is None or common_metadata is None:
        raise RuntimeError("final OV population was not loaded")
    null = random_factor_null(
        len(final_operators),
        int(common_metadata["d_model"]),
        int(common_metadata["d_head"]),
        np.asarray([operator.head for operator in final_operators]),
        components,
        args.null_repetitions,
        rng,
    )
    shared_reference = json.loads(args.shared_operator_results.read_text(encoding="utf-8"))
    complete_operator_reference = {
        kind: float(shared_reference["views"][kind]["primary"]["observed_full_operator_variance_recovered"])
        for kind in ("QK", "OV")
    }
    description_accounting = {}
    for rank in ranks:
        description = rank_description(int(common_metadata["d_model"]), rank)
        description_accounting[str(rank)] = {
            "unrestricted_rank_manifold_dimensions": description.unrestricted,
            "fixed_normalized_spectrum_dimensions": description.fixed_normalized_spectrum,
            "maximum_core_dimensions_saved": description.maximum_reusable_core_saving,
            "maximum_core_saving_fraction": description.saving_fraction,
        }

    report = {
        "status": "intrinsic-core MDL feasibility audit",
        "scientific_scope": "weight-only singular-profile reuse with head-specific frames supplied",
        "model": common_metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "initial_revision": args.initial_revision,
        "final_revision": args.final_revision,
        "normalization": "unit Frobenius norm per operator",
        "ranks": ranks,
        "profile_components": components,
        "views": views,
        "gaussian_factor_null": null,
        "synthetic_gate": synthetic_gate(rng),
        "description_accounting": description_accounting,
        "complete_operator_reference": complete_operator_reference,
        "interpretation_constraint": (
            "under independent input/output gauges, only singular values identify an intrinsic "
            "core; profile reconstruction supplies the true singular frames for free"
        ),
        "null_repetitions": args.null_repetitions,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report, args.figure)
    plot_profile_modes(report, args.profile_figure)
    print(f"saved audit to {args.output}")
    print(f"saved figure to {args.figure}")
    print(f"saved profile figure to {args.profile_figure}")


if __name__ == "__main__":
    main()
