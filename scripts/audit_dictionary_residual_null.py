"""Test whether dictionary reconstruction residuals retain non-random geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.distance_audit import summarize_distance_matrix
from head_atlas.distances import normalized_vector_distances
from head_atlas.nulls import sample_norm_matched_isotropic
from head_atlas.residuals import dictionary_residuals, reconstruction_energy_summary
from head_atlas.structure import pcoa_spectrum_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/dictionary_residual_null.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/dictionary_residual_null.png"),
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def numeric_null_summary(
    real: dict[str, int | float], nulls: list[dict[str, int | float]]
) -> dict[str, dict[str, float]]:
    result = {}
    denominator = len(nulls) + 1
    for key, value in real.items():
        if not isinstance(value, float):
            continue
        null_values = np.asarray([record[key] for record in nulls], dtype=np.float64)
        result[key] = {
            "real": value,
            "null_mean": float(np.mean(null_values)),
            "null_standard_deviation": float(np.std(null_values)),
            "lower_tail_p_value": float(
                (1 + np.count_nonzero(null_values <= value)) / denominator
            ),
            "upper_tail_p_value": float(
                (1 + np.count_nonzero(null_values >= value)) / denominator
            ),
        }
    return result


def audit_population(
    centered: np.ndarray,
    residuals: np.ndarray,
    mask: np.ndarray,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    selected_centered = centered[mask]
    selected_residuals = residuals[mask]
    real_distances = normalized_vector_distances(selected_residuals)
    real_distance_summary = summarize_distance_matrix(real_distances)
    real_structure_summary = pcoa_spectrum_summary(real_distances)
    null_distance_summaries = []
    null_structure_summaries = []
    for _ in range(repetitions):
        null_residuals = sample_norm_matched_isotropic(selected_residuals, rng)
        null_distances = normalized_vector_distances(null_residuals)
        null_distance_summaries.append(summarize_distance_matrix(null_distances))
        null_structure_summaries.append(pcoa_spectrum_summary(null_distances))
    return {
        "observations": int(np.count_nonzero(mask)),
        "reconstruction_energy": reconstruction_energy_summary(
            selected_centered, selected_residuals
        ),
        "distance_null_comparison": numeric_null_summary(
            real_distance_summary, null_distance_summaries
        ),
        "structure_null_comparison": numeric_null_summary(
            real_structure_summary, null_structure_summaries
        ),
    }


def plot_report(report: dict[str, object], output: Path) -> None:
    labels = []
    discovery_energy = []
    final_energy = []
    discovery_dimension_ratio = []
    final_dimension_ratio = []
    discovery_neighbor_ratio = []
    final_neighbor_ratio = []
    for view in ("QK", "OV", "JOINT"):
        for profile in ("compact", "optimal"):
            labels.append(f"{view}\n{profile}")
            populations = report["views"][view][profile]["populations"]
            discovery = populations["discovery"]
            final = populations["final_checkpoint"]
            discovery_energy.append(
                discovery["reconstruction_energy"]["global_energy_captured"]
            )
            final_energy.append(final["reconstruction_energy"]["global_energy_captured"])
            for record, target in (
                (discovery, discovery_dimension_ratio),
                (final, final_dimension_ratio),
            ):
                comparison = record["structure_null_comparison"][
                    "participation_dimension"
                ]
                target.append(comparison["real"] / comparison["null_mean"])
            for record, target in (
                (discovery, discovery_neighbor_ratio),
                (final, final_neighbor_ratio),
            ):
                comparison = record["distance_null_comparison"]["nearest_mean"]
                target.append(comparison["real"] / comparison["null_mean"])

    x = np.arange(len(labels))
    width = 0.37
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    axes[0].bar(x - width / 2, discovery_energy, width, label="all discovery checkpoints")
    axes[0].bar(x + width / 2, final_energy, width, label="final checkpoint only")
    axes[0].set_ylabel("Centered energy captured")
    axes[0].set_title("How much the dictionary removes")
    for axis, discovery_values, final_values, title, ylabel in (
        (
            axes[1],
            discovery_dimension_ratio,
            final_dimension_ratio,
            "Residual effective dimension",
            "Real / isotropic-null dimension",
        ),
        (
            axes[2],
            discovery_neighbor_ratio,
            final_neighbor_ratio,
            "Residual nearest neighbors",
            "Real / isotropic-null distance",
        ),
    ):
        axis.bar(x - width / 2, discovery_values, width)
        axis.bar(x + width / 2, final_values, width)
        axis.axhline(1.0, color="0.35", linestyle=":", linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Structure remaining after Pythia dictionary reconstruction",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = np.random.default_rng(args.seed)
    views = {}
    for view in ("QK", "OV", "JOINT"):
        views[view] = {}
        for profile, suffix in (("compact", "_compact"), ("optimal", "")):
            artifact = load_artifact(
                args.artifact_root / f"{view.lower()}{suffix}_dictionary.npz"
            )
            centered, residuals = dictionary_residuals(
                artifact["coordinates"],
                artifact["coordinate_mean"],
                artifact["atoms"],
                artifact["codes"],
            )
            discovery_mask = np.asarray(artifact["discovery_mask"], dtype=bool)
            final_mask = artifact["checkpoint_values"] == np.max(
                artifact["checkpoint_values"]
            )
            views[view][profile] = {
                "components": len(artifact["atoms"]),
                "selected_active_atoms": int(artifact["selected_active_atoms"]),
                "populations": {
                    "discovery": audit_population(
                        centered,
                        residuals,
                        discovery_mask,
                        repetitions=args.repetitions,
                        rng=rng,
                    ),
                    "final_checkpoint": audit_population(
                        centered,
                        residuals,
                        final_mask,
                        repetitions=args.repetitions,
                        rng=rng,
                    ),
                },
            }
            final_result = views[view][profile]["populations"]["final_checkpoint"]
            captured = final_result["reconstruction_energy"]["global_energy_captured"]
            participation = final_result["structure_null_comparison"][
                "participation_dimension"
            ]
            print(
                f"{view} {profile}: final captured={captured:.3f}, "
                f"residual dimension={participation['real']:.1f} vs "
                f"null={participation['null_mean']:.1f}",
                flush=True,
            )

    report = {
        "analysis_status": "descriptive post-dictionary residual null audit",
        "geometry": "exact PCoA coordinate residuals of normalized operators",
        "null_model": "per-observation-norm-matched independent isotropic directions",
        "interpretation": (
            "tests whether unexplained dictionary residual directions retain population "
            "structure; it is not an operator singular-spectrum null"
        ),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved residual null audit to {args.output} and figure to {args.figure}")


if __name__ == "__main__":
    main()
