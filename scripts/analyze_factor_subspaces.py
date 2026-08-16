"""Audit reusable query/key and read/write subspaces across Pythia training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from head_atlas.distance_audit import summarize_distance_matrix
from head_atlas.distances import chordal_subspace_distances
from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_singular_components
from head_atlas.nulls import haar_orthonormal_frame
from head_atlas.structure import categorical_permanova, pcoa_spectrum_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/factor_subspace_atlas.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/factor_subspace_atlas.png"),
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--null-repetitions", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def decompose_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    operators, _ = load_factor_bundle(path)
    left_bases = []
    spectra = []
    right_bases = []
    for operator in operators:
        left, values, right = factorized_singular_components(operator)
        left_bases.append(left.astype(np.float32))
        spectra.append(values)
        right_bases.append(right.astype(np.float32))
    layers = np.asarray([operator.layer for operator in operators], dtype=np.int64)
    return np.stack(left_bases), np.stack(spectra), np.stack(right_bases), layers


def cross_chordal_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Distances between aligned populations of equal-rank orthonormal bases."""

    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("cross populations must have matching basis arrays")
    rank = first.shape[2]
    inner = np.einsum("idr,jds->ijrs", first, second, optimize=True)
    overlaps = np.sum(inner**2, axis=(2, 3), dtype=np.float64)
    return np.sqrt(np.maximum(1.0 - np.clip(overlaps / rank, 0.0, 1.0), 0.0))


def trajectory_summary(cross_distances: np.ndarray) -> dict[str, float]:
    if cross_distances.ndim != 2 or cross_distances.shape[0] != cross_distances.shape[1]:
        raise ValueError("trajectory cross distances must be square")
    diagonal = np.diag(cross_distances)
    off_diagonal = cross_distances[~np.eye(len(cross_distances), dtype=bool)]
    return {
        "mean_same_head_distance": float(np.mean(diagonal)),
        "mean_different_head_distance": float(np.mean(off_diagonal)),
        "same_to_different_distance_ratio": float(
            np.mean(diagonal) / np.mean(off_diagonal)
        ),
        "nearest_neighbor_head_identity_accuracy": float(
            np.mean(np.argmin(cross_distances, axis=1) == np.arange(len(cross_distances)))
        ),
    }


def null_comparison(
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


def nearest_neighbors(distances: np.ndarray) -> np.ndarray:
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    return np.argmin(masked, axis=1)


def analyze_view(
    checkpoint_paths: list[Path],
    checkpoint_names: list[str],
    ranks: list[int],
    *,
    null_repetitions: int,
    permutations: int,
    rng: np.random.Generator,
    seed: int,
) -> dict[str, object]:
    checkpoints = []
    layers = None
    for path in checkpoint_paths:
        left, spectra, right, checkpoint_layers = decompose_checkpoint(path)
        if layers is None:
            layers = checkpoint_layers
        elif not np.array_equal(layers, checkpoint_layers):
            raise ValueError("checkpoint head metadata changed")
        checkpoints.append({"left": left, "spectra": spectra, "right": right})
    if layers is None:
        raise ValueError("no checkpoints supplied")

    final = checkpoints[-1]
    item_count, dimension, maximum_rank = final["left"].shape
    null_records = {
        rank: {side: {"distance": [], "structure": []} for side in ("left", "right")}
        for rank in ranks
    }
    for _ in range(null_repetitions):
        null_bases = {
            side: np.stack(
                [
                    haar_orthonormal_frame(dimension, maximum_rank, rng).astype(np.float32)
                    for _ in range(item_count)
                ]
            )
            for side in ("left", "right")
        }
        for rank in ranks:
            for side in ("left", "right"):
                distances = chordal_subspace_distances(null_bases[side][:, :, :rank])
                null_records[rank][side]["distance"].append(
                    summarize_distance_matrix(distances)
                )
                null_records[rank][side]["structure"].append(
                    pcoa_spectrum_summary(distances)
                )

    rank_results = {}
    trajectory_results = {str(rank): {"left": [], "right": []} for rank in ranks}
    for rank in ranks:
        side_distances = {
            side: chordal_subspace_distances(final[side][:, :, :rank])
            for side in ("left", "right")
        }
        upper = np.triu_indices(item_count, 1)
        correlation = float(
            spearmanr(side_distances["left"][upper], side_distances["right"][upper]).statistic
        )
        neighbor_agreement = float(
            np.mean(
                nearest_neighbors(side_distances["left"])
                == nearest_neighbors(side_distances["right"])
            )
        )
        sides = {}
        for side in ("left", "right"):
            distances = side_distances[side]
            spectrum_energy = final["spectra"] ** 2
            sides[side] = {
                "mean_operator_energy_in_rank": float(
                    np.mean(
                        np.sum(spectrum_energy[:, :rank], axis=1)
                        / np.sum(spectrum_energy, axis=1)
                    )
                ),
                "distance_null_comparison": null_comparison(
                    summarize_distance_matrix(distances),
                    null_records[rank][side]["distance"],
                ),
                "structure_null_comparison": null_comparison(
                    pcoa_spectrum_summary(distances),
                    null_records[rank][side]["structure"],
                ),
                "layer_association": categorical_permanova(
                    distances,
                    layers,
                    permutations=permutations,
                    seed=seed + rank + (0 if side == "left" else 1000),
                ),
            }
        rank_results[str(rank)] = {
            "sides": sides,
            "left_right_pairwise_spearman": correlation,
            "left_right_nearest_neighbor_agreement": neighbor_agreement,
        }

    for first_index in range(len(checkpoints) - 1):
        first = checkpoints[first_index]
        second = checkpoints[first_index + 1]
        for side in ("left", "right"):
            inner = np.einsum(
                "idr,jds->ijrs", first[side], second[side], optimize=True
            )
            for rank in ranks:
                overlaps = np.sum(
                    inner[:, :, :rank, :rank] ** 2, axis=(2, 3), dtype=np.float64
                )
                cross = np.sqrt(
                    np.maximum(1.0 - np.clip(overlaps / rank, 0.0, 1.0), 0.0)
                )
                trajectory_results[str(rank)][side].append(
                    {
                        "first_checkpoint": checkpoint_names[first_index],
                        "second_checkpoint": checkpoint_names[first_index + 1],
                        **trajectory_summary(cross),
                    }
                )

    return {
        "final_checkpoint": checkpoint_names[-1],
        "operator_count": item_count,
        "residual_width": dimension,
        "maximum_operator_rank": maximum_rank,
        "rank_results": rank_results,
        "adjacent_checkpoint_trajectories": trajectory_results,
    }


def plot_report(report: dict[str, object], ranks: list[int], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    side_labels = {
        "QK": {"left": "query", "right": "key"},
        "OV": {"left": "read", "right": "write"},
    }
    for row, view in enumerate(("QK", "OV")):
        view_result = report["views"][view]
        for side, style in (("left", "-o"), ("right", "-s")):
            dimensions = []
            null_dimensions = []
            trajectory_accuracy = []
            for rank in ranks:
                result = view_result["rank_results"][str(rank)]["sides"][side]
                comparison = result["structure_null_comparison"][
                    "participation_dimension"
                ]
                dimensions.append(comparison["real"])
                null_dimensions.append(comparison["null_mean"])
                trajectories = view_result["adjacent_checkpoint_trajectories"][
                    str(rank)
                ][side]
                trajectory_accuracy.append(
                    np.mean(
                        [
                            record["nearest_neighbor_head_identity_accuracy"]
                            for record in trajectories
                        ]
                    )
                )
            label = side_labels[view][side]
            axes[row, 0].plot(ranks, dimensions, style, label=label)
            axes[row, 0].plot(
                ranks,
                null_dimensions,
                linestyle=":",
                color=axes[row, 0].lines[-1].get_color(),
                alpha=0.7,
            )
            axes[row, 2].plot(ranks, trajectory_accuracy, style, label=label)

        correlations = [
            view_result["rank_results"][str(rank)]["left_right_pairwise_spearman"]
            for rank in ranks
        ]
        agreements = [
            view_result["rank_results"][str(rank)][
                "left_right_nearest_neighbor_agreement"
            ]
            for rank in ranks
        ]
        axes[row, 1].plot(ranks, correlations, "-o", label="pairwise Spearman")
        axes[row, 1].plot(ranks, agreements, "-s", label="nearest-neighbor agreement")
        axes[row, 0].set_ylabel(f"{view}\nparticipation dimension")
        axes[row, 1].set_ylabel("left/right dependence")
        axes[row, 2].set_ylabel("trajectory identity accuracy")
        axes[row, 1].set_ylim(0.0, 1.0)
        axes[row, 2].set_ylim(0.0, 1.02)
        for axis in axes[row]:
            axis.set_xlabel("Leading subspace rank")
            axis.set_xticks(ranks)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    axes[0, 0].set_title("Real (solid) versus random (dotted)")
    axes[0, 1].set_title("Do the two sides recombine independently?")
    axes[0, 2].set_title("Does a subspace preserve head identity?")
    figure.suptitle(
        "Pythia-70M attention-head factor subspaces across training",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ranks = sorted(set(args.ranks))
    if not ranks or ranks[0] < 1:
        raise ValueError("ranks must be positive")
    if args.null_repetitions < 1 or args.permutations < 1:
        raise ValueError("null repetitions and permutations must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint_names = [record["revision"] for record in manifest["records"]]
    rng = np.random.default_rng(args.seed)
    views = {}
    for view in ("QK", "OV"):
        paths = [Path(record["factors"][view]["path"]) for record in manifest["records"]]
        views[view] = analyze_view(
            paths,
            checkpoint_names,
            ranks,
            null_repetitions=args.null_repetitions,
            permutations=args.permutations,
            rng=rng,
            seed=args.seed + (0 if view == "QK" else 10000),
        )
        print(f"completed {view} subspace atlas", flush=True)
    report = {
        "analysis_status": "descriptive factor-subspace and trajectory audit",
        "metric": "normalized chordal projector distance",
        "null_model": "independent Haar-random subspaces at matched width and rank",
        "factorization": "exact skinny-factor SVD",
        "side_semantics": {
            "QK": {"left": "query-side", "right": "key-side"},
            "OV": {"left": "value/read-side", "right": "output/write-side"},
        },
        "ranks": ranks,
        "null_repetitions": args.null_repetitions,
        "permutations": args.permutations,
        "seed": args.seed,
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    plot_report(report, ranks, args.figure)
    print(f"saved subspace report to {args.output} and figure to {args.figure}")


if __name__ == "__main__":
    main()
