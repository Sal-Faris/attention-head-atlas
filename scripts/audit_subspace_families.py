"""Test whether factor subspaces form stable, reusable attention-head families."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from head_atlas.distances import chordal_subspace_distances
from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_singular_components
from head_atlas.families import (
    average_linkage_labels,
    best_silhouette_cut,
    mean_neighbor_overlap,
    nearest_neighbor_indices,
    stratified_permutation,
    subsampled_cluster_stability,
)
from head_atlas.nulls import haar_orthonormal_frame


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
        default=Path("results/pythia-70m-deduped/subspace_family_audit.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/subspace_family_audit.png"),
    )
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--late-checkpoints", type=int, default=3)
    parser.add_argument("--maximum-clusters", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--null-repetitions", type=int, default=499)
    parser.add_argument("--haar-repetitions", type=int, default=49)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def decompose_checkpoint(path: Path) -> dict[str, np.ndarray]:
    operators, _ = load_factor_bundle(path)
    left_bases = []
    spectra = []
    right_bases = []
    for operator in operators:
        left, values, right = factorized_singular_components(operator)
        left_bases.append(left.astype(np.float32))
        spectra.append(values.astype(np.float32))
        right_bases.append(right.astype(np.float32))
    return {
        "left": np.stack(left_bases),
        "spectra": np.stack(spectra),
        "right": np.stack(right_bases),
        "layers": np.asarray([operator.layer for operator in operators], dtype=np.int64),
        "heads": np.asarray([operator.head for operator in operators], dtype=np.int64),
    }


def upper_tail_comparison(observed: float, null_values: list[float]) -> dict[str, float]:
    values = np.asarray(null_values, dtype=np.float64)
    standard_deviation = float(np.std(values))
    return {
        "observed": observed,
        "null_mean": float(np.mean(values)),
        "null_standard_deviation": standard_deviation,
        "z_score": (
            float((observed - np.mean(values)) / standard_deviation)
            if standard_deviation > 0
            else float("inf")
        ),
        "upper_tail_p_value": float(
            (1 + np.count_nonzero(values >= observed)) / (len(values) + 1)
        ),
    }


def cluster_layer_summary(labels: np.ndarray, layers: np.ndarray) -> dict[str, float]:
    unique_layers = np.unique(layers)
    weighted_entropy = 0.0
    cross_layer_pairs = 0
    all_within_cluster_pairs = 0
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        counts = np.asarray(
            [np.count_nonzero(layers[indices] == layer) for layer in unique_layers],
            dtype=np.float64,
        )
        probabilities = counts[counts > 0] / len(indices)
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        weighted_entropy += len(indices) * entropy / np.log(len(unique_layers))
        for first, second in combinations(indices, 2):
            all_within_cluster_pairs += 1
            cross_layer_pairs += layers[first] != layers[second]
    return {
        "size_weighted_normalized_layer_entropy": weighted_entropy / len(labels),
        "within_cluster_cross_layer_pair_fraction": (
            cross_layer_pairs / all_within_cluster_pairs
            if all_within_cluster_pairs
            else 0.0
        ),
    }


def recurrent_cross_layer_edges(
    distances: list[np.ndarray],
    layers: np.ndarray,
    heads: np.ndarray,
    neighbor_count: int,
    maximum_edges: int = 12,
) -> list[dict[str, int | float]]:
    """Rank final cross-layer neighbor edges by recurrence through checkpoints."""

    neighbor_sets = []
    for matrix in distances:
        neighbors = nearest_neighbor_indices(
            matrix,
            neighbor_count,
            groups=layers,
            different_group_only=True,
        )
        neighbor_sets.append([set(row) for row in neighbors])

    final_sets = neighbor_sets[-1]
    candidates = set()
    for first, neighbors in enumerate(final_sets):
        for second in neighbors:
            candidates.add(tuple(sorted((first, int(second)))))

    records = []
    for first, second in candidates:
        recurrence = np.mean(
            [
                second in checkpoint[first] or first in checkpoint[second]
                for checkpoint in neighbor_sets
            ]
        )
        records.append(
            {
                "first_layer": int(layers[first]),
                "first_head": int(heads[first]),
                "second_layer": int(layers[second]),
                "second_head": int(heads[second]),
                "checkpoint_recurrence_fraction": float(recurrence),
                "mutual_at_final_checkpoint": int(
                    second in final_sets[first] and first in final_sets[second]
                ),
                "final_distance": float(distances[-1][first, second]),
            }
        )
    records.sort(
        key=lambda record: (
            -float(record["checkpoint_recurrence_fraction"]),
            -int(record["mutual_at_final_checkpoint"]),
            float(record["final_distance"]),
        )
    )
    return records[:maximum_edges]


def temporal_metrics(
    distances: list[np.ndarray],
    layers: np.ndarray,
    *,
    cluster_count: int,
    neighbor_count: int,
    null_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    final_labels = average_linkage_labels(distances[-1], cluster_count)
    final_neighbors = nearest_neighbor_indices(distances[-1], neighbor_count)
    final_cross_layer = nearest_neighbor_indices(
        distances[-1],
        neighbor_count,
        groups=layers,
        different_group_only=True,
    )

    observed_ari = []
    observed_neighbors = []
    observed_cross_layer = []
    earlier_labels = []
    for matrix in distances[:-1]:
        labels = average_linkage_labels(matrix, cluster_count)
        earlier_labels.append(labels)
        observed_ari.append(adjusted_rand_score(final_labels, labels))
        observed_neighbors.append(
            mean_neighbor_overlap(
                final_neighbors,
                nearest_neighbor_indices(matrix, neighbor_count),
            )
        )
        observed_cross_layer.append(
            mean_neighbor_overlap(
                final_cross_layer,
                nearest_neighbor_indices(
                    matrix,
                    neighbor_count,
                    groups=layers,
                    different_group_only=True,
                ),
            )
        )

    ari_null = []
    neighbor_null = []
    cross_layer_null = []
    for _ in range(null_repetitions):
        repetition_ari = []
        repetition_neighbors = []
        repetition_cross_layer = []
        for matrix, labels in zip(distances[:-1], earlier_labels, strict=True):
            permutation = stratified_permutation(layers, rng)
            permuted_matrix = matrix[np.ix_(permutation, permutation)]
            repetition_ari.append(adjusted_rand_score(final_labels, labels[permutation]))
            repetition_neighbors.append(
                mean_neighbor_overlap(
                    final_neighbors,
                    nearest_neighbor_indices(permuted_matrix, neighbor_count),
                )
            )
            repetition_cross_layer.append(
                mean_neighbor_overlap(
                    final_cross_layer,
                    nearest_neighbor_indices(
                        permuted_matrix,
                        neighbor_count,
                        groups=layers,
                        different_group_only=True,
                    ),
                )
            )
        ari_null.append(float(np.mean(repetition_ari)))
        neighbor_null.append(float(np.mean(repetition_neighbors)))
        cross_layer_null.append(float(np.mean(repetition_cross_layer)))

    return {
        "cluster_adjusted_rand": upper_tail_comparison(float(np.mean(observed_ari)), ari_null),
        "neighbor_overlap": upper_tail_comparison(
            float(np.mean(observed_neighbors)), neighbor_null
        ),
        "cross_layer_neighbor_overlap": upper_tail_comparison(
            float(np.mean(observed_cross_layer)), cross_layer_null
        ),
    }


def pairing_metrics(
    left_distances: np.ndarray,
    right_distances: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    layers: np.ndarray,
    *,
    neighbor_count: int,
    null_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    upper = np.triu_indices(len(left_distances), 1)
    left_neighbors = nearest_neighbor_indices(left_distances, neighbor_count)

    def statistics(matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
        correlation = float(spearmanr(left_distances[upper], matrix[upper]).statistic)
        overlap = mean_neighbor_overlap(
            left_neighbors,
            nearest_neighbor_indices(matrix, neighbor_count),
        )
        mutual_information = float(adjusted_mutual_info_score(left_labels, labels))
        return correlation, overlap, mutual_information

    observed = statistics(right_distances, right_labels)
    nulls = [[], [], []]
    for _ in range(null_repetitions):
        permutation = stratified_permutation(layers, rng)
        permuted_matrix = right_distances[np.ix_(permutation, permutation)]
        for values, value in zip(
            nulls,
            statistics(permuted_matrix, right_labels[permutation]),
            strict=True,
        ):
            values.append(value)
    return {
        "pairwise_distance_spearman": upper_tail_comparison(observed[0], nulls[0]),
        "neighbor_overlap": upper_tail_comparison(observed[1], nulls[1]),
        "cluster_adjusted_mutual_information": upper_tail_comparison(
            observed[2], nulls[2]
        ),
    }


def haar_clusterability_null(
    *,
    item_count: int,
    dimension: int,
    ranks: list[int],
    maximum_clusters: int,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[int, list[float]]:
    maximum_rank = max(ranks)
    results = {rank: [] for rank in ranks}
    for _ in range(repetitions):
        bases = np.stack(
            [
                haar_orthonormal_frame(dimension, maximum_rank, rng).astype(np.float32)
                for _ in range(item_count)
            ]
        )
        for rank in ranks:
            distances = chordal_subspace_distances(bases[:, :, :rank])
            cut = best_silhouette_cut(distances, maximum_clusters)
            results[rank].append(float(cut["silhouette"]))
    return results


def analyze_view(
    checkpoints: list[dict[str, np.ndarray]],
    checkpoint_names: list[str],
    ranks: list[int],
    haar_null: dict[int, list[float]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, object]:
    late = checkpoints[-args.late_checkpoints :]
    layers = checkpoints[-1]["layers"]
    heads = checkpoints[-1]["heads"]
    results = {}
    for rank in ranks:
        side_results = {}
        final_labels = {}
        final_distances = {}
        for side in ("left", "right"):
            distances = [
                chordal_subspace_distances(checkpoint[side][:, :, :rank])
                for checkpoint in late
            ]
            final_distances[side] = distances[-1]
            cut = best_silhouette_cut(distances[-1], args.maximum_clusters)
            cluster_count = int(cut["cluster_count"])
            labels = np.asarray(cut["labels"], dtype=np.int64)
            final_labels[side] = labels
            cluster_sizes = np.bincount(labels)
            assignments = [
                {"layer": int(layer), "head": int(head), "cluster": int(label)}
                for layer, head, label in zip(layers, heads, labels, strict=True)
            ]
            side_results[side] = {
                "selected_cluster_count": cluster_count,
                "cluster_sizes": sorted(int(size) for size in cluster_sizes),
                "singleton_count": int(np.count_nonzero(cluster_sizes == 1)),
                "best_silhouette": upper_tail_comparison(
                    float(cut["silhouette"]), haar_null[rank]
                ),
                "subsampled_cluster_stability": subsampled_cluster_stability(
                    distances[-1],
                    cluster_count,
                    repetitions=args.bootstrap_repetitions,
                    sample_fraction=0.8,
                    rng=rng,
                ),
                "late_checkpoint_stability": temporal_metrics(
                    distances,
                    layers,
                    cluster_count=cluster_count,
                    neighbor_count=args.neighbors,
                    null_repetitions=args.null_repetitions,
                    rng=rng,
                ),
                "layer_mixing": cluster_layer_summary(labels, layers),
                "recurrent_cross_layer_edges": recurrent_cross_layer_edges(
                    distances,
                    layers,
                    heads,
                    args.neighbors,
                ),
                "assignments": assignments,
            }
        results[str(rank)] = {
            "sides": side_results,
            "left_right_pairing": pairing_metrics(
                final_distances["left"],
                final_distances["right"],
                final_labels["left"],
                final_labels["right"],
                layers,
                neighbor_count=args.neighbors,
                null_repetitions=args.null_repetitions,
                rng=rng,
            ),
        }
    return {
        "late_checkpoints": checkpoint_names[-args.late_checkpoints :],
        "operator_count": len(layers),
        "rank_results": results,
    }


def plot_report(report: dict[str, object], output: Path) -> None:
    ranks = report["ranks"]
    figure, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
    labels = {
        "QK": {"left": "query", "right": "key"},
        "OV": {"left": "read", "right": "write"},
    }
    for row, view in enumerate(("QK", "OV")):
        view_result = report["views"][view]
        for side, marker in (("left", "o"), ("right", "s")):
            side_name = labels[view][side]
            color = None
            for column, metric_path in enumerate(
                (
                    ("best_silhouette",),
                    ("late_checkpoint_stability", "cluster_adjusted_rand"),
                    ("late_checkpoint_stability", "cross_layer_neighbor_overlap"),
                )
            ):
                records = []
                for rank in ranks:
                    record = view_result["rank_results"][str(rank)]["sides"][side]
                    for key in metric_path:
                        record = record[key]
                    records.append(record)
                observed = [record["observed"] for record in records]
                null_mean = [record["null_mean"] for record in records]
                line = axes[row, column].plot(
                    ranks,
                    observed,
                    marker=marker,
                    label=side_name,
                    color=color,
                )[0]
                color = line.get_color()
                axes[row, column].plot(
                    ranks,
                    null_mean,
                    linestyle=":",
                    marker=marker,
                    color=color,
                    alpha=0.65,
                )

        pairing = view_result["rank_results"]
        for metric, marker, label in (
            ("pairwise_distance_spearman", "o", "distance Spearman"),
            ("neighbor_overlap", "s", "neighbor overlap"),
            ("cluster_adjusted_mutual_information", "^", "cluster AMI"),
        ):
            observed = [pairing[str(rank)]["left_right_pairing"][metric]["observed"] for rank in ranks]
            null_mean = [pairing[str(rank)]["left_right_pairing"][metric]["null_mean"] for rank in ranks]
            line = axes[row, 3].plot(ranks, observed, marker=marker, label=label)[0]
            axes[row, 3].plot(
                ranks,
                null_mean,
                linestyle=":",
                marker=marker,
                color=line.get_color(),
                alpha=0.65,
            )

        axes[row, 0].set_ylabel(view)
        for axis in axes[row]:
            axis.set_xticks(ranks)
            axis.set_xlabel("subspace rank")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
    axes[0, 0].set_title("Best cluster silhouette")
    axes[0, 1].set_title("Late-checkpoint cluster recurrence")
    axes[0, 2].set_title("Cross-layer neighbor recurrence")
    axes[0, 3].set_title("Left/right coupling")
    figure.suptitle(
        "Pythia-70M factor-subspace families: observed (solid) vs matched null (dotted)",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ranks = sorted(set(args.ranks))
    if not ranks or ranks[0] < 1:
        raise ValueError("ranks must be positive")
    if args.late_checkpoints < 2:
        raise ValueError("at least two late checkpoints are required")
    if min(
        args.bootstrap_repetitions,
        args.null_repetitions,
        args.haar_repetitions,
    ) < 1:
        raise ValueError("all repetition counts must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if args.late_checkpoints > len(records):
        raise ValueError("late-checkpoint count exceeds the manifest")
    checkpoint_names = [record["revision"] for record in records]
    rng = np.random.default_rng(args.seed)

    decompositions = {}
    for view in ("QK", "OV"):
        decompositions[view] = [
            decompose_checkpoint(Path(record["factors"][view]["path"]))
            for record in records
        ]
        reference = decompositions[view][-1]
        for checkpoint in decompositions[view][:-1]:
            if not np.array_equal(checkpoint["layers"], reference["layers"]) or not np.array_equal(
                checkpoint["heads"], reference["heads"]
            ):
                raise ValueError("checkpoint head metadata changed")
        print(f"decomposed {view} factors", flush=True)

    reference = decompositions["QK"][-1]
    item_count, dimension, maximum_rank = reference["left"].shape
    if max(ranks) > maximum_rank:
        raise ValueError("requested rank exceeds the factor width")
    haar_null = haar_clusterability_null(
        item_count=item_count,
        dimension=dimension,
        ranks=ranks,
        maximum_clusters=args.maximum_clusters,
        repetitions=args.haar_repetitions,
        rng=rng,
    )
    print("completed Haar clusterability null", flush=True)

    views = {}
    for view in ("QK", "OV"):
        views[view] = analyze_view(
            decompositions[view],
            checkpoint_names,
            ranks,
            haar_null,
            args,
            rng,
        )
        print(f"completed {view} family audit", flush=True)

    report = {
        "analysis_status": "descriptive subspace-family stability audit",
        "model": manifest["model"],
        "metric": "normalized chordal projector distance",
        "cluster_method": "average linkage; final cut selected by maximum silhouette over fixed range",
        "temporal_null": "independent head-identity shuffles within layers",
        "clusterability_null": "independent Haar-random subspaces with matched width and rank",
        "bootstrap": "80-percent item subsampling compared with the full final-checkpoint cut",
        "ranks": ranks,
        "late_checkpoint_count": args.late_checkpoints,
        "maximum_clusters": args.maximum_clusters,
        "neighbors": args.neighbors,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "null_repetitions": args.null_repetitions,
        "haar_repetitions": args.haar_repetitions,
        "seed": args.seed,
        "side_semantics": {
            "QK": {"left": "query", "right": "key"},
            "OV": {"left": "read/value", "right": "write/output"},
        },
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved family audit to {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
