"""Audit gradients, nuisance associations, and global cluster cuts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from head_atlas.distances import weighted_product_distances
from head_atlas.structure import (
    categorical_permanova,
    design_permanova,
    pcoa_spectrum_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--qk-statistics", type=Path, required=True)
    parser.add_argument("--ov-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-clusters", type=int, default=20)
    return parser.parse_args()


def load_distance_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as bundle:
        distances = np.asarray(bundle["distances"], dtype=np.float64)
        layers = np.asarray(bundle["layers"], dtype=np.int64)
        heads = np.asarray(bundle["heads"], dtype=np.int64)
        kinds = np.asarray(bundle["kinds"])
    if np.unique(kinds).size != 1:
        raise ValueError(f"operator kinds in {path} must be uniform")
    return distances, layers, heads, str(kinds[0])


def load_diagnostics(
    path: Path,
    layers: np.ndarray,
    heads: np.ndarray,
    expected_kind: str,
) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    by_location = {(int(row["layer"]), int(row["head"])): row for row in rows}
    ordered = []
    for layer, head in zip(layers, heads, strict=True):
        location = (int(layer), int(head))
        if location not in by_location:
            raise ValueError(f"head L{layer}H{head} is missing from {path}")
        row = by_location[location]
        if row["kind"] != expected_kind:
            raise ValueError(f"unexpected operator kind in {path}: {row['kind']}")
        ordered.append(row)
    if len(by_location) != len(ordered):
        raise ValueError(f"{path} has extra or duplicate head records")
    return {
        "effective_rank": np.asarray(
            [float(row["effective_rank"]) for row in ordered], dtype=np.float64
        ),
        "top_1_energy": np.asarray(
            [float(row["top_1_energy"]) for row in ordered], dtype=np.float64
        ),
    }


def cluster_sweep(distances: np.ndarray, maximum_clusters: int) -> list[dict[str, object]]:
    results = []
    for cluster_count in range(2, maximum_clusters + 1):
        labels = AgglomerativeClustering(
            n_clusters=cluster_count,
            metric="precomputed",
            linkage="average",
        ).fit_predict(distances)
        sizes = np.bincount(labels)
        results.append(
            {
                "cluster_count": cluster_count,
                "silhouette": float(silhouette_score(distances, labels, metric="precomputed")),
                "cluster_sizes": sorted([int(size) for size in sizes]),
                "singleton_count": int(np.count_nonzero(sizes == 1)),
                "largest_cluster_fraction": float(np.max(sizes) / len(distances)),
            }
        )
    return results


def audit_view(
    distances: np.ndarray,
    layers: np.ndarray,
    predictors: dict[str, np.ndarray],
    permutations: int,
    seed: int,
    maximum_clusters: int,
) -> dict[str, object]:
    maximum_layer = max(int(np.max(layers)), 1)
    layer_one_hot = np.eye(maximum_layer + 1, dtype=np.float64)[layers]
    spectral_predictors = np.column_stack(list(predictors.values()))
    all_predictors = np.column_stack([layer_one_hot, spectral_predictors])

    associations = {
        "categorical_layer": categorical_permanova(
            distances, layers, permutations=permutations, seed=seed
        ),
        "normalized_depth": design_permanova(
            distances,
            layers / maximum_layer,
            permutations=permutations,
            seed=seed,
        ),
        "spectral_predictors_joint": design_permanova(
            distances,
            spectral_predictors,
            permutations=permutations,
            seed=seed,
        ),
        "layer_and_spectral_predictors_joint": design_permanova(
            distances,
            all_predictors,
            permutations=permutations,
            seed=seed,
        ),
    }
    for name, values in predictors.items():
        associations[name] = design_permanova(
            distances,
            values,
            permutations=permutations,
            seed=seed,
        )

    sweep = cluster_sweep(distances, maximum_clusters)
    return {
        "pcoa_spectrum": pcoa_spectrum_summary(distances),
        "associations": associations,
        "average_linkage_cluster_sweep": sweep,
        "best_silhouette_cut": max(sweep, key=lambda record: record["silhouette"]),
    }


def main() -> None:
    args = parse_args()
    if args.permutations < 1:
        raise ValueError("permutations must be positive")
    if args.maximum_clusters < 2:
        raise ValueError("maximum clusters must be at least two")

    qk_distances, qk_layers, qk_heads, qk_kind = load_distance_bundle(args.qk_input)
    ov_distances, ov_layers, ov_heads, ov_kind = load_distance_bundle(args.ov_input)
    if not np.array_equal(qk_layers, ov_layers) or not np.array_equal(qk_heads, ov_heads):
        raise ValueError("QK and OV bundles must contain heads in the same order")
    qk_diagnostics = load_diagnostics(
        args.qk_statistics, qk_layers, qk_heads, qk_kind
    )
    ov_diagnostics = load_diagnostics(
        args.ov_statistics, ov_layers, ov_heads, ov_kind
    )
    joint_distances = weighted_product_distances([qk_distances, ov_distances])
    joint_diagnostics = {
        "qk_effective_rank": qk_diagnostics["effective_rank"],
        "qk_top_1_energy": qk_diagnostics["top_1_energy"],
        "ov_effective_rank": ov_diagnostics["effective_rank"],
        "ov_top_1_energy": ov_diagnostics["top_1_energy"],
    }

    result = {
        "metric": "normalized_frobenius",
        "joint_metric": "equal-weight-euclidean-product-of-qk-and-ov",
        "permutations": args.permutations,
        "seed": args.seed,
        "views": {
            "QK": audit_view(
                qk_distances,
                qk_layers,
                qk_diagnostics,
                args.permutations,
                args.seed,
                args.maximum_clusters,
            ),
            "OV": audit_view(
                ov_distances,
                ov_layers,
                ov_diagnostics,
                args.permutations,
                args.seed,
                args.maximum_clusters,
            ),
            "JOINT": audit_view(
                joint_distances,
                qk_layers,
                joint_diagnostics,
                args.permutations,
                args.seed,
                args.maximum_clusters,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved population structure audit to {args.output}")
    for view_name, view_result in result["views"].items():
        layer_result = view_result["associations"]["categorical_layer"]
        best_cut = view_result["best_silhouette_cut"]
        print(
            f"{view_name}: layer R2={layer_result['explained_variance_fraction']:.4f}, "
            f"p={layer_result['p_value']:.4g}; best average-linkage silhouette="
            f"{best_cut['silhouette']:.4f} at k={best_cut['cluster_count']}"
        )


if __name__ == "__main__":
    main()

