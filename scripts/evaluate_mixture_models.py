"""Compare hard clusters with sparse mixtures and dense linear factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, DictionaryLearning, sparse_encode
from sklearn.model_selection import KFold

from head_atlas.distances import weighted_product_distances
from head_atlas.embedding import classical_mds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--active-atoms", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--dictionary-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_distances(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        distances = np.asarray(bundle["distances"], dtype=np.float64)
        layers = np.asarray(bundle["layers"], dtype=np.int64)
        heads = np.asarray(bundle["heads"], dtype=np.int64)
    return distances, layers, heads


def coordinates_from_distances(distances: np.ndarray) -> np.ndarray:
    result = classical_mds(distances)
    if result["negative_eigenvalue_mass_ratio"] > 1e-8:
        raise ValueError("mixture comparison requires Euclidean input distances")
    return np.asarray(result["coordinates"], dtype=np.float64)


def cross_validated_models(
    coordinates: np.ndarray,
    component_counts: list[int],
    active_atom_counts: list[int],
    folds: int,
    dictionary_alpha: float,
    seed: int,
) -> dict[str, object]:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    result = {}
    for component_count in component_counts:
        sums = {"baseline": 0.0, "kmeans": 0.0, "pca": 0.0}
        for active in active_atom_counts:
            if active <= component_count:
                sums[f"dictionary_{active}"] = 0.0

        for fold, (train_indices, test_indices) in enumerate(splitter.split(coordinates)):
            train = coordinates[train_indices]
            test = coordinates[test_indices]
            train_mean = np.mean(train, axis=0, keepdims=True)
            train_centered = train - train_mean
            test_centered = test - train_mean
            sums["baseline"] += float(np.sum(test_centered**2))

            kmeans = KMeans(
                n_clusters=component_count,
                n_init=20,
                random_state=seed + fold,
            ).fit(train_centered)
            kmeans_reconstruction = kmeans.cluster_centers_[kmeans.predict(test_centered)]
            sums["kmeans"] += float(np.sum((test_centered - kmeans_reconstruction) ** 2))

            pca = PCA(n_components=component_count, random_state=seed + fold).fit(
                train_centered
            )
            pca_reconstruction = pca.inverse_transform(pca.transform(test_centered))
            sums["pca"] += float(np.sum((test_centered - pca_reconstruction) ** 2))

            dictionary_model = DictionaryLearning(
                n_components=component_count,
                alpha=dictionary_alpha,
                max_iter=1000,
                fit_algorithm="cd",
                random_state=seed + fold,
            ).fit(train_centered)
            for active in active_atom_counts:
                if active > component_count:
                    continue
                codes = sparse_encode(
                    test_centered,
                    dictionary_model.components_,
                    algorithm="omp",
                    n_nonzero_coefs=active,
                )
                reconstruction = codes @ dictionary_model.components_
                sums[f"dictionary_{active}"] += float(
                    np.sum((test_centered - reconstruction) ** 2)
                )

        baseline = sums.pop("baseline")
        result[str(component_count)] = {
            "relative_test_squared_error": {
                model: error / baseline for model, error in sums.items()
            },
            "variance_recovered": {
                model: 1.0 - error / baseline for model, error in sums.items()
            },
        }
        print(
            f"components={component_count}: "
            + ", ".join(
                f"{model}={1.0 - error / baseline:.3f}"
                for model, error in sums.items()
            ),
            flush=True,
        )
    return result


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    if args.dictionary_alpha <= 0:
        raise ValueError("dictionary alpha must be positive")
    component_counts = sorted(set(args.components))
    active_atom_counts = sorted(set(args.active_atoms))
    if component_counts[0] < 2 or active_atom_counts[0] < 1:
        raise ValueError("component and active-atom counts must be positive")

    qk_distances, qk_layers, qk_heads = load_distances(args.qk_input)
    ov_distances, ov_layers, ov_heads = load_distances(args.ov_input)
    if not np.array_equal(qk_layers, ov_layers) or not np.array_equal(qk_heads, ov_heads):
        raise ValueError("QK and OV bundles must contain heads in the same order")
    distance_views = {
        "QK": qk_distances,
        "OV": ov_distances,
        "JOINT": weighted_product_distances([qk_distances, ov_distances]),
    }

    view_results = {}
    for view_name, distances in distance_views.items():
        print(f"evaluating {view_name}", flush=True)
        view_results[view_name] = cross_validated_models(
            coordinates_from_distances(distances),
            component_counts,
            active_atom_counts,
            args.folds,
            args.dictionary_alpha,
            args.seed,
        )

    result = {
        "analysis_status": "exploratory representation comparison",
        "feature_space": "exact PCoA coordinates of normalized-Frobenius geometry",
        "validation": f"{args.folds}-fold shuffled head-level cross-validation",
        "dictionary_fit_alpha": args.dictionary_alpha,
        "active_atom_counts": active_atom_counts,
        "component_counts": component_counts,
        "seed": args.seed,
        "views": view_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved mixture-model comparison to {args.output}")


if __name__ == "__main__":
    main()
