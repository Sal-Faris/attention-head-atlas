"""Unsupervised sparse-atom discovery with leakage-resistant validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, DictionaryLearning, sparse_encode
from sklearn.model_selection import GroupKFold

Array = np.ndarray


def joint_view_coordinates(qk: Array, ov: Array) -> Array:
    """Return equal-weight product-space coordinates with shared sample rows."""

    qk_array = np.asarray(qk, dtype=np.float64)
    ov_array = np.asarray(ov, dtype=np.float64)
    if qk_array.ndim != 2 or ov_array.ndim != 2:
        raise ValueError("coordinate views must be two-dimensional")
    if qk_array.shape[0] != ov_array.shape[0]:
        raise ValueError("coordinate views must contain the same samples")
    if not np.isfinite(qk_array).all() or not np.isfinite(ov_array).all():
        raise ValueError("coordinate views contain non-finite values")
    return np.concatenate([qk_array, ov_array], axis=1) / np.sqrt(2.0)


def head_trajectory_groups(layers: Array, heads: Array) -> Array:
    """Assign one group to every repeated layer/head identity."""

    layer_array = np.asarray(layers, dtype=np.int64)
    head_array = np.asarray(heads, dtype=np.int64)
    if layer_array.ndim != 1 or head_array.shape != layer_array.shape:
        raise ValueError("layers and heads must be matching one-dimensional arrays")
    if np.any(layer_array < 0) or np.any(head_array < 0):
        raise ValueError("layer and head indices must be nonnegative")
    head_count = int(np.max(head_array)) + 1
    return layer_array * head_count + head_array


def grouped_splits(groups: Array, folds: int) -> list[tuple[Array, Array]]:
    """Return folds that keep all samples from a trajectory together."""

    group_array = np.asarray(groups)
    if group_array.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    unique_count = len(np.unique(group_array))
    if folds < 2 or folds > unique_count:
        raise ValueError("fold count must lie between two and the group count")
    splitter = GroupKFold(n_splits=folds)
    placeholder = np.zeros((len(group_array), 1))
    return list(splitter.split(placeholder, groups=group_array))


def blocked_checkpoint_splits(
    checkpoint_values: Array,
    blocks: int,
) -> list[tuple[Array, Array]]:
    """Hold out contiguous regions of ordered checkpoints."""

    values = np.asarray(checkpoint_values, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("checkpoint values must be one-dimensional")
    unique_values = np.unique(values)
    if blocks < 2 or blocks > len(unique_values):
        raise ValueError("block count must lie between two and the checkpoint count")
    splits = []
    for held_out_values in np.array_split(unique_values, blocks):
        test_mask = np.isin(values, held_out_values)
        splits.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    return splits


def _squared_error(actual: Array, reconstructed: Array) -> float:
    return float(np.sum((actual - reconstructed) ** 2))


def cross_validated_reconstruction(
    coordinates: Array,
    splits: Sequence[tuple[Array, Array]],
    component_counts: Sequence[int],
    active_atom_counts: Sequence[int],
    *,
    dictionary_alpha: float,
    seed: int,
    max_iter: int = 1000,
) -> dict[str, object]:
    """Compare hard clusters, PCA, and sparse dictionaries on fixed folds."""

    data = np.asarray(coordinates, dtype=np.float64)
    if data.ndim != 2 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("coordinates must be a finite nontrivial matrix")
    if dictionary_alpha <= 0 or max_iter < 1:
        raise ValueError("dictionary alpha and max_iter must be positive")
    components = sorted({int(value) for value in component_counts})
    active_counts = sorted({int(value) for value in active_atom_counts})
    if not components or components[0] < 2 or not active_counts or active_counts[0] < 1:
        raise ValueError("component and active-atom counts must be positive")

    results: dict[str, object] = {}
    for component_count in components:
        fold_records = []
        for fold, (train_indices, test_indices) in enumerate(splits):
            train = data[np.asarray(train_indices)]
            test = data[np.asarray(test_indices)]
            if component_count >= min(len(train), data.shape[1] + 1):
                raise ValueError("component count is too large for a training fold")
            train_mean = np.mean(train, axis=0, keepdims=True)
            train_centered = train - train_mean
            test_centered = test - train_mean
            baseline_error = float(np.sum(test_centered**2))

            kmeans = KMeans(
                n_clusters=component_count,
                n_init=20,
                random_state=seed + fold,
            ).fit(train_centered)
            kmeans_error = _squared_error(
                test_centered,
                kmeans.cluster_centers_[kmeans.predict(test_centered)],
            )
            pca = PCA(n_components=component_count, random_state=seed + fold).fit(
                train_centered
            )
            pca_error = _squared_error(
                test_centered,
                pca.inverse_transform(pca.transform(test_centered)),
            )
            dictionary = DictionaryLearning(
                n_components=component_count,
                alpha=dictionary_alpha,
                max_iter=max_iter,
                fit_algorithm="cd",
                random_state=seed + fold,
            ).fit(train_centered)
            model_errors = {"kmeans": kmeans_error, "pca": pca_error}
            for active_count in active_counts:
                if active_count > component_count:
                    continue
                codes = sparse_encode(
                    test_centered,
                    dictionary.components_,
                    algorithm="omp",
                    n_nonzero_coefs=active_count,
                )
                model_errors[f"dictionary_{active_count}"] = _squared_error(
                    test_centered, codes @ dictionary.components_
                )
            fold_records.append(
                {
                    "fold": fold,
                    "train_samples": len(train),
                    "test_samples": len(test),
                    "baseline_squared_error": baseline_error,
                    "relative_squared_error": {
                        model: error / baseline_error
                        for model, error in model_errors.items()
                    },
                }
            )

        model_names = list(fold_records[0]["relative_squared_error"])
        results[str(component_count)] = {
            "folds": fold_records,
            "mean_relative_squared_error": {
                model: float(
                    np.mean(
                        [record["relative_squared_error"][model] for record in fold_records]
                    )
                )
                for model in model_names
            },
            "standard_error_relative_squared_error": {
                model: float(
                    np.std(
                        [record["relative_squared_error"][model] for record in fold_records],
                        ddof=1,
                    )
                    / np.sqrt(len(fold_records))
                )
                for model in model_names
            },
        }
    return results
