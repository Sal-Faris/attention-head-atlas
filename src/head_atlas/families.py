"""Stability diagnostics for candidate families in a distance geometry."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

Array = np.ndarray


def _validate_distances(distances: Array) -> Array:
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("distances must be a square matrix with at least two items")
    if not np.isfinite(matrix).all():
        raise ValueError("distances contain non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-8):
        raise ValueError("distances must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-8):
        raise ValueError("distances must have a zero diagonal")
    if np.min(matrix) < -1e-8:
        raise ValueError("distances cannot be negative")
    return matrix


def average_linkage_labels(distances: Array, cluster_count: int) -> Array:
    """Cut an average-linkage hierarchy at a requested cluster count."""

    matrix = _validate_distances(distances)
    if cluster_count < 2 or cluster_count >= len(matrix):
        raise ValueError("cluster_count must be between two and item_count - 1")
    return AgglomerativeClustering(
        n_clusters=cluster_count,
        metric="precomputed",
        linkage="average",
    ).fit_predict(matrix)


def best_silhouette_cut(distances: Array, maximum_clusters: int) -> dict[str, object]:
    """Return the strongest average-linkage cut in a fixed candidate range."""

    matrix = _validate_distances(distances)
    upper = min(maximum_clusters, len(matrix) - 1)
    if upper < 2:
        raise ValueError("maximum_clusters must be at least two")
    records = []
    for cluster_count in range(2, upper + 1):
        labels = average_linkage_labels(matrix, cluster_count)
        records.append(
            {
                "cluster_count": cluster_count,
                "silhouette": float(silhouette_score(matrix, labels, metric="precomputed")),
                "labels": labels,
            }
        )
    return max(records, key=lambda record: float(record["silhouette"]))


def nearest_neighbor_indices(
    distances: Array,
    neighbor_count: int,
    *,
    groups: Array | None = None,
    different_group_only: bool = False,
) -> Array:
    """Return deterministic nearest-neighbor indices, optionally across groups."""

    matrix = _validate_distances(distances).copy()
    item_count = len(matrix)
    if neighbor_count < 1:
        raise ValueError("neighbor_count must be positive")
    np.fill_diagonal(matrix, np.inf)
    if different_group_only:
        if groups is None:
            raise ValueError("groups are required for cross-group neighbors")
        group_array = np.asarray(groups)
        if group_array.shape != (item_count,):
            raise ValueError("groups must contain one value per item")
        matrix[group_array[:, None] == group_array[None, :]] = np.inf
    available = np.sum(np.isfinite(matrix), axis=1)
    if np.any(available < neighbor_count):
        raise ValueError("not enough eligible neighbors")
    return np.argsort(matrix, axis=1, kind="stable")[:, :neighbor_count]


def mean_neighbor_overlap(first: Array, second: Array) -> float:
    """Mean fraction of neighbors retained between two aligned populations."""

    first_array = np.asarray(first)
    second_array = np.asarray(second)
    if first_array.shape != second_array.shape or first_array.ndim != 2:
        raise ValueError("neighbor arrays must have matching two-dimensional shapes")
    if first_array.shape[1] < 1:
        raise ValueError("neighbor arrays cannot be empty")
    overlaps = [
        len(set(row_first).intersection(row_second)) / first_array.shape[1]
        for row_first, row_second in zip(first_array, second_array, strict=True)
    ]
    return float(np.mean(overlaps))


def stratified_permutation(groups: Array, rng: np.random.Generator) -> Array:
    """Permute item identities independently inside each stratum."""

    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.size < 2:
        raise ValueError("groups must be a one-dimensional population")
    permutation = np.arange(group_array.size)
    for group in np.unique(group_array):
        positions = np.flatnonzero(group_array == group)
        permutation[positions] = rng.permutation(positions)
    return permutation


def subsampled_cluster_stability(
    distances: Array,
    cluster_count: int,
    *,
    repetitions: int,
    sample_fraction: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compare a full clustering with cuts after repeatedly removing items."""

    matrix = _validate_distances(distances)
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0.5 <= sample_fraction < 1.0:
        raise ValueError("sample_fraction must lie in [0.5, 1)")
    sample_size = max(cluster_count + 1, int(np.ceil(sample_fraction * len(matrix))))
    if sample_size >= len(matrix):
        raise ValueError("sample_fraction leaves no items out")

    full_labels = average_linkage_labels(matrix, cluster_count)
    scores = []
    for _ in range(repetitions):
        indices = np.sort(rng.choice(len(matrix), size=sample_size, replace=False))
        subset = matrix[np.ix_(indices, indices)]
        subset_labels = average_linkage_labels(subset, cluster_count)
        scores.append(adjusted_rand_score(full_labels[indices], subset_labels))
    values = np.asarray(scores, dtype=np.float64)
    return {
        "mean_adjusted_rand": float(np.mean(values)),
        "median_adjusted_rand": float(np.median(values)),
        "q05_adjusted_rand": float(np.quantile(values, 0.05)),
        "q95_adjusted_rand": float(np.quantile(values, 0.95)),
    }
