"""Population summaries used to compare real and null distance geometry."""

import numpy as np

Array = np.ndarray


def _validated_distances(distance_matrix: Array, tolerance: float) -> Array:
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    if distances.shape[0] < 2:
        raise ValueError("distance matrix must contain at least two items")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if not np.isfinite(distances).all():
        raise ValueError("distance matrix contains non-finite values")
    if not np.allclose(distances, distances.T, rtol=0.0, atol=tolerance):
        raise ValueError("distance matrix must be symmetric")
    if not np.allclose(np.diag(distances), 0.0, rtol=0.0, atol=tolerance):
        raise ValueError("distance matrix diagonal must be zero")
    if np.min(distances) < -tolerance:
        raise ValueError("distance matrix contains negative values")
    return np.maximum(distances, 0.0)


def summarize_distance_matrix(
    distance_matrix: Array, tolerance: float = 1e-10
) -> dict[str, int | float]:
    """Summarize pairwise and nearest-neighbour distance distributions."""

    distances = _validated_distances(distance_matrix, tolerance)
    item_count = distances.shape[0]
    pairwise = distances[np.triu_indices(item_count, k=1)]
    without_diagonal = np.where(np.eye(item_count, dtype=bool), np.inf, distances)
    nearest = np.min(without_diagonal, axis=1)

    return {
        "item_count": item_count,
        "pair_count": pairwise.size,
        "pair_minimum": float(np.min(pairwise)),
        "pair_q01": float(np.quantile(pairwise, 0.01)),
        "pair_q05": float(np.quantile(pairwise, 0.05)),
        "pair_median": float(np.median(pairwise)),
        "pair_mean": float(np.mean(pairwise)),
        "pair_standard_deviation": float(np.std(pairwise)),
        "pair_q95": float(np.quantile(pairwise, 0.95)),
        "pair_maximum": float(np.max(pairwise)),
        "nearest_minimum": float(np.min(nearest)),
        "nearest_median": float(np.median(nearest)),
        "nearest_mean": float(np.mean(nearest)),
        "nearest_maximum": float(np.max(nearest)),
    }
