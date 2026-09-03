"""Activation-behavior distances and matched permutation tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .families import stratified_permutation

Array = np.ndarray


def normalized_distances_from_gram(gram: Array, eps: float = 1e-12) -> Array:
    """Convert a feature Gram matrix to scale-free Euclidean distances."""

    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("gram must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("gram contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-8):
        raise ValueError("gram must be symmetric")
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    squared_norms = np.diag(matrix)
    if np.any(squared_norms <= eps):
        raise ValueError("gram contains a near-zero feature vector")
    denominator = np.sqrt(squared_norms[:, None] * squared_norms[None, :])
    similarities = np.clip(matrix / denominator, -1.0, 1.0)
    distances = np.sqrt(np.maximum(2.0 - 2.0 * similarities, 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def distance_spearman(first: Array, second: Array) -> float:
    """Spearman correlation between two aligned distance geometries."""

    first_matrix = np.asarray(first, dtype=np.float64)
    second_matrix = np.asarray(second, dtype=np.float64)
    if first_matrix.shape != second_matrix.shape:
        raise ValueError("distance matrices must have matching shapes")
    if first_matrix.ndim != 2 or first_matrix.shape[0] != first_matrix.shape[1]:
        raise ValueError("distance matrices must be square")
    if not np.isfinite(first_matrix).all() or not np.isfinite(second_matrix).all():
        raise ValueError("distance matrices contain non-finite values")
    upper = np.triu_indices(len(first_matrix), 1)
    statistic = float(spearmanr(first_matrix[upper], second_matrix[upper]).statistic)
    if not np.isfinite(statistic):
        raise ValueError("distance correlation is undefined")
    return statistic


def stratified_distance_permutation_test(
    predictor: Array,
    target: Array,
    groups: Array,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Test distance association while preserving target group geometry."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    group_array = np.asarray(groups)
    if group_array.shape != (len(np.asarray(predictor)),):
        raise ValueError("groups must contain one value per item")
    observed = distance_spearman(predictor, target)
    null_values = []
    target_matrix = np.asarray(target)
    for _ in range(repetitions):
        permutation = stratified_permutation(group_array, rng)
        permuted = target_matrix[np.ix_(permutation, permutation)]
        null_values.append(distance_spearman(predictor, permuted))
    values = np.asarray(null_values, dtype=np.float64)
    deviation = float(np.std(values))
    return {
        "observed_spearman": observed,
        "null_mean": float(np.mean(values)),
        "null_standard_deviation": deviation,
        "z_score": (
            float((observed - np.mean(values)) / deviation) if deviation > 0 else float("inf")
        ),
        "upper_tail_p_value": float(
            (1 + np.count_nonzero(values >= observed)) / (repetitions + 1)
        ),
    }


def stratified_predictor_difference_test(
    primary: Array,
    baseline: Array,
    target: Array,
    groups: Array,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Test whether one fixed predictor correlates better than another."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    group_array = np.asarray(groups)
    target_matrix = np.asarray(target)
    observed = distance_spearman(primary, target_matrix) - distance_spearman(
        baseline, target_matrix
    )
    null_values = []
    for _ in range(repetitions):
        permutation = stratified_permutation(group_array, rng)
        permuted = target_matrix[np.ix_(permutation, permutation)]
        null_values.append(
            distance_spearman(primary, permuted) - distance_spearman(baseline, permuted)
        )
    values = np.asarray(null_values, dtype=np.float64)
    return {
        "observed_spearman_difference": observed,
        "null_mean": float(np.mean(values)),
        "upper_tail_p_value": float(
            (1 + np.count_nonzero(values >= observed)) / (repetitions + 1)
        ),
    }


def layer_pair_matched_edge_test(
    target: Array,
    layers: Array,
    edges: Array,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Test whether cross-layer edges are closer than layer-pair-matched pairs."""

    matrix = np.asarray(target, dtype=np.float64)
    layer_array = np.asarray(layers)
    edge_array = np.asarray(edges, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("target must be a square distance matrix")
    if layer_array.shape != (len(matrix),):
        raise ValueError("layers must contain one value per item")
    if edge_array.ndim != 2 or edge_array.shape[1] != 2 or len(edge_array) < 1:
        raise ValueError("edges must have shape (edge_count, 2)")
    if np.min(edge_array) < 0 or np.max(edge_array) >= len(matrix):
        raise ValueError("edge index is outside the population")
    if np.any(layer_array[edge_array[:, 0]] == layer_array[edge_array[:, 1]]):
        raise ValueError("all tested edges must cross layers")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    observed = float(np.mean(matrix[edge_array[:, 0], edge_array[:, 1]]))
    indices_by_layer = {
        layer: np.flatnonzero(layer_array == layer) for layer in np.unique(layer_array)
    }
    null_values = []
    for _ in range(repetitions):
        sampled = []
        for first, second in edge_array:
            sampled_first = int(rng.choice(indices_by_layer[layer_array[first]]))
            sampled_second = int(rng.choice(indices_by_layer[layer_array[second]]))
            sampled.append(matrix[sampled_first, sampled_second])
        null_values.append(float(np.mean(sampled)))
    values = np.asarray(null_values, dtype=np.float64)
    deviation = float(np.std(values))
    return {
        "edge_count": len(edge_array),
        "observed_mean_distance": observed,
        "null_mean": float(np.mean(values)),
        "null_standard_deviation": deviation,
        "z_score": (
            float((observed - np.mean(values)) / deviation) if deviation > 0 else float("-inf")
        ),
        "observed_to_null_mean_ratio": float(observed / np.mean(values)),
        "lower_tail_p_value": float(
            (1 + np.count_nonzero(values <= observed)) / (repetitions + 1)
        ),
    }
