"""Statistics that distinguish population structure from attractive projections."""

from collections.abc import Sequence

import numpy as np

from .embedding import classical_mds

Array = np.ndarray


def _centered_gram(distance_matrix: Array) -> Array:
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.isfinite(distances).all():
        raise ValueError("distance matrix contains non-finite values")
    squared = distances**2
    row_means = np.mean(squared, axis=1, keepdims=True)
    column_means = np.mean(squared, axis=0, keepdims=True)
    gram = -0.5 * (squared - row_means - column_means + np.mean(squared))
    return 0.5 * (gram + gram.T)


def residualize_euclidean_distances(
    distance_matrix: Array,
    predictors: Array,
    tolerance: float = 1e-10,
) -> Array:
    """Remove linear predictor directions from an entire Euclidean geometry.

    This is equivalent to regressing every (possibly implicit) feature column
    on the centered predictors, then recomputing distances between the
    residual feature vectors.  Working through the centered Gram matrix avoids
    materializing high-dimensional flattened operators.
    """

    gram = _centered_gram(distance_matrix)
    design = np.asarray(predictors, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2 or design.shape[0] != gram.shape[0]:
        raise ValueError("predictors must have one row per item")
    if not np.isfinite(design).all():
        raise ValueError("predictors contain non-finite values")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    centered_design = design - np.mean(design, axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered_design, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    if rank == 0:
        raise ValueError("predictors contain no nonconstant variation")
    if rank >= len(centered_design) - 1:
        raise ValueError("predictor design leaves no residual degrees of freedom")

    basis = left[:, :rank]
    residualizer = np.eye(len(gram)) - basis @ basis.T
    residual_gram = residualizer @ gram @ residualizer
    residual_gram = 0.5 * (residual_gram + residual_gram.T)

    eigenvalues = np.linalg.eigvalsh(residual_gram)
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -tolerance * eigenvalue_scale:
        raise ValueError("distance matrix is not Euclidean within tolerance")

    diagonal = np.diag(residual_gram)
    squared_distances = diagonal[:, None] + diagonal[None, :] - 2.0 * residual_gram
    squared_distances = np.maximum(squared_distances, 0.0)
    residual_distances = np.sqrt(squared_distances)
    np.fill_diagonal(residual_distances, 0.0)
    return residual_distances


def pcoa_spectrum_summary(
    distance_matrix: Array,
    tolerance: float = 1e-10,
) -> dict[str, int | float]:
    """Summarize how population variance is distributed over PCoA axes."""

    result = classical_mds(distance_matrix, tolerance=tolerance)
    eigenvalues = result["eigenvalues"]
    explained = result["explained_variance_ratio"]
    if not isinstance(eigenvalues, np.ndarray) or not isinstance(explained, np.ndarray):
        raise TypeError("classical_mds returned malformed spectrum arrays")

    positive = explained[explained > 0]
    cumulative = np.cumsum(positive)
    if positive.size == 0:
        participation_dimension = 0.0
    else:
        participation_dimension = float(1.0 / np.sum(positive**2))

    def retained_variance(dimensions: int) -> float:
        return float(np.sum(positive[:dimensions]))

    def dimensions_for(target: float) -> int:
        if cumulative.size == 0:
            return 0
        return int(np.searchsorted(cumulative, target) + 1)

    return {
        "item_count": int(np.asarray(distance_matrix).shape[0]),
        "positive_dimensions": int(positive.size),
        "participation_dimension": participation_dimension,
        "top_1_variance": retained_variance(1),
        "top_2_variance": retained_variance(2),
        "top_5_variance": retained_variance(5),
        "top_10_variance": retained_variance(10),
        "dimensions_for_50_percent": dimensions_for(0.50),
        "dimensions_for_80_percent": dimensions_for(0.80),
        "dimensions_for_90_percent": dimensions_for(0.90),
        "dimensions_for_95_percent": dimensions_for(0.95),
        "negative_eigenvalue_mass_ratio": float(
            result["negative_eigenvalue_mass_ratio"]
        ),
        "largest_eigenvalue": float(eigenvalues[0]),
    }


def categorical_permanova(
    distance_matrix: Array,
    labels: Sequence[object],
    permutations: int = 999,
    seed: int = 0,
    tolerance: float = 1e-10,
) -> dict[str, int | float]:
    """Test one categorical predictor using distance-based sums of squares.

    The effect size is the fraction of centered PCoA sum of squares explained
    by group centroids.  The plus-one permutation p-value asks how often a
    random reassignment preserving group sizes produces at least as large a
    pseudo-F statistic.
    """

    distances = np.asarray(distance_matrix, dtype=np.float64)
    label_array = np.asarray(labels)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    if label_array.ndim != 1 or label_array.size != distances.shape[0]:
        raise ValueError("labels must contain one value per item")
    if permutations < 0:
        raise ValueError("permutations must be nonnegative")

    gram = _centered_gram(distances)

    total_sum_squares = float(np.trace(gram))
    if total_sum_squares <= tolerance:
        raise ValueError("distance matrix has no nonzero centered variation")

    _, encoded_labels = np.unique(label_array, return_inverse=True)
    group_count = int(np.max(encoded_labels) + 1)
    item_count = label_array.size
    if group_count < 2:
        raise ValueError("at least two groups are required")
    if group_count >= item_count:
        raise ValueError("each group cannot contain exactly one item")

    def statistic(encoded: Array) -> tuple[float, float]:
        design = np.eye(group_count, dtype=np.float64)[encoded]
        group_sizes = np.sum(design, axis=0)
        projection = (design / group_sizes) @ design.T
        between = float(np.sum(projection * gram.T))
        between = min(max(between, 0.0), total_sum_squares)
        within = max(total_sum_squares - between, 0.0)
        degrees_between = group_count - 1
        degrees_within = item_count - group_count
        pseudo_f = (between / degrees_between) / (within / degrees_within)
        return between / total_sum_squares, pseudo_f

    explained_fraction, observed_f = statistic(encoded_labels)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        _, permuted_f = statistic(rng.permutation(encoded_labels))
        exceedances += permuted_f >= observed_f

    return {
        "item_count": int(item_count),
        "group_count": group_count,
        "explained_variance_fraction": explained_fraction,
        "pseudo_f": observed_f,
        "permutations": permutations,
        "seed": seed,
        "p_value": float((exceedances + 1) / (permutations + 1)),
    }


def design_permanova(
    distance_matrix: Array,
    predictors: Array,
    permutations: int = 999,
    seed: int = 0,
    tolerance: float = 1e-10,
) -> dict[str, int | float]:
    """Test the overall association of one or more predictors with geometry.

    Predictor columns are centered, so an explicit intercept is unnecessary.
    Permutations move complete predictor rows together and therefore preserve
    correlations among predictors. This tests the design as a whole; it does
    not estimate conditional significance for individual columns.
    """

    gram = _centered_gram(distance_matrix)
    design = np.asarray(predictors, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2 or design.shape[0] != gram.shape[0]:
        raise ValueError("predictors must have one row per item")
    if not np.isfinite(design).all():
        raise ValueError("predictors contain non-finite values")
    if permutations < 0:
        raise ValueError("permutations must be nonnegative")

    total_sum_squares = float(np.trace(gram))
    if total_sum_squares <= tolerance:
        raise ValueError("distance matrix has no nonzero centered variation")

    centered_design = design - np.mean(design, axis=0, keepdims=True)

    left, singular_values, _ = np.linalg.svd(centered_design, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    if rank == 0:
        raise ValueError("predictors contain no nonconstant variation")
    if rank >= len(centered_design) - 1:
        raise ValueError("predictor design leaves no residual degrees of freedom")
    basis = left[:, :rank]

    def statistic(candidate_basis: Array) -> tuple[float, float]:
        between = float(np.sum(candidate_basis * (gram @ candidate_basis)))
        between = min(max(between, 0.0), total_sum_squares)
        within = max(total_sum_squares - between, 0.0)
        if within <= tolerance * total_sum_squares:
            pseudo_f = float("inf")
        else:
            pseudo_f = (between / rank) / (within / (len(candidate_basis) - rank - 1))
        return between / total_sum_squares, pseudo_f

    explained_fraction, observed_f = statistic(basis)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        _, permuted_f = statistic(rng.permutation(basis, axis=0))
        exceedances += permuted_f >= observed_f

    return {
        "item_count": len(centered_design),
        "predictor_rank": rank,
        "explained_variance_fraction": explained_fraction,
        "pseudo_f": observed_f,
        "permutations": permutations,
        "seed": seed,
        "p_value": float((exceedances + 1) / (permutations + 1)),
    }
