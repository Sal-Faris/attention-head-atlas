"""Pairwise distances between populations of attention-head operators."""

from collections.abc import Sequence

import numpy as np

from head_atlas.operators import HeadOperator

Array = np.ndarray


def chordal_subspace_distances(bases: Array, tolerance: float = 1e-5) -> Array:
    """Pairwise normalized chordal distances between equal-rank subspaces.

    ``bases[i]`` contains orthonormal columns spanning one subspace. The result
    is ``sqrt(1 - trace(P_i P_j) / rank)``, where ``P_i`` is its projector.
    It lies in ``[0, 1]`` and is unchanged by rotating the columns of any basis
    or by applying the same orthogonal coordinate change to every subspace.
    """

    basis_array = np.asarray(bases)
    if basis_array.ndim != 3:
        raise ValueError("bases must have shape (items, dimensions, rank)")
    item_count, dimensions, rank = basis_array.shape
    if item_count < 1 or rank < 1 or rank > dimensions:
        raise ValueError("invalid subspace-basis dimensions")
    if not np.isfinite(basis_array).all():
        raise ValueError("bases contain non-finite values")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    identity = np.eye(rank)
    for basis in basis_array:
        if not np.allclose(basis.T @ basis, identity, rtol=0.0, atol=tolerance):
            raise ValueError("basis columns must be orthonormal")

    projectors = np.einsum(
        "ndk,nek->nde", basis_array, basis_array, optimize=True
    ).reshape(item_count, -1)
    overlaps = projectors @ projectors.T
    normalized_overlaps = np.clip(overlaps / rank, 0.0, 1.0)
    distances = np.sqrt(np.maximum(1.0 - normalized_overlaps, 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def weighted_product_distances(
    distance_matrices: Sequence[Array],
    weights: Sequence[float] | None = None,
    tolerance: float = 1e-10,
) -> Array:
    """Combine Euclidean views as a weighted orthogonal product space.

    If ``D_v`` is the Euclidean distance in view ``v``, the combined distance
    is ``sqrt(sum_v w_v D_v**2 / sum_v w_v)``. Equal weights therefore give QK
    and OV equal influence without fitting a combination to functional labels.
    """

    if not distance_matrices:
        raise ValueError("at least one distance matrix is required")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in distance_matrices]
    expected_shape = matrices[0].shape
    if len(expected_shape) != 2 or expected_shape[0] != expected_shape[1]:
        raise ValueError("distance matrices must be square")
    for matrix in matrices:
        if matrix.shape != expected_shape:
            raise ValueError("distance matrices must have the same shape")
        if not np.isfinite(matrix).all():
            raise ValueError("distance matrix contains non-finite values")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
            raise ValueError("distance matrices must be symmetric")
        if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=tolerance):
            raise ValueError("distance matrices must have zero diagonals")
        if np.min(matrix) < -tolerance:
            raise ValueError("distance matrices cannot contain negative values")

    if weights is None:
        weight_array = np.ones(len(matrices), dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (len(matrices),):
            raise ValueError("weights must contain one value per distance matrix")
        if not np.isfinite(weight_array).all() or np.any(weight_array < 0):
            raise ValueError("weights must be finite and nonnegative")
        if np.sum(weight_array) == 0:
            raise ValueError("at least one weight must be positive")

    squared = np.zeros(expected_shape, dtype=np.float64)
    for weight, matrix in zip(weight_array, matrices, strict=True):
        squared += weight * np.maximum(matrix, 0.0) ** 2
    combined = np.sqrt(np.maximum(squared / np.sum(weight_array), 0.0))
    np.fill_diagonal(combined, 0.0)
    return combined


def normalized_frobenius_distances(
    operators: Sequence[HeadOperator], eps: float = 1e-12
) -> Array:
    """Return all pairwise distances after removing operator scale.

    For operators ``M_i`` and ``M_j``, the distance is

    ``||M_i / ||M_i||_F - M_j / ||M_j||_F||_F``.

    The calculation uses the equivalent inner-product formula so all pairs
    can be computed together. Signs are retained: an operator and its negative
    have the maximum distance of 2 rather than being treated as equivalent.
    """

    if not operators:
        raise ValueError("at least one operator is required")
    if eps < 0:
        raise ValueError("eps must be nonnegative")

    expected_kind = operators[0].kind
    expected_shape = operators[0].matrix.shape
    if any(operator.kind != expected_kind for operator in operators):
        raise ValueError("operators must all have the same kind")
    if any(operator.matrix.shape != expected_shape for operator in operators):
        raise ValueError("operators must all have the same matrix shape")

    flattened = np.stack(
        [np.asarray(operator.matrix, dtype=np.float64).reshape(-1) for operator in operators]
    )
    norms = np.linalg.norm(flattened, axis=1)
    if np.any(norms <= eps):
        raise ValueError("cannot compare a near-zero operator")

    normalized = flattened / norms[:, None]
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    squared_distances = np.maximum(2.0 - 2.0 * similarities, 0.0)
    distances = np.sqrt(squared_distances)
    np.fill_diagonal(distances, 0.0)
    return distances
