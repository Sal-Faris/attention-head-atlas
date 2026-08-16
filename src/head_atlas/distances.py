"""Pairwise distances between populations of attention-head operators."""

from collections.abc import Sequence

import numpy as np

from head_atlas.operators import HeadOperator

Array = np.ndarray


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
