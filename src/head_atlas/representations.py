"""Complementary geometric and functional views of matrix operators."""

import numpy as np

Array = np.ndarray


def frobenius_normalize(matrix: Array, eps: float = 1e-12) -> Array:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(matrix, ord="fro")
    if norm <= eps:
        raise ValueError("cannot normalize a near-zero operator")
    return matrix / norm


def normalized_spectrum(matrix: Array, eps: float = 1e-12) -> Array:
    matrix = np.asarray(matrix, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    norm = np.linalg.norm(singular_values)
    if norm <= eps:
        raise ValueError("near-zero operator has no normalized spectrum")
    return singular_values / norm


def effective_rank(matrix: Array, relative_tolerance: float | None = None) -> float:
    """Entropy-based effective rank after removing numerical singular values.

    The default tolerance reflects the precision of the input matrix before the
    SVD is evaluated in float64. A caller can provide a different relative
    tolerance when the matrix has a known noise floor.
    """

    input_matrix = np.asarray(matrix)
    if not np.issubdtype(input_matrix.dtype, np.inexact):
        input_matrix = input_matrix.astype(np.float64)
    input_epsilon = np.finfo(input_matrix.dtype).eps
    matrix_64 = input_matrix.astype(np.float64, copy=False)
    singular_values = np.linalg.svd(matrix_64, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0.0
    if relative_tolerance is None:
        relative_tolerance = max(matrix_64.shape) * input_epsilon
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be nonnegative")
    singular_values = singular_values[singular_values > relative_tolerance * singular_values[0]]
    total = singular_values.sum()
    if total == 0:
        return 0.0
    probabilities = singular_values / total
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.exp(entropy))


def leading_subspace_projector(matrix: Array, rank: int, side: str) -> Array:
    """Projector onto a leading OV read or write subspace.

    With row-vector actions ``x @ M``, left singular vectors span the read
    subspace and right singular vectors span the write subspace under the
    chosen matrix convention. The names are declared explicitly here so tests
    and downstream reports do not silently swap them.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if rank < 1 or rank > min(matrix.shape):
        raise ValueError("rank is outside the matrix dimensions")
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    if side == "write":
        basis = vt[:rank].T
    elif side == "read":
        basis = u[:, :rank]
    else:
        raise ValueError("side must be 'read' or 'write'")
    return basis @ basis.T


def projector_distance(projector_a: Array, projector_b: Array) -> float:
    return float(np.linalg.norm(np.asarray(projector_a) - np.asarray(projector_b), ord="fro"))


def empirical_action_distance(a: Array, b: Array, activations: Array) -> float:
    """RMS difference between row-vector operator outputs on activations."""

    delta_outputs = np.asarray(activations) @ (np.asarray(a) - np.asarray(b))
    return float(np.sqrt(np.mean(np.sum(delta_outputs**2, axis=-1))))


def empirical_qk_score_distance(a: Array, b: Array, queries: Array, keys: Array) -> float:
    """RMS difference in bilinear QK scores on paired query/key states."""

    queries = np.asarray(queries)
    keys = np.asarray(keys)
    if queries.shape != keys.shape:
        raise ValueError("queries and keys must contain paired vectors")
    delta = np.asarray(a) - np.asarray(b)
    differences = np.einsum("...i,ij,...j->...", queries, delta, keys)
    return float(np.sqrt(np.mean(differences**2)))
