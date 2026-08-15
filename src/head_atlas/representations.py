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
    singular_values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    norm = np.linalg.norm(singular_values)
    if norm <= eps:
        raise ValueError("near-zero operator has no normalized spectrum")
    return singular_values / norm


def effective_rank(matrix: Array, eps: float = 1e-12) -> float:
    singular_values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    total = singular_values.sum()
    if total <= eps:
        return 0.0
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > eps]
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(np.exp(entropy))


def leading_subspace_projector(matrix: Array, rank: int, side: str) -> Array:
    """Projector onto a leading OV read or write subspace.

    With row-vector actions ``x @ M``, right singular vectors span the read
    subspace and left singular vectors span the write subspace under the
    chosen matrix convention. The names are declared explicitly here so tests
    and downstream reports do not silently swap them.
    """

    matrix = np.asarray(matrix)
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


def empirical_qk_score_distance(
    a: Array, b: Array, queries: Array, keys: Array
) -> float:
    """RMS difference in bilinear QK scores on paired query/key states."""

    queries = np.asarray(queries)
    keys = np.asarray(keys)
    if queries.shape != keys.shape:
        raise ValueError("queries and keys must contain paired vectors")
    delta = np.asarray(a) - np.asarray(b)
    differences = np.einsum("...i,ij,...j->...", queries, delta, keys)
    return float(np.sqrt(np.mean(differences**2)))

