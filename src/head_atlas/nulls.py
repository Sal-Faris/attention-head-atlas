"""Null operators that preserve progressively more observed structure."""

import numpy as np

Array = np.ndarray


def haar_orthogonal(size: int, rng: np.random.Generator) -> Array:
    """Sample an orthogonal matrix with Haar distribution via corrected QR."""

    q, r = np.linalg.qr(rng.standard_normal((size, size)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def spectrum_matched_rotation(matrix: Array, rng: np.random.Generator) -> Array:
    """Randomize singular directions while preserving singular values exactly."""

    matrix = np.asarray(matrix)
    rows, cols = matrix.shape
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    left = haar_orthogonal(rows, rng)[:, : singular_values.size]
    right = haar_orthogonal(cols, rng)[:, : singular_values.size]
    return (left * singular_values) @ right.T


def rank_norm_matched_gaussian(matrix: Array, rng: np.random.Generator) -> Array:
    """Gaussian low-rank null with observed algebraic rank and Frobenius norm."""

    matrix = np.asarray(matrix)
    rank = int(np.linalg.matrix_rank(matrix))
    target_norm = np.linalg.norm(matrix, ord="fro")
    if rank == 0 or target_norm == 0:
        return np.zeros_like(matrix, dtype=np.float64)
    left = rng.standard_normal((matrix.shape[0], rank))
    right = rng.standard_normal((rank, matrix.shape[1]))
    null = left @ right
    return null * (target_norm / np.linalg.norm(null, ord="fro"))
