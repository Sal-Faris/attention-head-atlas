"""Null operators that preserve progressively more observed structure."""

import numpy as np

Array = np.ndarray


def sample_norm_matched_isotropic(
    vectors: Array, rng: np.random.Generator
) -> Array:
    """Randomize vector directions independently while preserving row norms."""

    vector_array = np.asarray(vectors, dtype=np.float64)
    if vector_array.ndim != 2 or vector_array.shape[0] < 1 or vector_array.shape[1] < 1:
        raise ValueError("vectors must be a nonempty two-dimensional array")
    if not np.isfinite(vector_array).all():
        raise ValueError("vectors contain non-finite values")
    target_norms = np.linalg.norm(vector_array, axis=1)
    if np.any(target_norms == 0):
        raise ValueError("cannot randomize zero vectors")
    samples = rng.standard_normal(vector_array.shape)
    sample_norms = np.linalg.norm(samples, axis=1)
    return samples * (target_norms / sample_norms)[:, None]


def resolved_singular_values(matrix: Array) -> Array:
    """Singular values distinguishable at the supplied matrix precision."""

    input_matrix = np.asarray(matrix)
    if not np.issubdtype(input_matrix.dtype, np.inexact):
        input_matrix = input_matrix.astype(np.float64)
    input_epsilon = np.finfo(input_matrix.dtype).eps
    matrix_64 = input_matrix.astype(np.float64, copy=False)
    singular_values = np.linalg.svd(matrix_64, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return np.empty(0, dtype=np.float64)
    tolerance = max(matrix_64.shape) * input_epsilon * singular_values[0]
    return singular_values[singular_values > tolerance]


def _numerical_rank(matrix: Array) -> int:
    """Rank resolved at the precision of the supplied matrix."""

    return int(resolved_singular_values(matrix).size)


def _haar_orthonormal_frame(
    size: int, columns: int, rng: np.random.Generator
) -> Array:
    """Sample a uniform orthonormal frame without constructing a full basis."""

    if columns < 0 or columns > size:
        raise ValueError("frame columns must be between zero and size")
    if columns == 0:
        return np.empty((size, 0), dtype=np.float64)
    q, r = np.linalg.qr(rng.standard_normal((size, columns)), mode="reduced")
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def haar_orthonormal_frame(
    size: int, columns: int, rng: np.random.Generator
) -> Array:
    """Sample a Haar-uniform orthonormal frame with the requested shape."""

    return _haar_orthonormal_frame(size, columns, rng)


def haar_orthogonal(size: int, rng: np.random.Generator) -> Array:
    """Sample an orthogonal matrix with Haar distribution via corrected QR."""

    return _haar_orthonormal_frame(size, size, rng)


def spectrum_matched_rotation(matrix: Array, rng: np.random.Generator) -> Array:
    """Randomize directions while preserving numerically resolved singular values."""

    input_matrix = np.asarray(matrix)
    singular_values = resolved_singular_values(input_matrix)
    return sample_spectrum_matched(singular_values, input_matrix.shape, rng)


def sample_spectrum_matched(
    singular_values: Array,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> Array:
    """Sample a null from a precomputed singular spectrum and matrix shape."""

    singular_values = np.asarray(singular_values, dtype=np.float64)
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("shape must contain two positive dimensions")
    if singular_values.ndim != 1:
        raise ValueError("singular values must be one-dimensional")
    if singular_values.size > min(shape):
        raise ValueError("too many singular values for the requested shape")
    if not np.isfinite(singular_values).all() or np.any(singular_values < 0):
        raise ValueError("singular values must be finite and nonnegative")

    rows, cols = shape
    if singular_values.size == 0:
        return np.zeros(shape, dtype=np.float64)
    left = _haar_orthonormal_frame(rows, singular_values.size, rng)
    right = _haar_orthonormal_frame(cols, singular_values.size, rng)
    return (left * singular_values) @ right.T


def rank_norm_matched_gaussian(matrix: Array, rng: np.random.Generator) -> Array:
    """Gaussian low-rank null with observed algebraic rank and Frobenius norm."""

    matrix = np.asarray(matrix)
    rank = _numerical_rank(matrix)
    target_norm = np.linalg.norm(matrix, ord="fro")
    if rank == 0 or target_norm == 0:
        return np.zeros_like(matrix, dtype=np.float64)
    left = rng.standard_normal((matrix.shape[0], rank))
    right = rng.standard_normal((rank, matrix.shape[1]))
    null = left @ right
    return null * (target_norm / np.linalg.norm(null, ord="fro"))
