"""Diagnostics for comparing learned operator-motif dictionaries."""

import numpy as np
from scipy.optimize import linear_sum_assignment

Array = np.ndarray


def matched_atom_similarities(first: Array, second: Array, eps: float = 1e-12) -> Array:
    """Return absolute cosines for optimally matched atoms in first-dictionary order."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.ndim != 2 or second_array.ndim != 2:
        raise ValueError("dictionaries must be two-dimensional")
    if first_array.shape != second_array.shape or first_array.shape[0] < 1:
        raise ValueError("dictionaries must have the same nonempty shape")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("dictionaries contain non-finite values")
    first_norms = np.linalg.norm(first_array, axis=1)
    second_norms = np.linalg.norm(second_array, axis=1)
    if np.any(first_norms <= eps) or np.any(second_norms <= eps):
        raise ValueError("dictionary atoms must be nonzero")
    normalized_first = first_array / first_norms[:, None]
    normalized_second = second_array / second_norms[:, None]
    similarities = np.abs(normalized_first @ normalized_second.T)
    rows, columns = linear_sum_assignment(-similarities)
    matched = np.empty(first_array.shape[0], dtype=np.float64)
    matched[rows] = similarities[rows, columns]
    return matched


def matched_atom_similarity(first: Array, second: Array, eps: float = 1e-12) -> float:
    """Mean absolute cosine after optimal one-to-one atom matching.

    Dictionary atoms have arbitrary order and sign. Hungarian matching removes
    those symmetries before measuring reproducibility.
    """

    return float(np.mean(matched_atom_similarities(first, second, eps=eps)))
