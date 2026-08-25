"""Unrestricted population modes and their emergent operator dimensions."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def principal_operator_modes(matrices: Array, components: int) -> Array:
    """Return Frobenius-orthonormal, uncentered PCA modes of square matrices."""

    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("matrices must have shape (population, dimension, dimension)")
    if components < 1:
        raise ValueError("components must be positive")
    gram = np.einsum("nij,mij->nm", values, values, optimize=True)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    keep = min(components, int(np.sum(eigenvalues > 1e-12)))
    if keep == 0:
        raise ValueError("matrix population has no nonzero mode")
    modes = np.einsum(
        "nk,nij->kij",
        eigenvectors[:, :keep] / np.sqrt(eigenvalues[:keep]),
        values,
        optimize=True,
    )
    return modes


def truncate_operator_modes(modes: Array, rank: int) -> Array:
    """Independently truncate unrestricted operator modes to a chosen rank."""

    values = np.asarray(modes, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("modes must have shape (count, dimension, dimension)")
    if rank < 1:
        raise ValueError("rank must be positive")
    truncated = []
    for mode in values:
        left, singular, right = np.linalg.svd(mode, full_matrices=False)
        keep = min(rank, len(singular))
        truncated.append((left[:, :keep] * singular[:keep]) @ right[:keep])
    return np.stack(truncated)


def dictionary_variance_recovered(matrices: Array, dictionary: Array) -> float:
    """Return mean full-matrix variance recovered by a possibly nonorthogonal span."""

    values = np.asarray(matrices, dtype=np.float64).reshape(len(matrices), -1)
    atoms = np.asarray(dictionary, dtype=np.float64).reshape(len(dictionary), -1)
    gram = atoms @ atoms.T
    coefficients = np.linalg.lstsq(gram, (values @ atoms.T).T, rcond=1e-10)[0].T
    reconstruction = coefficients @ atoms
    squared_error = np.sum((values - reconstruction) ** 2, axis=1)
    return float(np.mean(1.0 - squared_error))


def singular_dimension_summary(matrix: Array) -> dict[str, float | int]:
    """Summarize intrinsic dimensions without selecting a decomposition rank."""

    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return singular_values_dimension_summary(singular)


def singular_values_dimension_summary(singular_values: Array) -> dict[str, float | int]:
    """Summarize intrinsic dimensions from a complete singular spectrum."""

    singular = np.asarray(singular_values, dtype=np.float64)
    if singular.ndim != 1 or len(singular) == 0 or np.any(singular < 0):
        raise ValueError("singular values must be a nonempty nonnegative vector")
    energy = singular**2
    energy /= max(float(np.sum(energy)), 1e-24)
    positive = energy[energy > 1e-18]
    cumulative = np.cumsum(energy)

    def energy_rank(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold) + 1)

    eligible = np.flatnonzero(cumulative[:-1] <= 0.99)
    if len(eligible):
        ratios = singular[eligible] / np.maximum(singular[eligible + 1], 1e-15)
        gap_index = int(eligible[int(np.argmax(ratios))])
        gap_rank = gap_index + 1
        gap_ratio = float(ratios.max())
    else:
        gap_rank = 1
        gap_ratio = 1.0
    return {
        "stable_rank": float(1.0 / max(energy[0], 1e-24)),
        "entropy_rank": float(np.exp(-np.sum(positive * np.log(positive)))),
        "rank_50_percent_energy": energy_rank(0.50),
        "rank_80_percent_energy": energy_rank(0.80),
        "rank_90_percent_energy": energy_rank(0.90),
        "rank_95_percent_energy": energy_rank(0.95),
        "largest_gap_rank_below_99_percent_energy": gap_rank,
        "largest_gap_ratio": gap_ratio,
    }
