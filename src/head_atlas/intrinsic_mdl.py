"""Intrinsic transformation-profile and description-length accounting.

These routines separate the predictability of a singular-value profile from
the cost of locating its input and output frames in the residual stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RankDescription:
    """Degrees of freedom for a rank-k square matrix and fixed-spectrum one."""

    width: int
    rank: int
    unrestricted: int
    fixed_normalized_spectrum: int
    maximum_reusable_core_saving: int

    @property
    def saving_fraction(self) -> float:
        return self.maximum_reusable_core_saving / self.unrestricted


def rank_description(width: int, rank: int) -> RankDescription:
    """Return exact manifold dimensions under independent orthogonal gauges."""

    if width < 1 or rank < 1 or rank > width:
        raise ValueError("require 1 <= rank <= width")
    unrestricted = 2 * width * rank - rank**2
    saving = rank - 1
    return RankDescription(
        width=width,
        rank=rank,
        unrestricted=unrestricted,
        fixed_normalized_spectrum=unrestricted - saving,
        maximum_reusable_core_saving=saving,
    )


def normalize_spectra(spectra: Array) -> Array:
    """Normalize every nonnegative singular-value vector to unit L2 norm."""

    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("spectra must be a nonempty two-dimensional array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("spectra must be finite and nonnegative")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("spectra must have positive norm")
    return values / norms


def rank_energy_curve(normalized_spectra: Array, ranks: list[int]) -> dict[int, float]:
    """Mean operator energy recovered by each leading-rank truncation."""

    spectra = normalize_spectra(normalized_spectra)
    result: dict[int, float] = {}
    for rank in ranks:
        if rank < 1 or rank > spectra.shape[1]:
            raise ValueError("rank is outside the available spectrum")
        result[rank] = float(np.mean(np.sum(spectra[:, :rank] ** 2, axis=1)))
    return result


def parity_splits(labels: Array) -> list[tuple[Array, Array]]:
    """Return two complementary complete-label parity splits."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("labels must be a one-dimensional population")
    splits = []
    for parity in (0, 1):
        test = np.flatnonzero(values % 2 == parity)
        train = np.flatnonzero(values % 2 != parity)
        if not len(train) or not len(test):
            raise ValueError("both parity classes must be represented")
        splits.append((train, test))
    return splits


def profile_reconstruction(
    spectra: Array,
    splits: list[tuple[Array, Array]],
    component_counts: list[int],
) -> dict[int, float]:
    """Cross-validated profile energy recovered by mean/PCA spectrum codes.

    Singular frames are treated as already known. The result is therefore a
    conditional core-profile score, not complete-operator reconstruction.
    """

    values = normalize_spectra(spectra)
    counts = sorted(set(component_counts))
    if not counts or counts[0] < 0:
        raise ValueError("component counts must be nonnegative")
    fold_values: dict[int, list[float]] = {count: [] for count in counts}
    for train, test in splits:
        train_values = values[np.asarray(train, dtype=np.int64)]
        test_values = values[np.asarray(test, dtype=np.int64)]
        mean = np.mean(train_values, axis=0, keepdims=True)
        _, _, directions = np.linalg.svd(train_values - mean, full_matrices=False)
        for count in counts:
            available = min(count, len(directions))
            if available:
                basis = directions[:available]
                reconstruction = mean + (test_values - mean) @ basis.T @ basis
            else:
                reconstruction = np.broadcast_to(mean, test_values.shape)
            error = np.sum((test_values - reconstruction) ** 2, axis=1)
            fold_values[count].append(float(1.0 - np.mean(error)))
    return {count: float(np.mean(scores)) for count, scores in fold_values.items()}


def gaussian_factor_spectra(
    count: int,
    width: int,
    rank: int,
    rng: np.random.Generator,
) -> Array:
    """Draw normalized spectra of Gaussian rank-factor products efficiently."""

    if count < 1:
        raise ValueError("count must be positive")
    spectra = []
    for _ in range(count):
        left = rng.standard_normal((width, rank))
        right = rng.standard_normal((width, rank))
        _, left_core = np.linalg.qr(left, mode="reduced")
        _, right_core = np.linalg.qr(right, mode="reduced")
        spectrum = np.linalg.svd(left_core @ right_core.T, compute_uv=False)
        spectra.append(spectrum)
    return normalize_spectra(np.stack(spectra))
