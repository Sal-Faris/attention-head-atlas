"""Diagnostics for structure left after a linear dictionary reconstruction."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def dictionary_residuals(
    coordinates: Array,
    coordinate_mean: Array,
    atoms: Array,
    codes: Array,
) -> tuple[Array, Array]:
    """Return centered observations and their dictionary residuals."""

    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    mean_array = np.asarray(coordinate_mean, dtype=np.float64)
    atom_array = np.asarray(atoms, dtype=np.float64)
    code_array = np.asarray(codes, dtype=np.float64)
    if coordinate_array.ndim != 2:
        raise ValueError("coordinates must be two-dimensional")
    if mean_array.shape not in {(coordinate_array.shape[1],), (1, coordinate_array.shape[1])}:
        raise ValueError("coordinate mean has the wrong shape")
    if atom_array.ndim != 2 or atom_array.shape[1] != coordinate_array.shape[1]:
        raise ValueError("atoms have the wrong coordinate width")
    if code_array.shape != (coordinate_array.shape[0], atom_array.shape[0]):
        raise ValueError("codes have the wrong shape")
    arrays = (coordinate_array, mean_array, atom_array, code_array)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("dictionary arrays must be finite")
    centered = coordinate_array - mean_array.reshape(1, -1)
    residuals = centered - code_array @ atom_array
    return centered, residuals


def reconstruction_energy_summary(centered: Array, residuals: Array) -> dict[str, float]:
    """Summarize the fraction of centered squared norm captured per sample."""

    centered_array = np.asarray(centered, dtype=np.float64)
    residual_array = np.asarray(residuals, dtype=np.float64)
    if centered_array.ndim != 2 or residual_array.shape != centered_array.shape:
        raise ValueError("centered observations and residuals must have matching shapes")
    centered_energy = np.sum(centered_array**2, axis=1)
    residual_energy = np.sum(residual_array**2, axis=1)
    if np.any(centered_energy <= 0):
        raise ValueError("centered observations must be nonzero")
    captured = 1.0 - residual_energy / centered_energy
    return {
        "global_energy_captured": float(
            1.0 - np.sum(residual_energy) / np.sum(centered_energy)
        ),
        "mean_sample_energy_captured": float(np.mean(captured)),
        "median_sample_energy_captured": float(np.median(captured)),
        "minimum_sample_energy_captured": float(np.min(captured)),
        "maximum_sample_energy_captured": float(np.max(captured)),
    }
