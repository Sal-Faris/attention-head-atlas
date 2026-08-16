"""Map learned PCoA dictionary directions back into residual-stream operators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .factors import FactorizedHeadOperator, factorized_frobenius_norm

Array = np.ndarray


def coordinate_atom_coefficients(coordinates: Array, atoms: Array) -> Array:
    """Return minimum-norm sample coefficients that reconstruct coordinate atoms."""

    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    atom_array = np.asarray(atoms, dtype=np.float64)
    if coordinate_array.ndim != 2 or atom_array.ndim != 2:
        raise ValueError("coordinates and atoms must be two-dimensional")
    if coordinate_array.shape[1] != atom_array.shape[1]:
        raise ValueError("atoms must use the supplied coordinate space")
    if not np.isfinite(coordinate_array).all() or not np.isfinite(atom_array).all():
        raise ValueError("coordinates and atoms must be finite")
    coefficients, _, _, _ = np.linalg.lstsq(coordinate_array.T, atom_array.T, rcond=None)
    reconstructed = coefficients.T @ coordinate_array
    if not np.allclose(reconstructed, atom_array, rtol=1e-7, atol=1e-9):
        raise ValueError("atoms contain directions outside the coordinate span")
    return coefficients


def materialize_operator_atoms(
    coefficients: Array,
    operators: Sequence[FactorizedHeadOperator],
) -> Array:
    """Materialize coordinate atoms as sums of normalized head operators."""

    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    if coefficient_array.ndim != 2 or coefficient_array.shape[0] != len(operators):
        raise ValueError("coefficients must contain one row per operator")
    if not operators:
        raise ValueError("at least one operator is required")
    d_model = operators[0].d_model
    if any(operator.d_model != d_model for operator in operators):
        raise ValueError("operators must share a residual-stream width")
    atoms = np.zeros((coefficient_array.shape[1], d_model, d_model), dtype=np.float64)
    for sample, operator in enumerate(operators):
        normalized = operator.materialize(dtype=np.float64) / factorized_frobenius_norm(
            operator
        )
        atoms += coefficient_array[sample, :, None, None] * normalized[None, :, :]
    return atoms
