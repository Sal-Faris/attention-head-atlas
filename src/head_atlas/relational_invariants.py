"""Gauge-invariant subspace statistics for typed component interfaces.

The routines in this module deliberately work in residual-stream coordinates.
They depend on the row/column spaces of component factors, but not on the
arbitrary coordinates chosen inside an attention head.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SubspaceReuseScore:
    """Reciprocal held-out capture inside one fixed component subspace."""

    ranks: Array
    split_captures: Array
    mean_pair_overlap: float
    mean_pair_low_rank_energy: Array

    @property
    def mean_capture(self) -> Array:
        return np.mean(self.split_captures, axis=0)


def orthonormal_span(vectors: Array, *, tolerance: float = 1e-10) -> Array:
    """Return an orthonormal basis for the columns of ``vectors``."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("vectors must be a nonempty matrix")
    if not np.isfinite(values).all():
        raise ValueError("vectors must be finite")
    left, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    if singular_values[0] <= 0:
        raise ValueError("vectors must have positive rank")
    rank = int(np.sum(singular_values > tolerance * singular_values[0]))
    return left[:, :rank]


def principal_cosine_squares(first: Array, second: Array) -> Array:
    """Squared principal cosines between two orthonormal subspace bases."""

    first_basis = _validated_basis(first, "first")
    second_basis = _validated_basis(second, "second")
    if first_basis.shape[0] != second_basis.shape[0]:
        raise ValueError("subspaces must share an ambient dimension")
    singular_values = np.linalg.svd(first_basis.T @ second_basis, compute_uv=False)
    return np.clip(singular_values**2, 0.0, 1.0)


def crossfit_subspace_reuse(
    fixed_basis: Array,
    partner_bases: tuple[Array, ...],
    *,
    ranks: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> SubspaceReuseScore:
    """Learn shared interface channels on alternating partners and cross-test.

    For a fixed subspace basis ``B`` and partner basis ``R_j``, the restricted
    overlap operator is ``B.T @ R_j @ R_j.T @ B``.  It is normalized to unit
    trace before fitting so the test asks whether *where* partners connect is
    reused, separately from how strongly each pair overlaps.
    """

    fixed = _validated_basis(fixed_basis, "fixed_basis")
    if len(partner_bases) < 4 or len(partner_bases) % 2:
        raise ValueError("an even family of at least four partners is required")
    partners = tuple(_validated_basis(value, "partner") for value in partner_bases)
    if any(value.shape[0] != fixed.shape[0] for value in partners):
        raise ValueError("all subspaces must share an ambient dimension")

    rank_values = np.asarray(ranks, dtype=np.int64)
    if (
        rank_values.ndim != 1
        or len(rank_values) == 0
        or np.any(rank_values < 1)
        or np.any(rank_values >= fixed.shape[1])
        or len(np.unique(rank_values)) != len(rank_values)
    ):
        raise ValueError("ranks must be unique positive values below the fixed rank")

    covariances = []
    overlaps = []
    low_rank = []
    for partner in partners:
        cross_gram = fixed.T @ partner
        covariance = cross_gram @ cross_gram.T
        trace = float(np.trace(covariance))
        if trace <= 1e-15:
            raise ValueError("a partner has numerically zero overlap with the fixed subspace")
        covariances.append(covariance / trace)
        overlaps.append(trace / min(fixed.shape[1], partner.shape[1]))
        squared = np.linalg.svd(cross_gram, compute_uv=False) ** 2
        cumulative = np.cumsum(squared) / np.sum(squared)
        # A requested shared-channel rank may exceed an individual partner's
        # rank even though it remains below the fixed-space rank.  In that
        # case the pairwise low-rank curve has already captured all its energy.
        pair_indices = np.minimum(rank_values, len(cumulative)) - 1
        low_rank.append(cumulative[pair_indices])

    split_captures = np.empty((2, len(rank_values)), dtype=np.float64)
    for offset in (0, 1):
        training = covariances[offset::2]
        held_out = covariances[1 - offset :: 2]
        _, eigenvectors = np.linalg.eigh(np.mean(training, axis=0))
        descending = eigenvectors[:, ::-1]
        for rank_index, rank in enumerate(rank_values):
            basis = descending[:, :rank]
            split_captures[offset, rank_index] = float(
                np.mean([np.trace(basis.T @ covariance @ basis) for covariance in held_out])
            )

    return SubspaceReuseScore(
        ranks=rank_values,
        split_captures=split_captures,
        mean_pair_overlap=float(np.mean(overlaps)),
        mean_pair_low_rank_energy=np.mean(low_rank, axis=0),
    )


def permute_ambient_coordinates(basis: Array, permutation: Array) -> Array:
    """Apply a residual-coordinate permutation to an orthonormal basis."""

    values = _validated_basis(basis, "basis")
    order = np.asarray(permutation)
    if order.shape != (values.shape[0],) or not np.array_equal(
        np.sort(order), np.arange(values.shape[0])
    ):
        raise ValueError("permutation must contain every ambient coordinate once")
    return values[order]


def _validated_basis(basis: Array, name: str) -> Array:
    values = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(values.T @ values, np.eye(values.shape[1]), atol=1e-7):
        raise ValueError(f"{name} must have orthonormal columns")
    return values
