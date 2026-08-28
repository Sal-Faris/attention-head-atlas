"""Direct variable-rank restricted-map models for attention operators."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .factors import FactorizedHeadOperator

Array = np.ndarray


@dataclass(frozen=True)
class RestrictedBlock:
    """One learned transformation between selected read and write supports."""

    read_indices: Array
    write_indices: Array
    core_rank: int
    scalar_cost: float
    energy_gain: float


@dataclass(frozen=True)
class RestrictedMapFit:
    """A sum of restricted blocks and its remaining coefficient residual."""

    residual: Array
    blocks: tuple[RestrictedBlock, ...]
    scalar_cost: float
    captured_energy: float


def population_operator_bases(
    operators: Sequence[FactorizedHeadOperator],
    dimension: int,
) -> tuple[Array, Array]:
    """Learn invariant population read/write bases from operator covariances."""

    if not operators:
        raise ValueError("at least one operator is required")
    width = operators[0].d_model
    if dimension < 1 or dimension > width:
        raise ValueError("basis dimension must lie within the residual width")
    if any(operator.d_model != width for operator in operators):
        raise ValueError("operators must share their residual width")
    read_covariance = np.zeros((width, width), dtype=np.float64)
    write_covariance = np.zeros((width, width), dtype=np.float64)
    for operator in operators:
        left = np.asarray(operator.left, dtype=np.float64)
        right = np.asarray(operator.right, dtype=np.float64)
        read_covariance += left @ (right.T @ right) @ left.T
        write_covariance += right @ (left.T @ left) @ right.T
    _, read_vectors = np.linalg.eigh(read_covariance)
    _, write_vectors = np.linalg.eigh(write_covariance)
    return read_vectors[:, -dimension:][:, ::-1], write_vectors[:, -dimension:][:, ::-1]


def project_operator(
    operator: FactorizedHeadOperator,
    read_basis: Array,
    write_basis: Array,
) -> Array:
    """Project an exact skinny operator into learned read/write coordinates."""

    read = np.asarray(read_basis, dtype=np.float64)
    write = np.asarray(write_basis, dtype=np.float64)
    if read.ndim != 2 or write.ndim != 2 or read.shape != write.shape:
        raise ValueError("read and write bases must have the same two-dimensional shape")
    if read.shape[0] != operator.d_model:
        raise ValueError("basis width does not match the operator")
    left_coordinates = read.T @ np.asarray(operator.left, dtype=np.float64)
    write_coordinates = write.T @ np.asarray(operator.right, dtype=np.float64)
    return left_coordinates @ write_coordinates.T


def shared_basis_scalar_cost(width: int, dimension: int, encoded_heads: int) -> float:
    """Amortized degrees of freedom for two orthonormal population bases."""

    if width < 1 or dimension < 1 or dimension > width or encoded_heads < 1:
        raise ValueError("invalid shared-basis dimensions")
    stiefel_degrees = width * dimension - dimension * (dimension + 1) / 2
    return float(2 * stiefel_degrees / encoded_heads)


def _log2_choose(total: int, selected: int) -> float:
    if selected < 0 or selected > total:
        return float("inf")
    if selected in {0, total}:
        return 0.0
    return float(
        (lgamma(total + 1) - lgamma(selected + 1) - lgamma(total - selected + 1))
        / log(2.0)
    )


def restricted_block_scalar_cost(
    dimension: int,
    read_size: int,
    write_size: int,
    core_rank: int,
    *,
    float_bits: int = 16,
) -> float:
    """Parameter-equivalent cost of supports plus a low-rank arbitrary core."""

    if (
        dimension < 1
        or read_size < 1
        or write_size < 1
        or read_size > dimension
        or write_size > dimension
        or core_rank < 1
        or core_rank > min(read_size, write_size)
        or float_bits < 1
    ):
        raise ValueError("invalid restricted block dimensions")
    core_scalars = core_rank * (read_size + write_size - core_rank) + core_rank
    support_bits = _log2_choose(dimension, read_size) + _log2_choose(
        dimension, write_size
    )
    return float(core_scalars + support_bits / float_bits + 1.0)


def _top_indices(values: Array, count: int) -> tuple[int, ...]:
    selected = np.argpartition(np.asarray(values), -count)[-count:]
    return tuple(sorted(int(index) for index in selected))


def candidate_rectangles(
    residual: Array,
    *,
    support_sizes: Sequence[int] = (4, 8, 16, 32, 64),
    spectral_seeds: int = 4,
) -> tuple[tuple[Array, Array], ...]:
    """Generate deterministic energy biclusters without fixing operator types."""

    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("residual must be a square coefficient matrix")
    dimension = len(values)
    sizes = tuple(sorted({size for size in support_sizes if 1 < size <= dimension}))
    if not sizes:
        raise ValueError("at least one valid nontrivial support size is required")
    energy = values**2
    left, _, right_transpose = np.linalg.svd(energy, full_matrices=False)
    row_scores = [np.sum(energy, axis=1)]
    column_scores = [np.sum(energy, axis=0)]
    for index in range(min(spectral_seeds, dimension)):
        row_scores.append(np.abs(left[:, index]))
        column_scores.append(np.abs(right_transpose[index]))

    rectangles: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for rows_score, columns_score in zip(row_scores, column_scores, strict=True):
        for read_size in sizes:
            rows = _top_indices(rows_score, read_size)
            for write_size in sizes:
                columns = _top_indices(columns_score, write_size)
                rectangles.add((rows, columns))
    return tuple(
        (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64))
        for rows, columns in sorted(rectangles)
    )


def fit_restricted_map(
    coefficients: Array,
    *,
    complexity_penalty: float,
    support_sizes: Sequence[int] = (4, 8, 16, 32, 64),
    core_ranks: Sequence[int] = (1, 2, 4, 8),
    maximum_blocks: int = 6,
) -> RestrictedMapFit:
    """Greedily minimize distortion plus complexity over restricted maps."""

    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("coefficients must be square")
    if not np.isfinite(values).all() or complexity_penalty < 0 or maximum_blocks < 1:
        raise ValueError("invalid coefficients, penalty, or block count")
    residual = values.copy()
    blocks = []
    scalar_cost = 0.0
    captured_energy = 0.0
    dimension = len(values)
    for _ in range(maximum_blocks):
        best: tuple[float, float, float, Array, Array, int, Array] | None = None
        for rows, columns in candidate_rectangles(
            residual, support_sizes=support_sizes
        ):
            submatrix = residual[np.ix_(rows, columns)]
            if len(rows) >= len(columns):
                eigenvalues, vectors = np.linalg.eigh(submatrix.T @ submatrix)
                order = np.argsort(eigenvalues)[::-1]
                eigenvalues = np.maximum(eigenvalues[order], 0.0)
                vectors = vectors[:, order]
                side = "write"
            else:
                eigenvalues, vectors = np.linalg.eigh(submatrix @ submatrix.T)
                order = np.argsort(eigenvalues)[::-1]
                eigenvalues = np.maximum(eigenvalues[order], 0.0)
                vectors = vectors[:, order]
                side = "read"
            for rank in core_ranks:
                if rank > min(len(rows), len(columns)):
                    continue
                gain = float(np.sum(eigenvalues[:rank]))
                cost = restricted_block_scalar_cost(
                    dimension, len(rows), len(columns), rank
                )
                score = gain - complexity_penalty * cost
                if best is None or score > best[0]:
                    selected_vectors = vectors[:, :rank]
                    if side == "write":
                        reconstruction = (submatrix @ selected_vectors) @ selected_vectors.T
                    else:
                        reconstruction = selected_vectors @ (selected_vectors.T @ submatrix)
                    best = score, gain, cost, rows, columns, rank, reconstruction
        if best is None or best[0] <= 0:
            break
        _, gain, cost, rows, columns, rank, reconstruction = best
        residual[np.ix_(rows, columns)] -= reconstruction
        blocks.append(
            RestrictedBlock(
                read_indices=rows,
                write_indices=columns,
                core_rank=rank,
                scalar_cost=cost,
                energy_gain=gain,
            )
        )
        scalar_cost += cost
        captured_energy += gain
    return RestrictedMapFit(
        residual=residual,
        blocks=tuple(blocks),
        scalar_cost=float(scalar_cost),
        captured_energy=float(captured_energy),
    )


def spectrum_matched_rotation(coefficients: Array, rng: np.random.Generator) -> Array:
    """Destroy support geometry while preserving every projected singular value."""

    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("coefficients must be square")
    singular_values = np.linalg.svd(values, compute_uv=False)
    left, _ = np.linalg.qr(rng.standard_normal(values.shape))
    right, _ = np.linalg.qr(rng.standard_normal(values.shape))
    return (left * singular_values[None, :]) @ right.T
