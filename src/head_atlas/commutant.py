"""Approximate commutants for detecting shared reducing subspaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, lobpcg

Array = np.ndarray


@dataclass(frozen=True)
class CommutantFit:
    """Lowest non-scalar modes of a symmetric operator family's commutant."""

    eigenvalues: Array
    modes: Array
    scalar_eigenvalue: float


def _validated_family(covariances: tuple[Array, ...]) -> tuple[Array, ...]:
    if len(covariances) < 2:
        raise ValueError("at least two covariance operators are required")
    values = tuple(np.asarray(covariance, dtype=np.float64) for covariance in covariances)
    dimension = values[0].shape[0]
    if dimension < 2 or any(matrix.shape != (dimension, dimension) for matrix in values):
        raise ValueError("covariances must share a nontrivial square shape")
    if any(not np.isfinite(matrix).all() for matrix in values):
        raise ValueError("covariances must be finite")
    normalized = []
    for matrix in values:
        symmetric = (matrix + matrix.T) * 0.5
        norm = float(np.linalg.norm(symmetric))
        if norm <= 1e-15:
            raise ValueError("covariances must have positive Frobenius norm")
        normalized.append(symmetric / norm)
    return tuple(normalized)


def _symmetric_indices(dimension: int) -> tuple[Array, Array]:
    return np.triu_indices(dimension)


def pack_symmetric(matrix: Array) -> Array:
    """Pack a symmetric matrix into Frobenius-isometric upper-triangle coordinates."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(values, values.T, atol=1e-10, rtol=0.0):
        raise ValueError("matrix must be symmetric")
    rows, columns = _symmetric_indices(len(values))
    packed = values[rows, columns].copy()
    packed[rows != columns] *= np.sqrt(2.0)
    return packed


def unpack_symmetric(coordinates: Array, dimension: int) -> Array:
    """Invert :func:`pack_symmetric`."""

    values = np.asarray(coordinates, dtype=np.float64)
    expected = dimension * (dimension + 1) // 2
    if values.shape != (expected,):
        raise ValueError("coordinates do not match the requested symmetric dimension")
    rows, columns = _symmetric_indices(dimension)
    upper = values.copy()
    upper[rows != columns] /= np.sqrt(2.0)
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    matrix[rows, columns] = upper
    matrix[columns, rows] = upper
    return matrix


def commutator_energy(mode: Array, covariances: tuple[Array, ...]) -> float:
    """Mean squared commutator norm for a unit-normalized symmetric mode."""

    family = _validated_family(covariances)
    values = np.asarray(mode, dtype=np.float64)
    if values.shape != family[0].shape or not np.allclose(values, values.T, atol=1e-10, rtol=0.0):
        raise ValueError("mode must be a matching symmetric matrix")
    denominator = float(np.sum(values**2))
    if denominator <= 1e-15:
        raise ValueError("mode must have positive norm")
    energies = []
    for covariance in family:
        commutator = covariance @ values - values @ covariance
        energies.append(float(np.sum(commutator**2) / denominator))
    return float(np.mean(energies))


def fit_approximate_commutant(
    covariances: tuple[Array, ...],
    *,
    mode_count: int = 6,
    seed: int = 0,
    tolerance: float = 1e-8,
    maximum_iterations: int = 5_000,
) -> CommutantFit:
    """Find the lowest-energy non-scalar symmetric commutant modes."""

    family = _validated_family(covariances)
    dimension = family[0].shape[0]
    coordinate_count = dimension * (dimension + 1) // 2
    if mode_count < 1 or mode_count >= coordinate_count:
        raise ValueError("invalid non-scalar mode count")

    def action(coordinates: Array) -> Array:
        mode = unpack_symmetric(coordinates, dimension)
        result = np.zeros_like(mode)
        for covariance in family:
            product = covariance @ mode - mode @ covariance
            result += covariance @ product - product @ covariance
        result /= len(family)
        return pack_symmetric((result + result.T) * 0.5)

    operator = LinearOperator(
        (coordinate_count, coordinate_count),
        matvec=action,
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal((coordinate_count, mode_count))
    scalar_coordinates = pack_symmetric(np.eye(dimension)) / np.sqrt(dimension)
    initial -= scalar_coordinates[:, None] * (scalar_coordinates @ initial)[None, :]
    initial, _ = np.linalg.qr(initial, mode="reduced")
    eigenvalues, eigenvectors = lobpcg(
        operator,
        initial,
        Y=scalar_coordinates[:, None],
        largest=False,
        tol=tolerance,
        maxiter=maximum_iterations,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    matrices = np.stack(
        [unpack_symmetric(eigenvectors[:, index], dimension) for index in range(len(order))]
    )
    return CommutantFit(
        eigenvalues=np.maximum(eigenvalues, 0.0),
        modes=matrices,
        scalar_eigenvalue=0.0,
    )


def spectrum_rotated_covariances(
    covariances: tuple[Array, ...],
    rng: np.random.Generator,
) -> tuple[Array, ...]:
    """Independently rotate every covariance while preserving its eigenvalues."""

    family = _validated_family(covariances)
    rotated = []
    for covariance in family:
        eigenvalues = np.linalg.eigvalsh(covariance)
        basis, _ = np.linalg.qr(rng.standard_normal(covariance.shape))
        rotated.append((basis * eigenvalues[None, :]) @ basis.T)
    return tuple(rotated)


def random_traceless_mode(dimension: int, rng: np.random.Generator) -> Array:
    """Sample a unit symmetric traceless comparison mode."""

    values = rng.standard_normal((dimension, dimension))
    values = (values + values.T) * 0.5
    values -= np.trace(values) / dimension * np.eye(dimension)
    return values / np.linalg.norm(values)


def projector_from_mode(mode: Array, *, minimum_rank: int = 2) -> tuple[Array, int, float]:
    """Cut a commutant mode at its largest nontrivial eigengap."""

    values = np.asarray(mode, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != values.shape[1]
        or not np.allclose(values, values.T, atol=1e-10, rtol=0.0)
    ):
        raise ValueError("mode must be square and symmetric")
    dimension = len(values)
    if minimum_rank < 1 or 2 * minimum_rank > dimension:
        raise ValueError("minimum rank leaves no valid two-sided split")
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    gaps = np.diff(eigenvalues)
    valid = np.arange(minimum_rank - 1, dimension - minimum_rank)
    cut = int(valid[np.argmax(gaps[valid])])
    lower_rank = cut + 1
    upper_rank = dimension - lower_rank
    selected = (
        eigenvectors[:, :lower_rank] if lower_rank <= upper_rank else eigenvectors[:, cut + 1 :]
    )
    projector = selected @ selected.T
    spectral_range = max(float(eigenvalues[-1] - eigenvalues[0]), 1e-15)
    return projector, int(min(lower_rank, upper_rank)), float(gaps[cut] / spectral_range)
