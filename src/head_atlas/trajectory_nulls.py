"""Matched null trajectories for QK reducing-subspace experiments.

The compact representation keeps only numerically resolved singular components;
the full representation keeps the complete thin SVD spectrum, including zeros.
All frames use columns as singular vectors, so an operator is reconstructed as
``left @ diag(singular_values) @ right.T`` at each checkpoint.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def _real_finite_array(value: Array, name: str) -> Array:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float64)


@dataclass(frozen=True)
class SVDTrajectory:
    """Singular frames and spectra for operators at successive checkpoints."""

    left: Array
    singular_values: Array
    right: Array

    def __post_init__(self) -> None:
        left = _real_finite_array(self.left, "left frames")
        singular_values = _real_finite_array(self.singular_values, "singular values")
        right = _real_finite_array(self.right, "right frames")
        if left.ndim != 3 or right.ndim != 3 or singular_values.ndim != 2:
            raise ValueError("frames must be three-dimensional and spectra two-dimensional")
        if left.shape[0] < 1 or left.shape[1] < 1 or right.shape[1] < 1:
            raise ValueError("trajectory frames must have nonempty checkpoint and ambient axes")
        if left.shape[0] != right.shape[0] or left.shape[0] != singular_values.shape[0]:
            raise ValueError("left, spectrum, and right checkpoint counts must match")
        rank = singular_values.shape[1]
        if left.shape[2] != rank or right.shape[2] != rank:
            raise ValueError("frame widths must equal the spectrum width")
        if np.any(singular_values < 0):
            raise ValueError("singular values must be nonnegative")
        if rank:
            identity = np.eye(rank)
            if not np.allclose(left.transpose(0, 2, 1) @ left, identity, atol=1e-10, rtol=0):
                raise ValueError("left frames must have orthonormal columns")
            if not np.allclose(right.transpose(0, 2, 1) @ right, identity, atol=1e-10, rtol=0):
                raise ValueError("right frames must have orthonormal columns")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "right", right)

    @property
    def checkpoint_count(self) -> int:
        return int(self.singular_values.shape[0])

    @property
    def rank(self) -> int:
        return int(self.singular_values.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.left.shape[1]), int(self.right.shape[1])

    def materialize(self) -> Array:
        """Return the represented checkpoint operators."""

        return np.einsum("tir,tr,tjr->tij", self.left, self.singular_values, self.right)


def _validate_matrices(matrices: Array) -> Array:
    matrix_array = _real_finite_array(matrices, "matrices")
    if matrix_array.ndim != 3:
        raise ValueError("matrices must have shape (checkpoints, rows, columns)")
    if any(dimension < 1 for dimension in matrix_array.shape):
        raise ValueError("matrices must have nonempty checkpoint and matrix axes")
    return matrix_array


def full_svd_trajectory(matrices: Array) -> SVDTrajectory:
    """Return the full thin SVD trajectory, retaining zero singular values.

    "Full" refers to retaining all ``min(rows, columns)`` thin-SVD components.
    This avoids the unused null-space columns of a rectangular full-matrix SVD.
    """

    matrix_array = _validate_matrices(matrices)
    left, singular_values, right_transpose = np.linalg.svd(matrix_array, full_matrices=False)
    return SVDTrajectory(left, singular_values, right_transpose.transpose(0, 2, 1))


def compact_svd_trajectory(
    matrices: Array, *, relative_tolerance: float | None = None
) -> SVDTrajectory:
    """Return a lossless-up-to-tolerance trajectory of resolved SVD components.

    A common compact rank is required so that a single frame width applies at
    every checkpoint.  The smooth null consequently has a well-defined overlap
    matrix at every adjacent checkpoint.
    """

    matrix_array = _validate_matrices(matrices)
    if relative_tolerance is not None and (
        not np.isfinite(relative_tolerance) or relative_tolerance < 0
    ):
        raise ValueError("relative_tolerance must be finite and nonnegative")
    full = full_svd_trajectory(matrix_array)
    if relative_tolerance is None:
        relative_tolerance = max(full.shape) * np.finfo(matrix_array.dtype).eps
    maxima = full.singular_values[:, :1]
    ranks = np.count_nonzero(full.singular_values > relative_tolerance * maxima, axis=1)
    if not np.all(ranks == ranks[0]):
        raise ValueError("compact SVD rank must be the same at every checkpoint")
    rank = int(ranks[0])
    return SVDTrajectory(
        full.left[:, :, :rank],
        full.singular_values[:, :rank],
        full.right[:, :, :rank],
    )


def _haar_frame(rows: int, columns: int, rng: np.random.Generator) -> Array:
    if columns == 0:
        return np.empty((rows, 0), dtype=np.float64)
    sample, triangular = np.linalg.qr(rng.standard_normal((rows, columns)), mode="reduced")
    signs = np.sign(np.diag(triangular))
    signs[signs == 0] = 1.0
    return sample * signs


def _complement_frame(previous: Array, rng: np.random.Generator) -> Array:
    """Sample an orthonormal frame in the orthogonal complement of ``previous``."""

    ambient_dimension, rank = previous.shape
    if ambient_dimension - rank < rank:
        raise ValueError("the frame orthogonal complement is too small for the trajectory rank")
    candidate = rng.standard_normal((ambient_dimension, rank))
    candidate -= previous @ (previous.T @ candidate)
    frame, _ = np.linalg.qr(candidate, mode="reduced")
    # Reproject to suppress the tiny in-span residue introduced by QR rounding.
    frame -= previous @ (previous.T @ frame)
    frame, _ = np.linalg.qr(frame, mode="reduced")
    return frame


def _symmetric_psd_square_root(matrix: Array, tolerance: float) -> Array:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("adjacent overlap is incompatible with orthonormal frames")
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T


def _smooth_frame_trajectory(
    frames: Array, rng: np.random.Generator, tolerance: float
) -> Array:
    checkpoints, ambient_dimension, rank = frames.shape
    if rank == 0:
        return np.empty((checkpoints, ambient_dimension, 0), dtype=np.float64)
    if ambient_dimension - rank < rank:
        raise ValueError("the frame orthogonal complement is too small for the trajectory rank")
    result = np.empty_like(frames)
    result[0] = _haar_frame(ambient_dimension, rank, rng)
    for checkpoint in range(1, checkpoints):
        overlap = frames[checkpoint - 1].T @ frames[checkpoint]
        innovation = _symmetric_psd_square_root(
            np.eye(rank) - overlap.T @ overlap, tolerance
        )
        complement = _complement_frame(result[checkpoint - 1], rng)
        result[checkpoint] = result[checkpoint - 1] @ overlap + complement @ innovation
        if not np.allclose(
            result[checkpoint - 1].T @ result[checkpoint], overlap, atol=tolerance, rtol=0
        ):
            raise RuntimeError("smooth frame construction failed to preserve adjacent overlap")
        if not np.allclose(
            result[checkpoint].T @ result[checkpoint], np.eye(rank), atol=tolerance, rtol=0
        ):
            raise RuntimeError("smooth frame construction failed to preserve orthonormality")
    return result


def independent_spectrum_haar_null(
    trajectory: SVDTrajectory, rng: np.random.Generator
) -> SVDTrajectory:
    """Independently Haar-randomize both singular frames at every checkpoint."""

    if not isinstance(trajectory, SVDTrajectory):
        raise TypeError("trajectory must be an SVDTrajectory")
    left = np.stack(
        [_haar_frame(trajectory.shape[0], trajectory.rank, rng) for _ in range(trajectory.checkpoint_count)]
    )
    right = np.stack(
        [_haar_frame(trajectory.shape[1], trajectory.rank, rng) for _ in range(trajectory.checkpoint_count)]
    )
    return SVDTrajectory(left, trajectory.singular_values.copy(), right)


def smooth_singular_frame_trajectory_null(
    trajectory: SVDTrajectory, rng: np.random.Generator, *, tolerance: float = 1e-10
) -> SVDTrajectory:
    """Haar-started frame trajectory preserving every adjacent frame overlap."""

    if not isinstance(trajectory, SVDTrajectory):
        raise TypeError("trajectory must be an SVDTrajectory")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    return SVDTrajectory(
        _smooth_frame_trajectory(trajectory.left, rng, tolerance),
        trajectory.singular_values.copy(),
        _smooth_frame_trajectory(trajectory.right, rng, tolerance),
    )


def within_group_side_trajectory_pairing_null(
    trajectories: Sequence[SVDTrajectory], groups: Sequence[object], rng: np.random.Generator
) -> tuple[tuple[SVDTrajectory, ...], Array]:
    """Pair each trajectory's left side with a different same-group right side.

    The returned index array identifies the right-frame donor for every output
    trajectory.  It is a derangement inside every group and is constant across
    all checkpoints.
    """

    trajectory_tuple = tuple(trajectories)
    if not trajectory_tuple:
        raise ValueError("at least one trajectory is required")
    if len(groups) != len(trajectory_tuple):
        raise ValueError("groups must have one entry per trajectory")
    if not all(isinstance(trajectory, SVDTrajectory) for trajectory in trajectory_tuple):
        raise ValueError("trajectories must contain SVDTrajectory instances")
    reference = trajectory_tuple[0]
    if any(
        trajectory.shape != reference.shape
        or trajectory.rank != reference.rank
        or trajectory.checkpoint_count != reference.checkpoint_count
        for trajectory in trajectory_tuple
    ):
        raise ValueError("all trajectories must have matching shapes, ranks, and checkpoints")

    donor_indices = np.empty(len(trajectory_tuple), dtype=np.int64)
    group_to_indices: dict[object, list[int]] = {}
    for index, group in enumerate(groups):
        try:
            group_to_indices.setdefault(group, []).append(index)
        except TypeError as error:
            raise ValueError("group labels must be hashable") from error
    for indices in group_to_indices.values():
        if len(indices) < 2:
            raise ValueError("every group needs at least two trajectories for a derangement")
        shift = int(rng.integers(1, len(indices)))
        ordered = np.asarray(indices, dtype=np.int64)
        donor_indices[ordered] = np.roll(ordered, -shift)
    paired = tuple(
        SVDTrajectory(
            trajectory.left.copy(),
            trajectory.singular_values.copy(),
            trajectory_tuple[donor_indices[index]].right.copy(),
        )
        for index, trajectory in enumerate(trajectory_tuple)
    )
    return paired, donor_indices


def within_layer_side_trajectory_pairing_null(
    trajectories: Sequence[SVDTrajectory], layers: Sequence[object], rng: np.random.Generator
) -> tuple[tuple[SVDTrajectory, ...], Array]:
    """Layer-labelled alias for :func:`within_group_side_trajectory_pairing_null`."""

    return within_group_side_trajectory_pairing_null(trajectories, layers, rng)


# A concise protocol-facing spelling for the independent spectrum-Haar null.
spectrum_haar_trajectory_null = independent_spectrum_haar_null
