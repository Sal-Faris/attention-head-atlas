"""Training-only estimation of shared QK reducing subspaces.

The estimator deliberately makes no assumption about the maps within either
block.  It only searches for fixed input and output projectors whose off-block
energy remains small across a matrix trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ReducingSubspaceFit:
    """A two-block fit, including its projectors in ambient coordinates."""

    output_support_basis: Array
    input_support_basis: Array
    output_core_projector: Array
    input_core_projector: Array
    output_projector: Array
    input_projector: Array
    output_support_projector: Array
    input_support_projector: Array
    objective: float
    iterations: int
    output_rank: int
    input_rank: int

    @property
    def support_dimension(self) -> int:
        return int(self.output_core_projector.shape[0])


def _trajectory(matrices: Array, *, name: str = "matrices") -> Array:
    """Coerce and validate a nonempty finite operator trajectory."""

    try:
        values = np.asarray(matrices, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError(f"{name} must have nonempty shape (checkpoints, output, input)")
    if values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError(f"{name} must have positive input and output dimensions")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _top_eigenspace(matrix: Array, rank: int) -> Array:
    """Return the projector onto the largest-rank eigenspace of a symmetric matrix."""

    eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) * 0.5)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    basis = eigenvectors[:, order[:rank]]
    return basis @ basis.T


def _random_projector(dimension: int, rank: int, rng: np.random.Generator) -> Array:
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, rank)), mode="reduced")
    return basis @ basis.T


def _active_rank(gram: Array) -> int:
    eigenvalues = np.linalg.eigvalsh((gram + gram.T) * 0.5)
    scale = float(np.max(np.abs(eigenvalues)))
    if scale == 0.0:
        return 0
    return int(np.count_nonzero(eigenvalues > 1e-12 * scale))


def active_support_bases(training_matrices: Array, support_dimension: int) -> tuple[Array, Array]:
    """Learn training-only active output and input bases.

    The requested dimension must be supported by nonzero training energy on
    both sides.  This prevents an inactive ambient kernel from becoming an
    arbitrary, apparently meaningful compartment.
    """

    values = _trajectory(training_matrices, name="training_matrices")
    dimension = _positive_int(support_dimension, "support_dimension")
    if dimension > min(values.shape[1], values.shape[2]):
        raise ValueError("support_dimension exceeds an operator dimension")
    output_gram = np.einsum("toi,tpi->op", values, values, optimize=True)
    input_gram = np.einsum("toi,toj->ij", values, values, optimize=True)
    if dimension > _active_rank(output_gram) or dimension > _active_rank(input_gram):
        raise ValueError("support_dimension exceeds the active training rank")
    output_eigenvalues, output_eigenvectors = np.linalg.eigh((output_gram + output_gram.T) * 0.5)
    input_eigenvalues, input_eigenvectors = np.linalg.eigh((input_gram + input_gram.T) * 0.5)
    output_order = np.argsort(output_eigenvalues, kind="stable")[::-1]
    input_order = np.argsort(input_eigenvalues, kind="stable")[::-1]
    return output_eigenvectors[:, output_order[:dimension]], input_eigenvectors[:, input_order[:dimension]]


def _objective(blocks: Array, output_projector: Array, input_projector: Array) -> float:
    d = blocks.shape[1]
    output_complement = np.eye(d) - output_projector
    input_complement = np.eye(d) - input_projector
    first = np.einsum("ij,tjk,kl->til", output_projector, blocks, input_projector, optimize=True)
    second = np.einsum(
        "ij,tjk,kl->til", output_complement, blocks, input_complement, optimize=True
    )
    return float(np.sum(first * first) + np.sum(second * second))


def _alternate(
    blocks: Array,
    output_projector: Array,
    input_projector: Array,
    output_rank: int,
    input_rank: int,
    max_iterations: int,
    tolerance: float,
) -> tuple[Array, Array, float, int]:
    total_output = np.einsum("tij,tkj->ik", blocks, blocks, optimize=True)
    total_input = np.einsum("tji,tjk->ik", blocks, blocks, optimize=True)
    for iteration in range(1, max_iterations + 1):
        output_update = 2.0 * np.einsum(
            "tij,jk,tlk->il", blocks, input_projector, blocks, optimize=True
        ) - total_output
        next_output = _top_eigenspace(output_update, output_rank)
        input_update = 2.0 * np.einsum(
            "tji,jk,tkl->il", blocks, next_output, blocks, optimize=True
        ) - total_input
        next_input = _top_eigenspace(input_update, input_rank)
        change = max(
            float(np.linalg.norm(next_output - output_projector, ord="fro")),
            float(np.linalg.norm(next_input - input_projector, ord="fro")),
        )
        output_projector, input_projector = next_output, next_input
        if change <= tolerance:
            break
    return output_projector, input_projector, _objective(blocks, output_projector, input_projector), iteration


def fit_reducing_subspaces(
    training_matrices: Array,
    support_dimension: int,
    output_rank: int,
    input_rank: int,
    *,
    random_starts: int = 8,
    seed: int = 0,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> ReducingSubspaceFit:
    """Fit fixed reducing projectors from a training trajectory only.

    Energy, mean-operator, and seeded random starts are all optimized with the
    exact alternating eigenspace updates from the frozen protocol.  The fit
    with greatest training block-diagonal energy is retained.
    """

    values = _trajectory(training_matrices, name="training_matrices")
    d = _positive_int(support_dimension, "support_dimension")
    p = _positive_int(output_rank, "output_rank")
    q = _positive_int(input_rank, "input_rank")
    if isinstance(random_starts, (bool, np.bool_)) or not isinstance(
        random_starts, (int, np.integer)
    ):
        raise TypeError("random_starts must be a nonnegative integer")
    if random_starts < 0:
        raise ValueError("random_starts must be a nonnegative integer")
    starts = int(random_starts)
    iterations = _positive_int(max_iterations, "max_iterations")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("tolerance must be a finite nonnegative number") from error
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")
    if p >= d or q >= d:
        raise ValueError("output_rank and input_rank must be smaller than support_dimension")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")

    output_basis, input_basis = active_support_bases(values, d)
    return fit_reducing_subspaces_with_support(
        values,
        output_basis,
        input_basis,
        p,
        q,
        random_starts=starts,
        seed=int(seed),
        max_iterations=iterations,
        tolerance=tolerance,
    )


def fit_reducing_subspaces_with_support(
    training_matrices: Array,
    output_support_basis: Array,
    input_support_basis: Array,
    output_rank: int,
    input_rank: int,
    *,
    random_starts: int = 8,
    seed: int = 0,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> ReducingSubspaceFit:
    """Fit reducing projectors inside supplied training-only active supports.

    This is mathematically identical to :func:`fit_reducing_subspaces`, but it
    lets a multiresolution sweep learn each active support once and reuse it
    across several block-rank pairs.
    """

    values = _trajectory(training_matrices, name="training_matrices")
    output_basis, input_basis, _ = _validate_support_bases(
        output_support_basis,
        input_support_basis,
        output_ambient_dimension=values.shape[1],
        input_ambient_dimension=values.shape[2],
    )
    blocks = np.einsum("oi,tij,jk->tok", output_basis.T, values, input_basis, optimize=True)
    return fit_reducing_subspaces_from_blocks(
        blocks,
        output_basis,
        input_basis,
        output_rank,
        input_rank,
        random_starts=random_starts,
        seed=seed,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def _validate_support_bases(
    output_support_basis: Array,
    input_support_basis: Array,
    *,
    output_ambient_dimension: int | None = None,
    input_ambient_dimension: int | None = None,
) -> tuple[Array, Array, int]:
    """Validate a pair of equally wide ambient orthonormal bases."""

    output_basis = np.asarray(output_support_basis, dtype=np.float64)
    input_basis = np.asarray(input_support_basis, dtype=np.float64)
    if output_basis.ndim != 2 or input_basis.ndim != 2:
        raise ValueError("support bases must be two-dimensional")
    if output_ambient_dimension is not None and output_basis.shape[0] != output_ambient_dimension:
        raise ValueError("support bases have incompatible ambient dimensions")
    if input_ambient_dimension is not None and input_basis.shape[0] != input_ambient_dimension:
        raise ValueError("support bases have incompatible ambient dimensions")
    if output_basis.shape[1] != input_basis.shape[1] or output_basis.shape[1] == 0:
        raise ValueError("support bases must have the same positive width")
    d = int(output_basis.shape[1])
    identity = np.eye(d)
    if not np.all(np.isfinite(output_basis)) or not np.all(np.isfinite(input_basis)):
        raise ValueError("support bases must be finite")
    if not np.allclose(output_basis.T @ output_basis, identity, atol=1e-9, rtol=0):
        raise ValueError("output_support_basis must have orthonormal columns")
    if not np.allclose(input_basis.T @ input_basis, identity, atol=1e-9, rtol=0):
        raise ValueError("input_support_basis must have orthonormal columns")
    return output_basis, input_basis, d


def fit_reducing_subspaces_from_blocks(
    training_blocks: Array,
    output_support_basis: Array,
    input_support_basis: Array,
    output_rank: int,
    input_rank: int,
    *,
    random_starts: int = 8,
    seed: int = 0,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> ReducingSubspaceFit:
    """Fit from operators already projected into supplied active supports.

    ``training_blocks[t]`` must equal ``U.T @ A_t @ V``, where ``U`` and
    ``V`` are the supplied output and input support bases.  This avoids
    rematerializing ambient operators when exact support-coordinate blocks are
    available from a compact factorization.
    """

    blocks = _trajectory(training_blocks, name="training_blocks")
    output_basis, input_basis, d = _validate_support_bases(
        output_support_basis, input_support_basis
    )
    if blocks.shape[1:] != (d, d):
        raise ValueError("training_blocks must be square with the support dimension")
    block_energy = np.einsum("tij,tij->t", blocks, blocks, optimize=True)
    if np.any(block_energy <= 0.0):
        raise ValueError("training_blocks must have positive Frobenius energy")
    p = _positive_int(output_rank, "output_rank")
    q = _positive_int(input_rank, "input_rank")
    if p >= d or q >= d:
        raise ValueError("output_rank and input_rank must be smaller than support dimension")
    if isinstance(random_starts, (bool, np.bool_)) or not isinstance(
        random_starts, (int, np.integer)
    ):
        raise TypeError("random_starts must be a nonnegative integer")
    if random_starts < 0:
        raise ValueError("random_starts must be a nonnegative integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    iterations = _positive_int(max_iterations, "max_iterations")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("tolerance must be a finite nonnegative number") from error
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative number")

    total_output = np.einsum("tij,tkj->ik", blocks, blocks, optimize=True)
    total_input = np.einsum("tji,tjk->ik", blocks, blocks, optimize=True)
    energy_start = (_top_eigenspace(total_output, p), _top_eigenspace(total_input, q))
    left, _, right_transpose = np.linalg.svd(np.mean(blocks, axis=0), full_matrices=False)
    mean_start = (left[:, :p] @ left[:, :p].T, right_transpose[:q].T @ right_transpose[:q])
    rng = np.random.default_rng(int(seed))
    candidates = [energy_start, mean_start]
    candidates.extend(
        (_random_projector(d, p, rng), _random_projector(d, q, rng))
        for _ in range(int(random_starts))
    )

    best: tuple[Array, Array, float, int] | None = None
    for candidate in candidates:
        result = _alternate(blocks, *candidate, p, q, iterations, tolerance)
        if best is None or result[2] > best[2] + 1e-12:
            best = result
    assert best is not None
    output_core, input_core, objective, used_iterations = best
    output_support = output_basis @ output_basis.T
    input_support = input_basis @ input_basis.T
    return ReducingSubspaceFit(
        output_support_basis=output_basis,
        input_support_basis=input_basis,
        output_core_projector=output_core,
        input_core_projector=input_core,
        output_projector=output_basis @ output_core @ output_basis.T,
        input_projector=input_basis @ input_core @ input_basis.T,
        output_support_projector=output_support,
        input_support_projector=input_support,
        objective=objective,
        iterations=used_iterations,
        output_rank=p,
        input_rank=q,
    )


def _check_fit(fit: ReducingSubspaceFit) -> None:
    if not isinstance(fit, ReducingSubspaceFit):
        raise TypeError("fit must be a ReducingSubspaceFit")
    d = fit.support_dimension
    if fit.input_core_projector.shape != (d, d):
        raise ValueError("fit has incompatible core projectors")
    checked = (
        fit.output_support_basis,
        fit.input_support_basis,
        fit.output_core_projector,
        fit.input_core_projector,
    )
    if not all(np.all(np.isfinite(value)) for value in checked):
        raise ValueError("fit must contain finite arrays")


def held_out_block_metrics(
    matrices: Array,
    fit: ReducingSubspaceFit,
    *,
    random_repetitions: int = 128,
    seed: int = 0,
) -> dict[str, Array | float]:
    """Evaluate protocol metrics for validation or confirmation operators."""

    values = _trajectory(matrices)
    _check_fit(fit)
    if values.shape[1] != fit.output_support_basis.shape[0] or values.shape[2] != fit.input_support_basis.shape[0]:
        raise ValueError("matrices have incompatible ambient dimensions for fit")
    blocks = np.einsum(
        "oi,tij,jk->tok",
        fit.output_support_basis.T,
        values,
        fit.input_support_basis,
        optimize=True,
    )
    full_energy = np.einsum("tij,tij->t", values, values, optimize=True)
    return held_out_block_metrics_from_blocks(
        blocks,
        full_energy,
        fit,
        random_repetitions=random_repetitions,
        seed=seed,
    )


def held_out_block_metrics_from_blocks(
    blocks: Array,
    full_operator_energies: Array,
    fit: ReducingSubspaceFit,
    *,
    random_repetitions: int = 128,
    seed: int = 0,
) -> dict[str, Array | float]:
    """Evaluate a fit from held-out support blocks and ambient energies.

    The energy at index ``t`` must be ``||A_t||_F^2`` for the ambient operator
    whose projected block is supplied at the same index.
    """

    projected = _trajectory(blocks, name="blocks")
    _check_fit(fit)
    d = fit.support_dimension
    if projected.shape[1:] != (d, d):
        raise ValueError("blocks must be square with the fit support dimension")
    repetitions = _positive_int(random_repetitions, "random_repetitions")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    try:
        full_energy = np.asarray(full_operator_energies, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("full_operator_energies must be a numeric array") from error
    if full_energy.shape != (len(projected),):
        raise ValueError("full_operator_energies must have one value per held-out block")
    if not np.all(np.isfinite(full_energy)) or np.any(full_energy <= 0.0):
        raise ValueError("full_operator_energies must be finite and positive")
    active_energy = np.einsum("tij,tij->t", projected, projected, optimize=True)
    if np.any(active_energy > full_energy * (1.0 + 1e-9)):
        raise ValueError("projected block energy cannot exceed full operator energy")
    diagonal_energy = np.empty(len(projected))
    cross_energy = np.empty(len(projected))
    p, q = fit.output_core_projector, fit.input_core_projector
    output_complement, input_complement = np.eye(d) - p, np.eye(d) - q
    for index, block in enumerate(projected):
        diagonal_energy[index] = _objective(block[None], p, q)
        first_cross = p @ block @ input_complement
        second_cross = output_complement @ block @ q
        cross_energy[index] = float(np.sum(first_cross * first_cross) + np.sum(second_cross * second_cross))
    coordinate_fraction = (
        fit.output_rank * fit.input_rank + (d - fit.output_rank) * (d - fit.input_rank)
    ) / d**2
    concentration = np.divide(
        diagonal_energy,
        active_energy,
        out=np.zeros_like(active_energy),
        where=active_energy > 0,
    )
    leakage = np.divide(
        cross_energy,
        active_energy,
        out=np.zeros_like(active_energy),
        where=active_energy > 0,
    )

    rng = np.random.default_rng(int(seed))
    random_concentrations = np.empty((repetitions, len(projected)))
    for repetition in range(repetitions):
        random_output = _random_projector(d, fit.output_rank, rng)
        random_input = _random_projector(d, fit.input_rank, rng)
        for index, block in enumerate(projected):
            random_concentrations[repetition, index] = (
                _objective(block[None], random_output, random_input) / active_energy[index]
                if active_energy[index] > 0
                else 0.0
            )
    random_mean = np.mean(random_concentrations, axis=0)
    return {
        "active_support_energy_fraction": active_energy / full_energy,
        "block_diagonal_energy_fraction": diagonal_energy / full_energy,
        "block_diagonal_concentration": concentration,
        "cross_leakage_fraction": leakage,
        "retained_core_coordinate_fraction": float(coordinate_fraction),
        "excess_concentration": concentration - coordinate_fraction,
        "random_block_diagonal_concentration": random_mean,
        "gain_over_random": concentration - random_mean,
    }


def normalized_trace_overlap(first: Array, second: Array) -> float:
    """Return normalized trace overlap of two nonzero ambient projectors."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("projectors must be equal-size square arrays")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("projectors must be finite")
    left_rank, right_rank = float(np.trace(left)), float(np.trace(right))
    if left_rank <= 0 or right_rank <= 0:
        raise ValueError("projectors must have positive rank")
    return float(np.trace(left @ right) / np.sqrt(left_rank * right_rank))


def reducing_pair_overlap(first: ReducingSubspaceFit, second: ReducingSubspaceFit) -> float:
    """Compare two fits, allowing a legal simultaneous exchange of block labels."""

    _check_fit(first)
    _check_fit(second)
    direct = 0.5 * (
        normalized_trace_overlap(first.output_projector, second.output_projector)
        + normalized_trace_overlap(first.input_projector, second.input_projector)
    )
    d_first, d_second = first.support_dimension, second.support_dimension
    complement_is_legal = (
        first.output_rank == d_first - second.output_rank
        and first.input_rank == d_first - second.input_rank
        and d_first == d_second
    )
    if not complement_is_legal:
        return direct
    complementary = 0.5 * (
        normalized_trace_overlap(
            first.output_projector, second.output_support_projector - second.output_projector
        )
        + normalized_trace_overlap(
            first.input_projector, second.input_support_projector - second.input_projector)
    )
    return max(direct, complementary)


def ambient_projectors(fit: ReducingSubspaceFit) -> tuple[Array, Array]:
    """Return the fitted output and input projectors in ambient coordinates."""

    _check_fit(fit)
    return fit.output_projector.copy(), fit.input_projector.copy()
