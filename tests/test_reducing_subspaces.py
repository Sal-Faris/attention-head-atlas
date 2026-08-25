import numpy as np
import pytest

from head_atlas.reducing_subspaces import (
    active_support_bases,
    ambient_projectors,
    fit_reducing_subspaces,
    fit_reducing_subspaces_with_support,
    held_out_block_metrics,
    normalized_trace_overlap,
    reducing_pair_overlap,
)


def _orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((dimension, dimension)))[0]


def _fixed_block_trajectory(
    rng: np.random.Generator,
    count: int,
    *,
    ambient: int = 12,
    support: int = 6,
    output_rank: int = 3,
    input_rank: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense, independently changing cores in fixed input/output blocks."""

    output_frame = _orthogonal(rng, ambient)
    input_frame = _orthogonal(rng, ambient)
    matrices = []
    for _ in range(count):
        core = np.zeros((support, support))
        core[:output_rank, :input_rank] = rng.standard_normal((output_rank, input_rank))
        core[output_rank:, input_rank:] = rng.standard_normal(
            (support - output_rank, support - input_rank)
        )
        matrices.append(output_frame[:, :support] @ core @ input_frame[:, :support].T)
    return np.stack(matrices), output_frame, input_frame


def test_recovers_fixed_dense_blocks_and_held_out_concentration() -> None:
    rng = np.random.default_rng(101)
    training, output_frame, input_frame = _fixed_block_trajectory(rng, 24)
    held_cores = []
    for _ in range(8):
        core = np.zeros((6, 6))
        core[:3, :2] = rng.standard_normal((3, 2))
        core[3:, 2:] = rng.standard_normal((3, 4))
        held_cores.append(output_frame[:, :6] @ core @ input_frame[:, :6].T)
    held_out = np.stack(held_cores)

    fit = fit_reducing_subspaces(training, 6, 3, 2, random_starts=6, seed=7)
    metrics = held_out_block_metrics(held_out, fit, random_repetitions=48, seed=8)
    expected_output = output_frame[:, :3] @ output_frame[:, :3].T
    expected_input = input_frame[:, :2] @ input_frame[:, :2].T

    assert normalized_trace_overlap(fit.output_projector, expected_output) > 1 - 1e-8
    assert normalized_trace_overlap(fit.input_projector, expected_input) > 1 - 1e-8
    assert np.all(metrics["active_support_energy_fraction"] > 1 - 1e-10)
    assert np.all(metrics["block_diagonal_concentration"] > 1 - 1e-9)
    assert np.all(metrics["cross_leakage_fraction"] < 1e-9)
    assert np.all(metrics["gain_over_random"] > 0.2)


def test_dense_random_family_has_no_held_out_block_advantage() -> None:
    rng = np.random.default_rng(102)
    training = rng.standard_normal((40, 8, 8))
    validation = rng.standard_normal((80, 8, 8))
    fit = fit_reducing_subspaces(training, 8, 4, 3, random_starts=6, seed=3)
    metrics = held_out_block_metrics(validation, fit, random_repetitions=96, seed=4)

    assert abs(float(np.mean(metrics["gain_over_random"]))) < 0.08
    assert abs(float(np.mean(metrics["excess_concentration"]))) < 0.1


def test_independent_input_output_gauge_rotates_ambient_projectors() -> None:
    rng = np.random.default_rng(103)
    matrices, _, _ = _fixed_block_trajectory(rng, 25)
    output_gauge = _orthogonal(rng, 12)
    input_gauge = _orthogonal(rng, 12)
    transformed = np.einsum("ij,tjk,lk->til", output_gauge, matrices, input_gauge, optimize=True)

    fit = fit_reducing_subspaces(matrices, 6, 3, 2, random_starts=5, seed=5)
    transformed_fit = fit_reducing_subspaces(transformed, 6, 3, 2, random_starts=5, seed=5)
    output_projector, input_projector = ambient_projectors(fit)
    transformed_output, transformed_input = ambient_projectors(transformed_fit)

    assert np.isclose(fit.objective, transformed_fit.objective, atol=1e-8)
    assert normalized_trace_overlap(
        transformed_output, output_gauge @ output_projector @ output_gauge.T
    ) > 1 - 1e-8
    assert normalized_trace_overlap(
        transformed_input, input_gauge @ input_projector @ input_gauge.T
    ) > 1 - 1e-8


def test_active_support_excludes_inactive_ambient_kernel() -> None:
    rng = np.random.default_rng(104)
    matrices, output_frame, input_frame = _fixed_block_trajectory(rng, 20, ambient=15, support=5, output_rank=2, input_rank=2)
    output_basis, input_basis = active_support_bases(matrices, 5)
    fit = fit_reducing_subspaces(matrices, 5, 2, 2, random_starts=4, seed=1)

    assert normalized_trace_overlap(output_basis @ output_basis.T, output_frame[:, :5] @ output_frame[:, :5].T) > 1 - 1e-10
    assert normalized_trace_overlap(input_basis @ input_basis.T, input_frame[:, :5] @ input_frame[:, :5].T) > 1 - 1e-10
    assert np.allclose(fit.output_support_projector @ output_frame[:, 5:], 0.0, atol=1e-9)
    assert np.allclose(fit.input_support_projector @ input_frame[:, 5:], 0.0, atol=1e-9)


def test_fixed_seed_is_deterministic_and_complement_alignment_is_allowed() -> None:
    rng = np.random.default_rng(105)
    matrices, _, _ = _fixed_block_trajectory(rng, 20, support=6, output_rank=3, input_rank=3)
    first = fit_reducing_subspaces(matrices, 6, 3, 3, random_starts=7, seed=11)
    second = fit_reducing_subspaces(matrices, 6, 3, 3, random_starts=7, seed=11)

    assert np.array_equal(first.output_core_projector, second.output_core_projector)
    assert np.array_equal(first.input_core_projector, second.input_core_projector)
    assert first.objective == second.objective
    assert reducing_pair_overlap(first, second) > 1 - 1e-12


def test_cached_active_support_gives_the_same_fit() -> None:
    rng = np.random.default_rng(106)
    matrices, _, _ = _fixed_block_trajectory(rng, 20)
    output_basis, input_basis = active_support_bases(matrices, 6)

    direct = fit_reducing_subspaces(matrices, 6, 3, 2, random_starts=3, seed=9)
    cached = fit_reducing_subspaces_with_support(
        matrices,
        output_basis,
        input_basis,
        3,
        2,
        random_starts=3,
        seed=9,
    )

    assert np.allclose(direct.output_projector, cached.output_projector)
    assert np.allclose(direct.input_projector, cached.input_projector)
    assert direct.objective == cached.objective


@pytest.mark.parametrize(
    ("matrices", "support", "output_rank", "input_rank"),
    [
        (np.empty((0, 4, 4)), 2, 1, 1),
        (np.ones((3, 4, 3)), 4, 1, 1),
        (np.full((3, 4, 4), np.nan), 2, 1, 1),
        (np.zeros((3, 4, 4)), 2, 1, 1),
        (np.ones((3, 4, 4)), 4, 4, 1),
        (np.ones((3, 4, 4)), 4, 1, 0),
    ],
)
def test_validation_errors(matrices: np.ndarray, support: int, output_rank: int, input_rank: int) -> None:
    with pytest.raises(ValueError):
        fit_reducing_subspaces(matrices, support, output_rank, input_rank)
