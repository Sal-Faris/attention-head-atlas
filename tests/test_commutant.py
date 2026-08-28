import numpy as np

from head_atlas.commutant import (
    commutator_energy,
    fit_approximate_commutant,
    pack_symmetric,
    projector_from_mode,
    spectrum_rotated_covariances,
    unpack_symmetric,
)


def _block_family(rng: np.random.Generator, count: int) -> tuple[np.ndarray, ...]:
    family = []
    for _ in range(count):
        first = rng.standard_normal((3, 3))
        second = rng.standard_normal((3, 3))
        covariance = np.zeros((6, 6))
        covariance[:3, :3] = first @ first.T
        covariance[3:, 3:] = second @ second.T
        family.append(covariance)
    return tuple(family)


def test_symmetric_packing_is_frobenius_isometric() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((5, 5))
    matrix = (matrix + matrix.T) * 0.5
    coordinates = pack_symmetric(matrix)
    np.testing.assert_allclose(unpack_symmetric(coordinates, 5), matrix)
    assert np.isclose(np.sum(coordinates**2), np.sum(matrix**2))


def test_approximate_commutant_detects_planted_reducing_blocks() -> None:
    rng = np.random.default_rng(2)
    family = _block_family(rng, 8)
    fit = fit_approximate_commutant(family, mode_count=2, seed=3)
    null = fit_approximate_commutant(
        spectrum_rotated_covariances(family, rng), mode_count=2, seed=3
    )
    assert fit.eigenvalues[0] < 1e-10
    assert null.eigenvalues[0] > fit.eigenvalues[0] + 1e-3


def test_fitted_mode_generalizes_across_block_family() -> None:
    rng = np.random.default_rng(4)
    training = _block_family(rng, 6)
    held_out = _block_family(rng, 4)
    fit = fit_approximate_commutant(training, mode_count=1, seed=5)
    assert commutator_energy(fit.modes[0], held_out) < 1e-10


def test_mode_projector_recovers_a_nontrivial_spectral_block() -> None:
    mode = np.diag([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    projector, rank, relative_gap = projector_from_mode(mode, minimum_rank=2)
    assert rank == 3
    assert relative_gap == 1.0
    expected = np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert np.allclose(projector, expected) or np.allclose(projector, np.eye(6) - expected)
