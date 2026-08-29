import numpy as np

from head_atlas.relational_invariants import (
    crossfit_subspace_reuse,
    orthonormal_span,
    permute_ambient_coordinates,
    principal_cosine_squares,
)


def test_subspace_statistics_ignore_internal_invertible_coordinates() -> None:
    rng = np.random.default_rng(2)
    writer = rng.standard_normal((18, 5))
    reader = rng.standard_normal((18, 5))
    writer_gauge = rng.standard_normal((5, 5)) + 3 * np.eye(5)
    reader_gauge = rng.standard_normal((5, 5)) + 3 * np.eye(5)
    original = principal_cosine_squares(orthonormal_span(writer), orthonormal_span(reader))
    transformed = principal_cosine_squares(
        orthonormal_span(writer @ writer_gauge),
        orthonormal_span(reader @ reader_gauge),
    )
    assert np.allclose(original, transformed, atol=1e-10)


def test_crossfit_reuse_ignores_independent_partner_coordinate_gauges() -> None:
    """The population statistic must depend on spans, not factor coordinates."""

    rng = np.random.default_rng(19)
    fixed_factors = rng.standard_normal((24, 6))
    partner_factors = tuple(rng.standard_normal((24, 5)) for _ in range(8))
    baseline = crossfit_subspace_reuse(
        orthonormal_span(fixed_factors),
        tuple(orthonormal_span(value) for value in partner_factors),
        ranks=(1, 2, 4),
    )

    fixed_gauge = rng.standard_normal((6, 6)) + 4.0 * np.eye(6)
    partner_gauges = tuple(
        rng.standard_normal((5, 5)) + 4.0 * np.eye(5) for _ in partner_factors
    )
    transformed = crossfit_subspace_reuse(
        orthonormal_span(fixed_factors @ fixed_gauge),
        tuple(
            orthonormal_span(value @ gauge)
            for value, gauge in zip(partner_factors, partner_gauges, strict=True)
        ),
        ranks=(1, 2, 4),
    )

    np.testing.assert_allclose(
        transformed.split_captures, baseline.split_captures, atol=1e-10
    )
    np.testing.assert_allclose(
        transformed.mean_pair_low_rank_energy,
        baseline.mean_pair_low_rank_energy,
        atol=1e-10,
    )
    assert np.isclose(transformed.mean_pair_overlap, baseline.mean_pair_overlap)


def test_crossfit_finds_a_planted_reused_channel() -> None:
    rng = np.random.default_rng(4)
    ambient = 30
    fixed = np.eye(ambient)[:, :6]
    partners = []
    for _ in range(8):
        variable = rng.standard_normal((ambient, 3))
        variable[: fixed.shape[1]] = 0.0
        variable[:, 0] = fixed[:, 0]
        partners.append(orthonormal_span(variable))
    score = crossfit_subspace_reuse(fixed, tuple(partners), ranks=(1, 2, 4))
    assert score.mean_capture[0] > 0.8
    assert score.mean_capture[0] > score.mean_capture[1] / 2


def test_joint_permutation_preserves_partner_geometry() -> None:
    rng = np.random.default_rng(7)
    first = orthonormal_span(rng.standard_normal((20, 4)))
    second = orthonormal_span(rng.standard_normal((20, 4)))
    permutation = rng.permutation(20)
    before = principal_cosine_squares(first, second)
    after = principal_cosine_squares(
        permute_ambient_coordinates(first, permutation),
        permute_ambient_coordinates(second, permutation),
    )
    assert np.allclose(before, after, atol=1e-12)
