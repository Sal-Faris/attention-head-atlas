import numpy as np

from head_atlas.architectural_modules import (
    axis_usage_profiles,
    module_pair_energy,
    partition_axis_profiles,
    top_pair_statistics,
)


def test_axis_profiles_recover_anchor_usage_groups() -> None:
    covariances = (
        np.diag([1.0, 0.8, 0.0, 0.0]),
        np.diag([0.0, 0.0, 0.7, 1.0]),
    )
    profiles = axis_usage_profiles(covariances, np.eye(4))
    labels = partition_axis_profiles(profiles, 2, seed=3)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_module_pair_statistics_detect_concentrated_transformations() -> None:
    coefficients = np.zeros((6, 6))
    coefficients[:3, :3] = 1.0
    coefficients[3:, 3:] = 0.5
    labels = np.repeat([0, 1], 3)
    energy, area = module_pair_energy(coefficients, labels, labels, 2)
    statistics = top_pair_statistics(energy, area, pair_count=2)
    assert statistics["energy_fraction"] == 1.0
    assert statistics["area_fraction"] == 0.5
    assert statistics["energy_enrichment"] == 2.0


def test_module_pair_energy_preserves_total_matrix_energy_and_area() -> None:
    rng = np.random.default_rng(8)
    coefficients = rng.standard_normal((7, 7))
    read = np.asarray([0, 0, 1, 1, 1, 2, 2])
    write = np.asarray([0, 1, 1, 1, 2, 2, 2])
    energy, area = module_pair_energy(coefficients, read, write, 3)
    assert np.isclose(np.sum(energy), np.sum(coefficients**2))
    assert np.sum(area) == coefficients.size
