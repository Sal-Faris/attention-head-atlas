import numpy as np

from head_atlas.operator_modes import (
    dictionary_variance_recovered,
    principal_operator_modes,
    singular_dimension_summary,
    truncate_operator_modes,
)


def test_unrestricted_modes_recover_known_population_span() -> None:
    rng = np.random.default_rng(12)
    raw = rng.standard_normal((3, 5, 5))
    basis, _ = np.linalg.qr(raw.reshape(3, -1).T)
    basis = basis.T.reshape(3, 5, 5)
    matrices = np.einsum("nk,kij->nij", rng.standard_normal((10, 3)), basis)
    matrices /= np.linalg.norm(matrices, axis=(1, 2), keepdims=True)

    modes = principal_operator_modes(matrices, 3)

    assert np.allclose(
        np.einsum("kij,lij->kl", modes, modes), np.eye(3), atol=1e-9
    )
    assert dictionary_variance_recovered(matrices, modes) > 1 - 1e-10


def test_truncation_and_dimension_summary_find_rank_two() -> None:
    matrix = np.diag([3.0, 2.0, 0.0, 0.0])
    truncated = truncate_operator_modes(matrix[None], 1)
    summary = singular_dimension_summary(matrix)

    assert np.linalg.matrix_rank(truncated[0]) == 1
    assert summary["rank_95_percent_energy"] == 2
