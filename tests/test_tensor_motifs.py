import numpy as np

from head_atlas.tensor_motifs import (
    encode_rank_one_motifs,
    fit_rank_one_motifs,
    reconstruct_rank_one_motifs,
)


def test_rank_one_motifs_recover_shared_synthetic_dictionary() -> None:
    rng = np.random.default_rng(4)
    left, _ = np.linalg.qr(rng.standard_normal((8, 3)))
    right, _ = np.linalg.qr(rng.standard_normal((8, 3)))
    coefficients = rng.standard_normal((20, 3))
    tensor = np.einsum("nk,ik,jk->nij", coefficients, left, right)

    motifs = fit_rank_one_motifs(tensor, 3, seed=7, iterations=200, restarts=3)
    recovered = reconstruct_rank_one_motifs(tensor, motifs)
    encoded = encode_rank_one_motifs(tensor, motifs)

    relative_error = np.linalg.norm(tensor - recovered) / np.linalg.norm(tensor)
    assert relative_error < 1e-4
    assert encoded.shape == (20, 3)


def test_rank_one_motifs_validate_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        fit_rank_one_motifs(np.ones((2, 3, 4)), 2)
