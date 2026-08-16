import unittest

import numpy as np

from head_atlas.nulls import rank_norm_matched_gaussian, spectrum_matched_rotation


class NullTests(unittest.TestCase):
    def test_spectrum_matched_rotation_preserves_singular_values(self):
        rng = np.random.default_rng(4)
        matrix = rng.standard_normal((6, 6))
        null = spectrum_matched_rotation(matrix, rng)
        np.testing.assert_allclose(
            np.linalg.svd(null, compute_uv=False),
            np.linalg.svd(matrix, compute_uv=False),
            atol=1e-10,
        )

    def test_rank_norm_null_preserves_rank_and_norm(self):
        rng = np.random.default_rng(5)
        matrix = rng.standard_normal((7, 2)) @ rng.standard_normal((2, 7))
        null = rank_norm_matched_gaussian(matrix, rng)
        self.assertEqual(np.linalg.matrix_rank(null), np.linalg.matrix_rank(matrix))
        self.assertAlmostEqual(np.linalg.norm(null, ord="fro"), np.linalg.norm(matrix, ord="fro"))


if __name__ == "__main__":
    unittest.main()
