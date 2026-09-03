import unittest

import numpy as np

from head_atlas.nulls import (
    haar_orthonormal_frame,
    rank_norm_matched_gaussian,
    sample_norm_matched_isotropic,
    spectrum_matched_rotation,
)


class NullTests(unittest.TestCase):
    def test_isotropic_vector_null_preserves_each_norm(self):
        vectors = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
        null = sample_norm_matched_isotropic(vectors, np.random.default_rng(7))

        np.testing.assert_allclose(
            np.linalg.norm(null, axis=1), np.linalg.norm(vectors, axis=1), atol=1e-12
        )

    def test_haar_frame_is_orthonormal(self):
        frame = haar_orthonormal_frame(7, 3, np.random.default_rng(9))
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)

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

    def test_rank_norm_null_respects_float32_low_rank_structure(self):
        rng = np.random.default_rng(6)
        matrix = rng.standard_normal((12, 3), dtype=np.float32) @ rng.standard_normal(
            (3, 12), dtype=np.float32
        )
        null = rank_norm_matched_gaussian(matrix, rng)
        self.assertEqual(np.linalg.matrix_rank(null), 3)


if __name__ == "__main__":
    unittest.main()
