import unittest

import numpy as np

from head_atlas.representations import (
    effective_rank,
    empirical_action_distance,
    empirical_qk_score_distance,
    frobenius_normalize,
    leading_subspace_projector,
)


class RepresentationTests(unittest.TestCase):
    def test_frobenius_normalization(self):
        normalized = frobenius_normalize(np.diag([3.0, 4.0]))
        self.assertAlmostEqual(np.linalg.norm(normalized, ord="fro"), 1.0)

    def test_effective_rank(self):
        self.assertAlmostEqual(effective_rank(np.eye(4)), 4.0)
        self.assertAlmostEqual(effective_rank(np.diag([2.0, 0.0])), 1.0)

    def test_effective_rank_ignores_float32_roundoff_beyond_true_rank(self):
        rng = np.random.default_rng(17)
        matrix = rng.standard_normal((12, 3), dtype=np.float32) @ rng.standard_normal(
            (3, 12), dtype=np.float32
        )
        self.assertLessEqual(effective_rank(matrix), 3.0 + 1e-6)

    def test_projectors_ignore_basis_rotation_inside_subspace(self):
        matrix = np.diag([4.0, 3.0, 0.0])
        projector = leading_subspace_projector(matrix, rank=2, side="read")
        np.testing.assert_allclose(projector, np.diag([1.0, 1.0, 0.0]), atol=1e-12)

    def test_read_and_write_projectors_follow_row_vector_convention(self):
        matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
        read = leading_subspace_projector(matrix, rank=1, side="read")
        write = leading_subspace_projector(matrix, rank=1, side="write")
        np.testing.assert_allclose(read, np.diag([1.0, 0.0]), atol=1e-12)
        np.testing.assert_allclose(write, np.diag([0.0, 1.0]), atol=1e-12)

    def test_empirical_distances_are_zero_for_identical_operators(self):
        rng = np.random.default_rng(3)
        matrix = rng.standard_normal((4, 4))
        states = rng.standard_normal((10, 4))
        self.assertEqual(empirical_action_distance(matrix, matrix, states), 0.0)
        self.assertEqual(empirical_qk_score_distance(matrix, matrix, states, states), 0.0)


if __name__ == "__main__":
    unittest.main()
