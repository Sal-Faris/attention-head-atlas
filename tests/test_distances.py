import unittest

import numpy as np

from head_atlas.distances import normalized_frobenius_distances
from head_atlas.operators import HeadOperator


def qk_operator(matrix, head=0):
    return HeadOperator(layer=0, head=head, kind="QK", matrix=np.asarray(matrix))


class DistanceTests(unittest.TestCase):
    def test_known_normalized_frobenius_distances(self):
        first = qk_operator([[1.0, 0.0], [0.0, 0.0]], head=0)
        orthogonal = qk_operator([[0.0, 0.0], [0.0, 1.0]], head=1)
        opposite = qk_operator([[-1.0, 0.0], [0.0, 0.0]], head=2)

        distances = normalized_frobenius_distances([first, orthogonal, opposite])

        expected = np.array(
            [
                [0.0, np.sqrt(2.0), 2.0],
                [np.sqrt(2.0), 0.0, np.sqrt(2.0)],
                [2.0, np.sqrt(2.0), 0.0],
            ]
        )
        np.testing.assert_allclose(distances, expected, atol=1e-12)
        np.testing.assert_allclose(distances, distances.T, atol=0.0)
        np.testing.assert_array_equal(np.diag(distances), 0.0)

    def test_positive_scale_does_not_change_direction(self):
        first = qk_operator([[1.0, 2.0], [3.0, 4.0]], head=0)
        scaled = qk_operator([[7.0, 14.0], [21.0, 28.0]], head=1)

        distances = normalized_frobenius_distances([first, scaled])

        np.testing.assert_allclose(distances, 0.0, atol=1e-12)

    def test_shared_orthogonal_basis_change_preserves_distances(self):
        first = qk_operator([[1.0, 2.0], [0.0, 1.0]], head=0)
        second = qk_operator([[0.0, -1.0], [3.0, 2.0]], head=1)
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])

        original = normalized_frobenius_distances([first, second])
        transformed = normalized_frobenius_distances(
            [
                qk_operator(rotation.T @ first.matrix @ rotation, head=0),
                qk_operator(rotation.T @ second.matrix @ rotation, head=1),
            ]
        )

        np.testing.assert_allclose(transformed, original, atol=1e-12)

    def test_rejects_mixed_operator_kinds(self):
        qk = qk_operator(np.eye(2))
        ov = HeadOperator(layer=0, head=1, kind="OV", matrix=np.eye(2))

        with self.assertRaises(ValueError):
            normalized_frobenius_distances([qk, ov])

    def test_rejects_empty_mismatched_or_zero_operators(self):
        with self.assertRaises(ValueError):
            normalized_frobenius_distances([])
        with self.assertRaises(ValueError):
            normalized_frobenius_distances([qk_operator(np.eye(2)), qk_operator(np.eye(3))])
        with self.assertRaises(ValueError):
            normalized_frobenius_distances([qk_operator(np.zeros((2, 2)))])


if __name__ == "__main__":
    unittest.main()
