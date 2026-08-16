import unittest

import numpy as np

from head_atlas.motifs import matched_atom_similarity


class MotifTests(unittest.TestCase):
    def test_matching_ignores_atom_order_and_sign(self):
        first = np.eye(3)
        second = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])

        self.assertAlmostEqual(matched_atom_similarity(first, second), 1.0)

    def test_matching_finds_orthogonal_difference(self):
        first = np.asarray([[1.0, 0.0, 0.0]])
        second = np.asarray([[0.0, 1.0, 0.0]])

        self.assertAlmostEqual(matched_atom_similarity(first, second), 0.0)

    def test_matching_validates_shapes(self):
        with self.assertRaisesRegex(ValueError, "same nonempty shape"):
            matched_atom_similarity(np.eye(2), np.eye(3))


if __name__ == "__main__":
    unittest.main()
