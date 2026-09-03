import unittest

import numpy as np

from head_atlas.atoms import coordinate_atom_coefficients, materialize_operator_atoms
from head_atlas.factors import FactorizedHeadOperator


class AtomTests(unittest.TestCase):
    def test_coordinate_atom_round_trip_matches_feature_space_direction(self):
        matrices = np.asarray(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        )
        operators = [
            FactorizedHeadOperator(
                0,
                index,
                "OV",
                matrix,
                np.eye(2),
            )
            for index, matrix in enumerate(matrices)
        ]
        normalized = matrices.reshape(4, -1)
        centered = normalized - np.mean(normalized, axis=0, keepdims=True)
        left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
        coordinates = left[:, :3] * singular_values[:3]
        atom = np.asarray([[0.2, -0.4, 0.7]])

        coefficients = coordinate_atom_coefficients(coordinates, atom)
        materialized = materialize_operator_atoms(coefficients, operators)[0]

        np.testing.assert_allclose(coefficients.T @ coordinates, atom, atol=1e-12)
        np.testing.assert_allclose(
            materialized.reshape(-1), (atom @ right[:3]).reshape(-1), atol=1e-12
        )

    def test_coordinate_atom_validation(self):
        with self.assertRaisesRegex(ValueError, "coordinate space"):
            coordinate_atom_coefficients(np.ones((3, 2)), np.ones((1, 3)))
        with self.assertRaisesRegex(ValueError, "one row"):
            materialize_operator_atoms(np.ones((2, 1)), [])


if __name__ == "__main__":
    unittest.main()
