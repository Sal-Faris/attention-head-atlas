import unittest

import numpy as np

from head_atlas.distances import normalized_frobenius_distances
from head_atlas.factors import (
    FactorizedHeadOperator,
    factorized_action,
    factorized_frobenius_norm,
    factorized_inner_product,
    factorized_qk_scores,
    normalized_factorized_frobenius_distances,
)
from head_atlas.operators import HeadOperator


class FactorTests(unittest.TestCase):
    def test_materialization_and_actions(self):
        left = np.asarray([[1.0, 2.0], [0.0, 1.0], [2.0, -1.0]])
        right = np.asarray([[0.5, 1.0], [1.0, 0.0], [-1.0, 2.0]])
        operator = FactorizedHeadOperator(0, 0, "OV", left, right)
        states = np.asarray([[1.0, -2.0, 0.5], [0.0, 1.0, 1.0]])

        expected = states @ (left @ right.T)

        np.testing.assert_allclose(factorized_action(operator, states), expected)
        np.testing.assert_allclose(operator.materialize(), left @ right.T)

    def test_qk_scores_match_materialized_bilinear_form(self):
        rng = np.random.default_rng(4)
        left = rng.standard_normal((5, 2))
        right = rng.standard_normal((5, 2))
        queries = rng.standard_normal((3, 5))
        keys = rng.standard_normal((4, 5))
        operator = FactorizedHeadOperator(0, 0, "QK", left, right)

        expected = queries @ operator.materialize() @ keys.T

        np.testing.assert_allclose(
            factorized_qk_scores(operator, queries, keys), expected, atol=1e-12
        )

    def test_inner_product_and_norm_match_full_matrices(self):
        rng = np.random.default_rng(12)
        first = FactorizedHeadOperator(
            0, 0, "QK", rng.standard_normal((7, 3)), rng.standard_normal((7, 3))
        )
        second = FactorizedHeadOperator(
            0, 1, "QK", rng.standard_normal((7, 3)), rng.standard_normal((7, 3))
        )

        expected_inner = np.sum(first.materialize() * second.materialize())
        expected_norm = np.linalg.norm(first.materialize(), ord="fro")

        self.assertAlmostEqual(factorized_inner_product(first, second), expected_inner)
        self.assertAlmostEqual(factorized_frobenius_norm(first), expected_norm)

    def test_population_distances_match_explicit_operators(self):
        rng = np.random.default_rng(19)
        factorized = [
            FactorizedHeadOperator(
                0,
                head,
                "OV",
                rng.standard_normal((8, 3)),
                rng.standard_normal((8, 3)),
            )
            for head in range(5)
        ]
        explicit = [
            HeadOperator(operator.layer, operator.head, operator.kind, operator.materialize())
            for operator in factorized
        ]

        expected = normalized_frobenius_distances(explicit)
        actual = normalized_factorized_frobenius_distances(factorized)

        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_head_space_gauge_rotation_preserves_operator(self):
        rng = np.random.default_rng(22)
        left = rng.standard_normal((6, 3))
        right = rng.standard_normal((6, 3))
        rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        first = FactorizedHeadOperator(0, 0, "QK", left, right)
        second = FactorizedHeadOperator(0, 1, "QK", left @ rotation, right @ rotation)

        distances = normalized_factorized_frobenius_distances([first, second])

        np.testing.assert_allclose(first.materialize(), second.materialize(), atol=1e-12)
        np.testing.assert_allclose(distances, 0.0, atol=1e-12)

    def test_validation_rejects_malformed_factors(self):
        with self.assertRaisesRegex(ValueError, "same two-dimensional shape"):
            FactorizedHeadOperator(0, 0, "OV", np.ones((3, 2)), np.ones((3, 1)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            FactorizedHeadOperator(
                0, 0, "OV", np.asarray([[np.nan]]), np.asarray([[1.0]])
            )


if __name__ == "__main__":
    unittest.main()
