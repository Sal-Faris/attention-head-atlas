import unittest

import numpy as np

from head_atlas.distances import normalized_frobenius_distances
from head_atlas.factors import (
    FactorizedHeadOperator,
    blockwise_factorized_frobenius_distances,
    factorized_action,
    factorized_frobenius_norm,
    factorized_inner_product,
    factorized_qk_scores,
    factorized_singular_components,
    normalized_factorized_frobenius_distances,
    rotary_head_rotation,
    rotate_qk_relative,
)
from head_atlas.operators import HeadOperator


class FactorTests(unittest.TestCase):
    def test_skinny_svd_reconstructs_operator(self):
        rng = np.random.default_rng(23)
        operator = FactorizedHeadOperator(
            0,
            0,
            "OV",
            rng.standard_normal((8, 3)),
            rng.standard_normal((8, 3)),
        )

        left, values, right = factorized_singular_components(operator)

        np.testing.assert_allclose(
            (left * values) @ right.T,
            operator.materialize(dtype=np.float64),
            atol=1e-12,
        )
        np.testing.assert_allclose(left.T @ left, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(right.T @ right, np.eye(3), atol=1e-12)

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

    def test_blockwise_distances_match_skinny_factor_formula(self):
        rng = np.random.default_rng(27)
        operators = [
            FactorizedHeadOperator(
                0,
                head,
                "OV",
                rng.standard_normal((7, 3)).astype(np.float32),
                rng.standard_normal((7, 3)).astype(np.float32),
            )
            for head in range(5)
        ]

        expected = normalized_factorized_frobenius_distances(operators)
        actual = blockwise_factorized_frobenius_distances(operators, block_size=2)

        np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_blockwise_distance_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            blockwise_factorized_frobenius_distances([])
        operator = FactorizedHeadOperator(0, 0, "QK", np.eye(2), np.eye(2))
        with self.assertRaisesRegex(ValueError, "positive"):
            blockwise_factorized_frobenius_distances([operator], block_size=0)

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

    def test_neox_rotary_rotation_is_orthogonal_and_zero_is_identity(self):
        zero = rotary_head_rotation(8, 0.5, 10_000.0, 0)
        shifted = rotary_head_rotation(8, 0.5, 10_000.0, 17)

        np.testing.assert_array_equal(zero, np.eye(8))
        np.testing.assert_allclose(shifted @ shifted.T, np.eye(8), atol=1e-12)

    def test_relative_qk_rotation_matches_explicit_head_space_rotation(self):
        rng = np.random.default_rng(44)
        operator = FactorizedHeadOperator(
            0, 0, "QK", rng.standard_normal((10, 4)), rng.standard_normal((10, 4))
        )
        rotation = rotary_head_rotation(4, 1.0, 10_000.0, 3)

        rotated = rotate_qk_relative(
            operator, 3, rotary_fraction=1.0, base=10_000.0
        )

        np.testing.assert_allclose(
            rotated.materialize(), operator.left @ rotation @ operator.right.T
        )

    def test_validation_rejects_malformed_factors(self):
        with self.assertRaisesRegex(ValueError, "same two-dimensional shape"):
            FactorizedHeadOperator(0, 0, "OV", np.ones((3, 2)), np.ones((3, 1)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            FactorizedHeadOperator(
                0, 0, "OV", np.asarray([[np.nan]]), np.asarray([[1.0]])
            )


if __name__ == "__main__":
    unittest.main()
