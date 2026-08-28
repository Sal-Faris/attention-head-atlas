import unittest

import numpy as np

from head_atlas.factors import FactorizedHeadOperator
from head_atlas.restricted_maps import (
    fit_restricted_map,
    population_operator_bases,
    project_operator,
    restricted_block_scalar_cost,
    spectrum_matched_rotation,
)


class RestrictedMapTests(unittest.TestCase):
    def test_population_bases_and_projection_follow_operator_geometry(self):
        left = np.asarray([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        right = np.asarray([[1.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
        operator = FactorizedHeadOperator(0, 0, "OV", left, right)
        read, write = population_operator_bases([operator], 2)
        projected = project_operator(operator, read, write)
        self.assertAlmostEqual(np.linalg.norm(projected), np.linalg.norm(operator.materialize()))
        np.testing.assert_allclose(read.T @ read, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(write.T @ write, np.eye(2), atol=1e-10)

    def test_restricted_fit_recovers_planted_nontrivial_blocks(self):
        rng = np.random.default_rng(7)
        coefficients = np.zeros((32, 32))
        coefficients[:8, :8] = rng.standard_normal((8, 2)) @ rng.standard_normal((2, 8))
        coefficients[16:32, 16:32] = (
            rng.standard_normal((16, 3)) @ rng.standard_normal((3, 16))
        )
        coefficients /= np.linalg.norm(coefficients)
        fit = fit_restricted_map(
            coefficients,
            complexity_penalty=2e-4,
            support_sizes=(4, 8, 16, 32),
            maximum_blocks=4,
        )
        self.assertGreaterEqual(len(fit.blocks), 2)
        self.assertGreater(fit.captured_energy, 0.9)
        self.assertTrue(all(min(len(block.read_indices), len(block.write_indices)) > 1 for block in fit.blocks))

    def test_spectrum_matched_rotation_preserves_singular_values(self):
        rng = np.random.default_rng(3)
        values = rng.standard_normal((12, 12))
        rotated = spectrum_matched_rotation(values, rng)
        np.testing.assert_allclose(
            np.linalg.svd(rotated, compute_uv=False),
            np.linalg.svd(values, compute_uv=False),
            atol=1e-10,
        )

    def test_block_cost_penalizes_larger_supports(self):
        small = restricted_block_scalar_cost(64, 4, 4, 2)
        large = restricted_block_scalar_cost(64, 16, 16, 2)
        self.assertGreater(large, small)


if __name__ == "__main__":
    unittest.main()
