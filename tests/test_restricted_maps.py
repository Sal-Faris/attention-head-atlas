import unittest
from itertools import pairwise

import numpy as np

from head_atlas.factors import FactorizedHeadOperator
from head_atlas.restricted_maps import (
    architectural_operator_bases,
    fit_restricted_map,
    fit_restricted_map_path,
    population_operator_bases,
    project_operator,
    restricted_block_scalar_cost,
    spectrum_matched_rotation,
)


class RestrictedMapTests(unittest.TestCase):
    @staticmethod
    def _architectural_population() -> tuple[
        list[FactorizedHeadOperator], list[FactorizedHeadOperator]
    ]:
        ov = []
        qk = []
        for layer in range(3):
            for head in range(2):
                read = np.zeros((4, 1))
                write = np.zeros((4, 1))
                query = np.zeros((4, 1))
                key = np.zeros((4, 1))
                read[(layer + head + 1) % 4] = 1.0
                write[head] = 2.0 + layer
                query[head] = 1.0
                key[(head + 2) % 4] = 1.0
                ov.append(FactorizedHeadOperator(layer, head, "OV", read, write))
                qk.append(FactorizedHeadOperator(layer, head, "QK", query, key))
        return ov, qk

    def test_architectural_bases_are_orthonormal_and_target_independent(self):
        ov, qk = self._architectural_population()
        read, write = architectural_operator_bases(
            ov, qk, target_layer=1, anchor_head_parity=0, dimension=2
        )
        changed_ov = list(ov)
        changed_qk = list(qk)
        target = next(
            index for index, operator in enumerate(ov) if operator.layer == 1 and operator.head == 0
        )
        changed_ov[target] = FactorizedHeadOperator(
            1, 0, "OV", np.full((4, 1), 1000.0), np.full((4, 1), -2000.0)
        )
        changed_qk[target] = FactorizedHeadOperator(
            1, 0, "QK", np.full((4, 1), -3000.0), np.full((4, 1), 4000.0)
        )
        changed_read, changed_write = architectural_operator_bases(
            changed_ov,
            changed_qk,
            target_layer=1,
            anchor_head_parity=0,
            dimension=2,
        )
        np.testing.assert_allclose(read.T @ read, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(write.T @ write, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(np.abs(read.T @ changed_read), np.eye(2), atol=1e-10)
        np.testing.assert_allclose(np.abs(write.T @ changed_write), np.eye(2), atol=1e-10)

    def test_architectural_bases_require_both_sides_of_target_layer(self):
        ov, qk = self._architectural_population()
        with self.assertRaisesRegex(ValueError, "both earlier producers and later consumers"):
            architectural_operator_bases(ov, qk, target_layer=0, anchor_head_parity=0, dimension=2)

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
        coefficients[16:32, 16:32] = rng.standard_normal((16, 3)) @ rng.standard_normal((3, 16))
        coefficients /= np.linalg.norm(coefficients)
        fit = fit_restricted_map(
            coefficients,
            complexity_penalty=2e-4,
            support_sizes=(4, 8, 16, 32),
            maximum_blocks=4,
        )
        self.assertGreaterEqual(len(fit.blocks), 2)
        self.assertGreater(fit.captured_energy, 0.9)
        self.assertTrue(
            all(min(len(block.read_indices), len(block.write_indices)) > 1 for block in fit.blocks)
        )

    def test_restricted_fit_path_keeps_every_accepted_prefix(self):
        coefficients = np.zeros((16, 16))
        coefficients[:4, :4] = np.eye(4)
        coefficients[8:12, 8:12] = 0.5 * np.eye(4)
        path = fit_restricted_map_path(
            coefficients,
            complexity_penalty=0.0,
            support_sizes=(4, 8),
            maximum_blocks=3,
        )
        self.assertGreater(len(path), 1)
        self.assertEqual([len(fit.blocks) for fit in path], list(range(len(path))))
        self.assertTrue(
            all(
                earlier.scalar_cost < later.scalar_cost
                and earlier.captured_energy < later.captured_energy
                for earlier, later in pairwise(path)
            )
        )

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
