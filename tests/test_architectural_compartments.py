import unittest

import numpy as np

from head_atlas.architectural_compartments import (
    confirmation_r2,
    factor_overlap,
    fit_compartments,
    weighted_subspace_overlap,
)


class ArchitecturalCompartmentTests(unittest.TestCase):
    def test_two_latent_groups_are_recovered_and_confirmed(self):
        rng = np.random.default_rng(4)
        labels = np.repeat([0, 1], 24)
        discovery = labels[:, None] * 4.0 + rng.normal(0, 0.15, size=(48, 4))
        confirmation = labels[:, None] * -3.0 + rng.normal(0, 0.15, size=(48, 3))
        fit = fit_compartments(discovery, confirmation, maximum_components=4, seed=2)
        self.assertEqual(fit.component_count, 2)
        self.assertGreater(fit.confirmation_r2, 0.9)

    def test_confirmation_r2_is_zero_for_one_group(self):
        values = np.arange(24, dtype=float).reshape(8, 3)
        self.assertEqual(confirmation_r2(values, np.zeros(8, dtype=int)), 0.0)

    def test_overlap_helpers_recover_coordinate_alignment(self):
        modes = np.eye(3)
        weighted = weighted_subspace_overlap(modes, np.eye(3), np.asarray([2.0, 1.0, 0.0]))
        np.testing.assert_allclose(weighted, [0.8, 0.2, 0.0])
        factor = np.asarray([[2.0], [0.0], [0.0]])
        np.testing.assert_allclose(factor_overlap(modes, factor), [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
