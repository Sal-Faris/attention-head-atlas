import unittest

import numpy as np

from head_atlas.intrinsic_mdl import (
    normalize_spectra,
    parity_splits,
    profile_reconstruction,
    rank_description,
    rank_energy_curve,
)


class IntrinsicMdlTests(unittest.TestCase):
    def test_rank_description_matches_rank_manifold(self):
        description = rank_description(512, 16)
        self.assertEqual(description.unrestricted, 16128)
        self.assertEqual(description.maximum_reusable_core_saving, 15)
        self.assertEqual(description.fixed_normalized_spectrum, 16113)

    def test_normalized_energy_curve_reaches_one(self):
        spectra = normalize_spectra(np.asarray([[3.0, 1.0], [1.0, 1.0]]))
        curve = rank_energy_curve(spectra, [1, 2])
        self.assertLess(curve[1], 1.0)
        self.assertAlmostEqual(curve[2], 1.0)

    def test_shared_profile_is_recovered_on_unseen_observations(self):
        base = np.asarray([4.0, 2.0, 1.0, 0.5])
        spectra = np.stack([base * (1.0 + 0.001 * index) for index in range(8)])
        report = profile_reconstruction(spectra, parity_splits(np.arange(8)), [0, 1])
        self.assertGreater(report[0], 0.999999)
        self.assertGreater(report[1], 0.999999)

    def test_profile_score_is_deterministic(self):
        rng = np.random.default_rng(3)
        spectra = normalize_spectra(rng.uniform(0.1, 1.0, size=(8, 4)))
        report = profile_reconstruction(spectra, parity_splits(np.arange(8)), [0, 2])
        repeated = profile_reconstruction(spectra.copy(), parity_splits(np.arange(8)), [0, 2])
        self.assertEqual(report, repeated)


if __name__ == "__main__":
    unittest.main()
