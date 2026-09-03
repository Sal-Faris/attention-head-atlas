import unittest

import numpy as np

from head_atlas.residuals import dictionary_residuals, reconstruction_energy_summary


class ResidualTests(unittest.TestCase):
    def test_dictionary_residuals_apply_mean_and_sparse_reconstruction(self):
        coordinates = np.asarray([[2.0, 1.0], [1.0, 3.0]])
        mean = np.asarray([[1.0, 1.0]])
        atoms = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        codes = np.asarray([[0.5, 0.0], [0.0, 1.5]])

        centered, residuals = dictionary_residuals(coordinates, mean, atoms, codes)

        np.testing.assert_allclose(centered, [[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_allclose(residuals, [[0.5, 0.0], [0.0, 0.5]])

    def test_reconstruction_energy_summary(self):
        summary = reconstruction_energy_summary(
            np.asarray([[1.0, 0.0], [0.0, 2.0]]),
            np.asarray([[0.5, 0.0], [0.0, 1.0]]),
        )

        self.assertAlmostEqual(summary["global_energy_captured"], 0.75)
        self.assertAlmostEqual(summary["median_sample_energy_captured"], 0.75)


if __name__ == "__main__":
    unittest.main()
