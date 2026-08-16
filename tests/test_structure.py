import unittest

import numpy as np

from head_atlas.structure import (
    categorical_permanova,
    design_permanova,
    pcoa_spectrum_summary,
    residualize_euclidean_distances,
)


class StructureTests(unittest.TestCase):
    def test_pcoa_summary_recovers_one_dimensional_population(self):
        points = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        distances = np.abs(points - points.T)

        summary = pcoa_spectrum_summary(distances)

        self.assertEqual(summary["positive_dimensions"], 1)
        self.assertAlmostEqual(summary["participation_dimension"], 1.0)
        self.assertAlmostEqual(summary["top_1_variance"], 1.0)
        self.assertEqual(summary["dimensions_for_90_percent"], 1)
        self.assertAlmostEqual(summary["negative_eigenvalue_mass_ratio"], 0.0)

    def test_permanova_detects_separated_group_centroids(self):
        points = np.asarray([[0.0], [0.1], [10.0], [10.1]])
        distances = np.abs(points - points.T)

        result = categorical_permanova(
            distances,
            labels=["near_zero", "near_zero", "near_ten", "near_ten"],
            permutations=19,
            seed=3,
        )

        self.assertGreater(result["explained_variance_fraction"], 0.99)
        self.assertGreater(result["pseudo_f"], 1000.0)
        self.assertGreaterEqual(result["p_value"], 1 / 20)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_permanova_rejects_wrong_label_count(self):
        with self.assertRaisesRegex(ValueError, "one value per item"):
            categorical_permanova(np.zeros((3, 3)), labels=[0, 1])

    def test_design_permanova_recovers_continuous_axis(self):
        points = np.arange(8, dtype=np.float64)[:, None]
        distances = np.abs(points - points.T)

        result = design_permanova(
            distances,
            predictors=points,
            permutations=19,
            seed=2,
        )

        self.assertAlmostEqual(result["explained_variance_fraction"], 1.0)
        self.assertEqual(result["predictor_rank"], 1)

    def test_residualization_removes_predictor_axis(self):
        predictor = np.asarray([-3.0, -1.0, 1.0, 3.0])[:, None]
        nuisance = np.asarray([1.0, -1.0, -1.0, 1.0])[:, None]
        points = np.hstack([predictor, nuisance])
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

        residual_distances = residualize_euclidean_distances(distances, predictor)
        expected = np.abs(nuisance - nuisance.T)

        np.testing.assert_allclose(residual_distances, expected, atol=1e-7)

    def test_residualization_rejects_constant_predictor(self):
        distances = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
        with self.assertRaisesRegex(ValueError, "no nonconstant variation"):
            residualize_euclidean_distances(distances, np.ones(4))


if __name__ == "__main__":
    unittest.main()
