import unittest

import numpy as np

from head_atlas.dictionary import (
    blocked_checkpoint_splits,
    cross_validated_reconstruction,
    grouped_splits,
    head_trajectory_groups,
    joint_view_coordinates,
    residualize_group_means,
)


class DictionaryTests(unittest.TestCase):
    def test_joint_coordinates_give_equal_weight_product_distance(self):
        qk = np.asarray([[0.0, 0.0], [3.0, 0.0]])
        ov = np.asarray([[0.0], [4.0]])
        joint = joint_view_coordinates(qk, ov)

        self.assertAlmostEqual(np.linalg.norm(joint[0] - joint[1]), 5.0 / np.sqrt(2.0))

    def test_grouped_splits_never_leak_head_trajectories(self):
        layers = np.tile([0, 0, 1, 1], 3)
        heads = np.tile([0, 1, 0, 1], 3)
        groups = head_trajectory_groups(layers, heads)

        for train, test in grouped_splits(groups, folds=2):
            self.assertTrue(set(groups[train]).isdisjoint(groups[test]))

    def test_checkpoint_splits_hold_out_contiguous_values(self):
        values = np.repeat([0, 10, 20, 30], 3)
        splits = blocked_checkpoint_splits(values, blocks=2)

        self.assertEqual(set(values[splits[0][1]]), {0, 10})
        self.assertEqual(set(values[splits[1][1]]), {20, 30})

    def test_group_residuals_have_zero_group_means(self):
        coordinates = np.asarray([[1.0, 2.0], [3.0, 4.0], [8.0, 1.0], [6.0, 5.0]])
        groups = np.asarray([0, 0, 1, 1])

        residuals = residualize_group_means(coordinates, groups)

        np.testing.assert_allclose(np.mean(residuals[:2], axis=0), 0.0)
        np.testing.assert_allclose(np.mean(residuals[2:], axis=0), 0.0)

    def test_cross_validated_models_return_each_baseline(self):
        rng = np.random.default_rng(7)
        coordinates = rng.standard_normal((24, 6))
        groups = np.repeat(np.arange(12), 2)
        result = cross_validated_reconstruction(
            coordinates,
            grouped_splits(groups, folds=3),
            [2],
            [1, 2],
            dictionary_alpha=0.05,
            seed=0,
            max_iter=20,
        )

        models = result["2"]["mean_relative_squared_error"]
        self.assertEqual(set(models), {"kmeans", "pca", "dictionary_1", "dictionary_2"})


if __name__ == "__main__":
    unittest.main()
