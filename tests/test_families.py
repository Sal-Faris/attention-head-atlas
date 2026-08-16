import unittest

import numpy as np

from head_atlas.families import (
    best_silhouette_cut,
    mean_neighbor_overlap,
    nearest_neighbor_indices,
    stratified_permutation,
    subsampled_cluster_stability,
)


class FamilyTests(unittest.TestCase):
    def test_cross_group_neighbors_exclude_same_group(self):
        points = np.asarray([[0.0], [0.1], [1.0], [1.1]])
        distances = np.abs(points - points.T)
        groups = np.asarray([0, 0, 1, 1])

        neighbors = nearest_neighbor_indices(
            distances, 1, groups=groups, different_group_only=True
        )

        self.assertTrue(np.all(groups[neighbors[:, 0]] != groups))

    def test_neighbor_overlap_uses_fraction_of_retained_neighbors(self):
        first = np.asarray([[1, 2], [0, 2], [0, 1]])
        second = np.asarray([[2, 1], [2, 0], [1, 0]])
        self.assertEqual(mean_neighbor_overlap(first, second), 1.0)

    def test_stratified_permutation_preserves_groups(self):
        groups = np.repeat(np.arange(3), 5)
        permutation = stratified_permutation(groups, np.random.default_rng(4))

        np.testing.assert_array_equal(groups[permutation], groups)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(len(groups)))

    def test_stable_separated_clusters_survive_subsampling(self):
        points = np.asarray([[-10.2], [-10.1], [-10.0], [10.0], [10.1], [10.2]])
        distances = np.abs(points - points.T)

        cut = best_silhouette_cut(distances, maximum_clusters=3)
        stability = subsampled_cluster_stability(
            distances,
            cluster_count=2,
            repetitions=20,
            sample_fraction=0.8,
            rng=np.random.default_rng(2),
        )

        self.assertEqual(cut["cluster_count"], 2)
        self.assertGreater(cut["silhouette"], 0.98)
        self.assertEqual(stability["mean_adjusted_rand"], 1.0)


if __name__ == "__main__":
    unittest.main()
