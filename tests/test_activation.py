import unittest

import numpy as np

from head_atlas.activation import (
    distance_spearman,
    layer_pair_matched_edge_test,
    normalized_distances_from_gram,
    stratified_distance_permutation_test,
)


class ActivationTests(unittest.TestCase):
    def test_gram_distances_recover_orthogonal_and_opposite_vectors(self):
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        distances = normalized_distances_from_gram(vectors @ vectors.T)

        self.assertAlmostEqual(distances[0, 1], np.sqrt(2.0))
        self.assertAlmostEqual(distances[0, 2], 2.0)
        np.testing.assert_allclose(np.diag(distances), 0.0)

    def test_distance_spearman_recovers_matching_order(self):
        points = np.asarray([[0.0], [1.0], [3.0], [7.0]])
        distances = np.abs(points - points.T)
        self.assertAlmostEqual(distance_spearman(distances, distances), 1.0)

    def test_stratified_test_preserves_group_effect_but_breaks_identity(self):
        points = np.asarray([[0.0], [1.0], [4.0], [10.0], [11.0], [15.0]])
        distances = np.abs(points - points.T)
        groups = np.asarray([0, 0, 0, 1, 1, 1])

        result = stratified_distance_permutation_test(
            distances,
            distances,
            groups,
            repetitions=49,
            rng=np.random.default_rng(3),
        )

        self.assertEqual(result["observed_spearman"], 1.0)
        self.assertGreater(result["observed_spearman"], result["null_mean"])
        self.assertGreaterEqual(result["upper_tail_p_value"], 1 / 50)

    def test_layer_pair_matched_edges_detect_close_cross_group_pairs(self):
        points = np.asarray([[0.0], [10.0], [0.1], [20.0]])
        distances = np.abs(points - points.T)
        layers = np.asarray([0, 0, 1, 1])

        result = layer_pair_matched_edge_test(
            distances,
            layers,
            edges=np.asarray([[0, 2]]),
            repetitions=99,
            rng=np.random.default_rng(5),
        )

        self.assertAlmostEqual(result["observed_mean_distance"], 0.1)
        self.assertLess(result["observed_mean_distance"], result["null_mean"])


if __name__ == "__main__":
    unittest.main()
