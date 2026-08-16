import unittest

import numpy as np

from head_atlas.retrieval import evaluate_family_retrieval, permutation_retrieval_test


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        points = np.asarray([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])
        self.distances = np.abs(points - points.T)
        self.layers = np.asarray([0, 0, 1, 1, 2, 2])
        self.heads = np.asarray([0, 1, 0, 1, 0, 1])
        self.families = {
            "first": [(0, 0), (0, 1)],
            "second": [(1, 0), (1, 1)],
            "third": [(2, 0), (2, 1)],
        }

    def test_perfect_pair_families_retrieve_each_other(self):
        result = evaluate_family_retrieval(
            self.distances, self.layers, self.heads, self.families
        )

        self.assertAlmostEqual(result["aggregate"]["mean_average_precision"], 1.0)
        self.assertAlmostEqual(result["aggregate"]["mean_reciprocal_rank"], 1.0)
        self.assertAlmostEqual(result["aggregate"]["nearest_neighbour_hit_rate"], 1.0)
        self.assertAlmostEqual(result["aggregate"]["mean_recall_at_5"], 1.0)

    def test_permutation_test_is_reproducible(self):
        first = permutation_retrieval_test(
            self.distances,
            self.layers,
            self.heads,
            self.families,
            permutations=19,
            seed=4,
        )
        second = permutation_retrieval_test(
            self.distances,
            self.layers,
            self.heads,
            self.families,
            permutations=19,
            seed=4,
        )

        self.assertEqual(first, second)
        self.assertGreater(first["observed_family_balanced_map"], first["null_mean"])

    def test_overlapping_primary_families_are_rejected(self):
        overlapping = {"first": [(0, 0), (0, 1)], "second": [(0, 1), (1, 0)]}
        with self.assertRaisesRegex(ValueError, "cannot overlap"):
            evaluate_family_retrieval(
                self.distances, self.layers, self.heads, overlapping
            )


if __name__ == "__main__":
    unittest.main()

