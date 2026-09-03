import unittest

import numpy as np

from head_atlas.qk_events import (
    batched_matched_source_events,
    causal_softmax,
    matched_source_events,
    max_attention_difference,
    offset_bin,
    qk_logits,
    relative_offset_statistics,
    residualize_by_offset,
)


class QKEventTests(unittest.TestCase):
    def test_qk_logits_and_causal_softmax_match_hand_calculation(self):
        queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        keys = np.asarray([[2.0, 0.0], [0.0, 4.0]])

        logits = qk_logits(queries, keys, scale=1.0)
        attention = causal_softmax(logits)

        np.testing.assert_allclose(logits, [[2.0, 0.0], [0.0, 4.0]])
        np.testing.assert_allclose(attention, [[1.0, 0.0], [1.0 / (1.0 + np.exp(4.0)), np.exp(4.0) / (1.0 + np.exp(4.0))]])

    def test_offset_statistics_and_residualization_use_destination_minus_source(self):
        logits = np.asarray(
            [
                [0.0, 100.0, 100.0],
                [2.0, 0.0, 100.0],
                [4.0, 6.0, 0.0],
            ]
        )

        means, standard_deviations = relative_offset_statistics(logits)
        residualized = residualize_by_offset(logits, means, standard_deviations)

        np.testing.assert_allclose(means, [0.0, 4.0, 4.0])
        np.testing.assert_allclose(standard_deviations, [0.0, 2.0, 0.0])
        self.assertAlmostEqual(residualized[1, 0], -1.0)
        self.assertAlmostEqual(residualized[2, 1], 1.0)
        self.assertTrue(np.isnan(residualized[0, 1]))

    def test_event_selector_matches_offset_and_uses_deterministic_ties(self):
        residualized = np.full((10, 10), np.nan)
        for destination in range(1, 10):
            residualized[destination, 1:destination] = 0.0
        residualized[8, 3] = 8.0  # offset 5, positive source
        residualized[8, 4] = -2.0  # offset 4, not eligible as matched negative
        residualized[8, 5] = 1.0
        residualized[8, 6] = 1.0

        events = matched_source_events(residualized, minimum_destination=8)

        selected = events[events[:, 0] == 8]
        # Offset 5 belongs to the [5, 8] stratum, whose eligible alternatives
        # are sources 1 and 2.  The lower-index tie is source 1.
        np.testing.assert_array_equal(selected, [[8, 3, 1, 2]])

    def test_batched_event_selection_preserves_each_logit_matrix(self):
        residualized = np.full((2, 9, 9), np.nan)
        for matrix in residualized:
            for destination in range(1, 9):
                matrix[destination, 1:destination] = 0.0
        residualized[0, 8, 3] = 5.0
        residualized[1, 8, 4] = 6.0

        events = batched_matched_source_events(residualized, minimum_destination=8)

        self.assertEqual(len(events), 2)
        np.testing.assert_array_equal(events[0][0, :3], [8, 3, 1])
        np.testing.assert_array_equal(events[1][0, :3], [8, 4, 5])

    def test_offset_bin_and_attention_difference_validate_input(self):
        self.assertEqual(offset_bin(1), 0)
        self.assertEqual(offset_bin(8), 2)
        self.assertEqual(offset_bin(64), -1)
        self.assertAlmostEqual(max_attention_difference([[0.0]], [[0.1]]), 0.1)
        with self.assertRaises(ValueError):
            max_attention_difference([[0.0]], [[0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
