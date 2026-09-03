import unittest

import numpy as np

from head_atlas.trajectory_nulls import (
    SVDTrajectory,
    compact_svd_trajectory,
    full_svd_trajectory,
    independent_spectrum_haar_null,
    smooth_singular_frame_trajectory_null,
    within_group_side_trajectory_pairing_null,
)


def _frame(rows: int, columns: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    frame, _ = np.linalg.qr(generator.standard_normal((rows, columns)), mode="reduced")
    return frame


def _trajectory(offset: int = 0) -> SVDTrajectory:
    # Both ambient complements have at least rank columns, as required by the
    # smooth null construction.
    left = np.stack([_frame(8, 3, offset + checkpoint) for checkpoint in range(3)])
    right = np.stack([_frame(9, 3, 10 + offset + checkpoint) for checkpoint in range(3)])
    spectra = np.asarray([[5.0, 3.0, 1.0], [4.0, 2.5, 1.5], [3.5, 2.0, 0.5]])
    return SVDTrajectory(left, spectra + offset / 100.0, right)


class TrajectoryRepresentationTests(unittest.TestCase):
    def test_compact_and_full_svd_representations_reconstruct_operators(self):
        trajectory = _trajectory()
        matrices = trajectory.materialize()
        compact = compact_svd_trajectory(matrices)
        full = full_svd_trajectory(matrices)

        self.assertEqual(compact.rank, 3)
        self.assertEqual(full.rank, 8)
        np.testing.assert_allclose(compact.materialize(), matrices, atol=1e-12)
        np.testing.assert_allclose(full.materialize(), matrices, atol=1e-12)

    def test_compact_svd_rejects_changing_resolved_rank(self):
        matrices = np.zeros((2, 8, 9))
        matrices[0, :3, :3] = np.eye(3)
        matrices[1, :2, :2] = np.eye(2)
        with self.assertRaisesRegex(ValueError, "same at every checkpoint"):
            compact_svd_trajectory(matrices)


class TrajectoryNullTests(unittest.TestCase):
    def test_independent_spectrum_haar_preserves_spectra_and_is_deterministic(self):
        trajectory = _trajectory()
        first = independent_spectrum_haar_null(trajectory, np.random.default_rng(4))
        second = independent_spectrum_haar_null(trajectory, np.random.default_rng(4))

        np.testing.assert_array_equal(first.left, second.left)
        np.testing.assert_array_equal(first.right, second.right)
        np.testing.assert_array_equal(first.singular_values, trajectory.singular_values)
        np.testing.assert_allclose(
            np.linalg.svd(first.materialize(), compute_uv=False)[:, : trajectory.rank],
            trajectory.singular_values,
            atol=1e-12,
        )
        for frames in (first.left, first.right):
            np.testing.assert_allclose(
                frames.transpose(0, 2, 1) @ frames,
                np.broadcast_to(np.eye(3), (trajectory.checkpoint_count, 3, 3)),
                atol=1e-12,
            )

    def test_smooth_null_preserves_spectra_orthonormality_and_adjacent_overlaps(self):
        trajectory = _trajectory()
        null = smooth_singular_frame_trajectory_null(trajectory, np.random.default_rng(8))
        repeat = smooth_singular_frame_trajectory_null(trajectory, np.random.default_rng(8))

        np.testing.assert_array_equal(null.left, repeat.left)
        np.testing.assert_array_equal(null.right, repeat.right)
        np.testing.assert_array_equal(null.singular_values, trajectory.singular_values)
        for original, randomized in ((trajectory.left, null.left), (trajectory.right, null.right)):
            np.testing.assert_allclose(
                randomized.transpose(0, 2, 1) @ randomized,
                np.broadcast_to(np.eye(3), (trajectory.checkpoint_count, 3, 3)),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                randomized[:-1].transpose(0, 2, 1) @ randomized[1:],
                original[:-1].transpose(0, 2, 1) @ original[1:],
                atol=1e-10,
            )
        np.testing.assert_allclose(
            np.linalg.svd(null.materialize(), compute_uv=False)[:, : trajectory.rank],
            trajectory.singular_values,
            atol=1e-10,
        )

    def test_pairing_preserves_receiver_spectrum_and_left_frames_with_group_derangements(self):
        trajectories = tuple(_trajectory(offset) for offset in (0, 20, 40, 60))
        paired, donors = within_group_side_trajectory_pairing_null(
            trajectories, [0, 0, 1, 1], np.random.default_rng(12)
        )

        for index, output in enumerate(paired):
            self.assertNotEqual(index, donors[index])
            self.assertEqual([0, 0, 1, 1][index], [0, 0, 1, 1][donors[index]])
            np.testing.assert_array_equal(output.left, trajectories[index].left)
            np.testing.assert_array_equal(output.singular_values, trajectories[index].singular_values)
            np.testing.assert_array_equal(output.right, trajectories[donors[index]].right)
            np.testing.assert_allclose(
                np.linalg.svd(output.materialize(), compute_uv=False)[:, : trajectories[index].rank],
                trajectories[index].singular_values,
                atol=1e-10,
            )

    def test_validation_errors(self):
        trajectory = _trajectory()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            full_svd_trajectory(np.asarray([[[np.nan]]]))
        with self.assertRaisesRegex(ValueError, "orthogonal complement"):
            smooth_singular_frame_trajectory_null(
                SVDTrajectory(_frame(5, 3, 1)[None], np.ones((1, 3)), _frame(6, 3, 2)[None]),
                np.random.default_rng(2),
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            within_group_side_trajectory_pairing_null([trajectory], [0], np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "one entry"):
            within_group_side_trajectory_pairing_null([trajectory, trajectory], [0], np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
