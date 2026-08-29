import numpy as np

from scripts.pilot_invariant_channel_reuse import _primary_statistics, balanced_split_schedule


def test_balanced_schedule_is_deterministic_and_shared_shape() -> None:
    first = balanced_split_schedule()
    second = balanced_split_schedule()
    assert len(first) == 16
    assert all(len(training) == 4 and len(held_out) == 4 for training, held_out in first)
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(first, second, strict=True))


def test_balanced_schedule_supports_twelve_head_models() -> None:
    schedule = balanced_split_schedule(partner_count=12)
    assert all(len(training) == 6 and len(held_out) == 6 for training, held_out in schedule)
    assert all(
        set(training).union(held_out) == set(range(12))
        and not set(training).intersection(held_out)
        for training, held_out in schedule
    )


def test_primary_statistics_keeps_three_scales_separate() -> None:
    capture = np.full((3, 16), 0.2)
    weighted = np.full((3, 16), 0.4)
    overlap = np.full(3, 0.1)
    result = _primary_statistics(capture, weighted, overlap)
    expected = {
        "mean_pair_overlap": 0.1,
        "equal_partner_capture": 0.2,
        "overlap_weighted_capture": 0.4,
    }
    assert all(np.isclose(result[key], value) for key, value in expected.items())
