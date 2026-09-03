import numpy as np

from scripts.pilot_channel_reuse_emergence import (
    aggregate,
    balanced_splits,
    shared_permute,
)


def test_balanced_splits_are_reusable_and_disjoint():
    splits = balanced_splits(4, 2)
    assert all(set(train).isdisjoint(test) and len(train) == len(test) == 4 for train, test in splits)


def test_layer_equal_aggregation():
    values = {"0": [{"overlap": 1.0, "equal": 2.0, "weighted": 3.0}] * 8, "1": [{"overlap": 3.0, "equal": 4.0, "weighted": 5.0}] * 8}
    assert aggregate(values) == {"overlap": 2.0, "equal": 3.0, "weighted": 4.0}


def test_shared_permutation_preserves_cross_checkpoint_reader_grams():
    rng = np.random.default_rng(8)
    readers = {
        checkpoint: {
            (layer, head): np.linalg.qr(rng.standard_normal((16, 4)), mode="reduced")[0]
            for layer in range(2)
            for head in range(8)
        }
        for checkpoint in (0, 10)
    }
    loaded = {
        checkpoint: ({}, checkpoint_readers, {})
        for checkpoint, checkpoint_readers in readers.items()
    }
    permuted = shared_permute(loaded, np.random.default_rng(11))

    for key in readers[0]:
        before = readers[0][key].T @ readers[10][key]
        after = permuted[0][1][key].T @ permuted[10][1][key]
        np.testing.assert_allclose(after, before, atol=1e-12)


def test_removing_one_layer_really_changes_layer_equal_aggregate():
    values = {
        "0": [{"overlap": 0.0, "equal": 0.0, "weighted": 0.0}],
        "1": [{"overlap": 2.0, "equal": 4.0, "weighted": 6.0}],
    }
    without_zero = {layer: rows for layer, rows in values.items() if layer != "0"}
    assert aggregate(without_zero) == {"overlap": 2.0, "equal": 4.0, "weighted": 6.0}
