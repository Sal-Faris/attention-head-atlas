import numpy as np

from scripts.confirm_causal_invariant_bus import (
    aggregate_outcomes,
    ambient_channel,
    infer_population,
    random_channel,
)


def test_ambient_channel_is_rank_k_projector_inside_writer_span():
    rng = np.random.default_rng(2)
    writer, _ = np.linalg.qr(rng.standard_normal((20, 8)))
    readers = tuple(np.linalg.qr(rng.standard_normal((20, 8)))[0] for _ in range(4))
    projector = ambient_channel(writer, readers, 4)
    assert np.allclose(projector, projector.T)
    assert np.allclose(projector @ projector, projector, atol=1e-10)
    assert np.isclose(np.trace(projector), 4)
    assert np.linalg.norm((np.eye(20) - writer @ writer.T) @ projector) < 1e-10


def test_random_channel_has_requested_rank_and_is_deterministic():
    rng = np.random.default_rng(3)
    writer, _ = np.linalg.qr(rng.standard_normal((20, 8)))
    projector = random_channel(writer, 4, np.random.default_rng(4))
    assert np.isclose(np.trace(projector), 4)
    assert np.allclose(projector @ projector, projector, atol=1e-10)


def test_layer_equal_aggregation_and_iut():
    records = [
        {"source_layer": 0, "controls": [{"js_divergence": 2.0, "removed_energy_fraction": 1.0}]},
        {"source_layer": 0, "controls": [{"js_divergence": 4.0, "removed_energy_fraction": 1.0}]},
        {"source_layer": 1, "controls": [{"js_divergence": 6.0, "removed_energy_fraction": 2.0}]},
    ]
    aggregate = aggregate_outcomes(records)
    assert aggregate["layer_equal_mean_js"] == 4.5
    assert aggregate["layer_equal_ratio_of_sums_js_energy"] == 3.0
    result = infer_population({"a": 2.0, "b": 3.0}, [{"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 1.0}])
    assert result["IUT_p_value"] == 1 / 3
