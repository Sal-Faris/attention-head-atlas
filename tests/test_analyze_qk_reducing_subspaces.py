from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from head_atlas.factor_io import save_factor_bundle
from head_atlas.factors import FactorizedHeadOperator
from head_atlas.reducing_subspaces import active_support_bases
from head_atlas.trajectory_nulls import SVDTrajectory
from scripts.analyze_qk_reducing_subspaces import (
    CHECKPOINTS,
    HeadTrajectory,
    active_support_from_trajectory,
    aggregate_grid,
    config_id,
    finite_upper_tail_p_value,
    generate_null_heads,
    load_qk_trajectories,
    observed_fixed_reports,
    run_null_analysis,
    run_real_analysis,
    select_validation_config,
    trajectory_blocks,
)


def _frame(rng: np.random.Generator, dimension: int, rank: int) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((dimension, rank)), mode="reduced")[0]


def _heads() -> tuple[HeadTrajectory, ...]:
    rng = np.random.default_rng(7)
    heads = []
    for layer in range(2):
        for head in range(2):
            left = _frame(rng, 8, 4)
            right = _frame(rng, 8, 4)
            left_trajectory = []
            right_trajectory = []
            spectra = []
            for checkpoint in range(len(CHECKPOINTS)):
                left_step, _ = np.linalg.qr(
                    left + 0.015 * checkpoint * rng.standard_normal(left.shape),
                    mode="reduced",
                )
                right_step, _ = np.linalg.qr(
                    right + 0.015 * checkpoint * rng.standard_normal(right.shape),
                    mode="reduced",
                )
                spectrum = np.asarray([1.5, 1.0, 0.55, 0.25])
                spectrum *= 1.0 + 0.01 * checkpoint * np.asarray([1.0, -0.3, 0.2, -0.1])
                left_trajectory.append(left_step)
                right_trajectory.append(right_step)
                spectra.append(spectrum / np.linalg.norm(spectrum))
            heads.append(
                HeadTrajectory(
                    layer,
                    head,
                    SVDTrajectory(
                        np.stack(left_trajectory),
                        np.stack(spectra),
                        np.stack(right_trajectory),
                    ),
                    tuple(float(index + 1) for index in range(len(CHECKPOINTS))),
                )
            )
    return tuple(heads)


def test_config_id_and_finite_upper_tail_p_value() -> None:
    assert config_id((64, 32, 16)) == "d064_p032_q016"
    assert finite_upper_tail_p_value(2.0, [1.0, 2.0, 3.0]) == pytest.approx(0.75)
    assert finite_upper_tail_p_value(4.0, [1.0] * 19) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="nonempty"):
        finite_upper_tail_p_value(1.0, [])


def test_aggregate_and_selection_use_validation_only() -> None:
    reports = []
    for head, first_validation, second_validation in ((0, 0.1, 0.4), (1, 0.3, 0.2)):
        configurations = {}
        for key, validation, confirmation in (
            ("d002_p001_q001", first_validation, 100.0),
            ("d004_p002_q002", second_validation, -100.0),
        ):
            configurations[key] = {
                "validation": {"gain_over_random": [validation]},
                "confirmation": {"gain_over_random": [confirmation]},
            }
        reports.append({"configurations": configurations, "key": f"L0H{head}"})
    population = aggregate_grid(reports)
    assert population["d002_p001_q001"]["validation"]["gain_over_random"]["mean"] == 0.2
    assert select_validation_config(population) == "d004_p002_q002"


def test_null_generators_preserve_required_quantities() -> None:
    heads = _heads()
    smooth = generate_null_heads(heads, "smooth_singular_frame_trajectory", seed=9)
    independent = generate_null_heads(heads, "independent_spectrum_haar", seed=9)
    paired = generate_null_heads(heads, "within_layer_side_trajectory_pairing", seed=9)
    for original, smooth_head, independent_head, paired_head in zip(
        heads, smooth, independent, paired, strict=True
    ):
        np.testing.assert_allclose(
            smooth_head.trajectory.singular_values, original.trajectory.singular_values
        )
        np.testing.assert_allclose(
            independent_head.trajectory.singular_values, original.trajectory.singular_values
        )
        np.testing.assert_allclose(
            paired_head.trajectory.singular_values, original.trajectory.singular_values
        )
        for checkpoint in range(1, len(CHECKPOINTS)):
            expected = (
                original.trajectory.left[checkpoint - 1].T @ original.trajectory.left[checkpoint]
            )
            observed = (
                smooth_head.trajectory.left[checkpoint - 1].T
                @ smooth_head.trajectory.left[checkpoint]
            )
            np.testing.assert_allclose(observed, expected, atol=1e-9)
        np.testing.assert_allclose(paired_head.trajectory.left, original.trajectory.left)


def test_factorized_support_and_blocks_equal_dense_computation() -> None:
    trajectory = _heads()[0].trajectory
    indices = np.asarray([0, 1, 2, 3])
    matrices = trajectory.materialize()[indices]
    output_basis, input_basis = active_support_from_trajectory(trajectory, indices, 4)
    dense_output, dense_input = active_support_bases(matrices, 4)

    np.testing.assert_allclose(
        output_basis @ output_basis.T,
        dense_output @ dense_output.T,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        input_basis @ input_basis.T,
        dense_input @ dense_input.T,
        atol=1e-9,
    )
    blocks, full_energy = trajectory_blocks(
        trajectory, indices, (output_basis, input_basis)
    )
    expected = np.einsum(
        "oi,tij,jk->tok", output_basis.T, matrices, input_basis, optimize=True
    )
    np.testing.assert_allclose(blocks, expected, atol=1e-10)
    np.testing.assert_allclose(
        full_energy,
        np.einsum("tij,tij->t", matrices, matrices, optimize=True),
        atol=1e-10,
    )


def test_synthetic_real_and_null_pipeline_is_json_serializable() -> None:
    heads = _heads()
    grid = ((2, 1, 1), (4, 1, 1), (4, 2, 2))
    primary = (4, 2, 2)
    real = run_real_analysis(
        heads,
        grid=grid,
        primary_config=primary,
        random_starts=0,
        max_iterations=8,
        random_projector_repetitions=3,
        workers=2,
        seed=3,
    )
    assert real["analysis_roles"]["validation_selected"]["confirmation_used_for_selection"] is False
    assert set(real["stability"]["fixed_primary"]) == {"primary"}
    assert len(real["splits"]["primary"]["per_head"]) == 4
    observed = observed_fixed_reports(real, primary)
    nulls = run_null_analysis(
        heads,
        observed,
        primary_config=primary,
        null_repetitions=1,
        random_starts=0,
        max_iterations=5,
        random_projector_repetitions=2,
        workers=2,
        seed=4,
    )
    assert set(nulls) == {
        "independent_spectrum_haar",
        "within_layer_side_trajectory_pairing",
        "smooth_singular_frame_trajectory",
    }
    assert "stability_overlap" not in observed["late_sensitivity"]["population"]
    json.dumps({"real": real, "observed": observed, "nulls": nulls})


def test_loader_uses_exact_compact_svd_without_real_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    records = []
    expected_norms = []
    for checkpoint_index, revision in enumerate(CHECKPOINTS):
        operators = []
        checkpoint_norms = []
        for head in range(2):
            left = rng.standard_normal((6, 2))
            right = rng.standard_normal((6, 2))
            operator = FactorizedHeadOperator(0, head, "QK", left, right)
            operators.append(operator)
            checkpoint_norms.append(float(np.linalg.norm(operator.materialize())))
        factor_path = tmp_path / revision / "qk_factors.npz"
        save_factor_bundle(
            factor_path,
            operators,
            {
                "revision": revision,
                "snapshot_commit": str(checkpoint_index),
                "model": "synthetic",
                "n_layers": 1,
                "n_heads": 2,
                "model_type": "synthetic",
                "source_format": "test",
                "weight_processing": "none",
            },
        )
        records.append({"revision": revision, "factors": {"QK": {"path": str(factor_path)}}})
        expected_norms.append(checkpoint_norms)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"model": "synthetic", "experiment_id": "test", "records": records}),
        encoding="utf-8",
    )
    heads, report = load_qk_trajectories(manifest_path)
    assert len(heads) == 2
    assert report["head_count"] == 2
    for head_index, head in enumerate(heads):
        np.testing.assert_allclose(np.linalg.norm(head.trajectory.singular_values, axis=1), 1.0)
        np.testing.assert_allclose(
            head.raw_frobenius_norms,
            [norms[head_index] for norms in expected_norms],
        )
