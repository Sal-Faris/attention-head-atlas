from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from head_atlas.trajectory_nulls import SVDTrajectory
from scripts.analyze_qk_reducing_subspaces import CHECKPOINTS, HeadTrajectory
from scripts.audit_qk_reducing_convergence import (
    CONFIRMATION_METRICS,
    _validate_iteration_budgets,
    run_convergence_audit,
)


def _frame(rng: np.random.Generator, dimension: int, rank: int) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((dimension, rank)), mode="reduced")[0]


def _tiny_heads() -> tuple[HeadTrajectory, ...]:
    rng = np.random.default_rng(1701)
    heads = []
    for head_index in range(2):
        base_left = _frame(rng, 8, 4)
        base_right = _frame(rng, 8, 4)
        left = []
        right = []
        spectra = []
        for checkpoint in range(len(CHECKPOINTS)):
            left_step, _ = np.linalg.qr(
                base_left + checkpoint * 0.005 * rng.standard_normal(base_left.shape),
                mode="reduced",
            )
            right_step, _ = np.linalg.qr(
                base_right + checkpoint * 0.005 * rng.standard_normal(base_right.shape),
                mode="reduced",
            )
            spectrum = np.asarray([1.6, 1.0, 0.5, 0.2], dtype=np.float64)
            spectra.append(spectrum / np.linalg.norm(spectrum))
            left.append(left_step)
            right.append(right_step)
        heads.append(
            HeadTrajectory(
                layer=0,
                head=head_index,
                trajectory=SVDTrajectory(np.stack(left), np.stack(spectra), np.stack(right)),
                raw_frobenius_norms=tuple(1.0 for _ in CHECKPOINTS),
            )
        )
    return tuple(heads)


def _tiny_splits() -> dict[str, dict[str, tuple[str, ...]]]:
    split = {
        "train": ("step0", "step64", "step512"),
        "validation": ("step1000",),
        "confirmation": ("step4000", "step16000"),
        "stability_a": ("step0",),
        "stability_b": ("step64",),
    }
    late = {
        "train": ("step512", "step1000", "step4000"),
        "validation": ("step16000",),
        "confirmation": ("step64000",),
        "stability_a": ("step512",),
        "stability_b": ("step1000",),
    }
    return {"primary": split, "late_sensitivity": late}


def test_convergence_audit_is_complete_deterministic_and_json_serializable() -> None:
    kwargs = {
        "config": (4, 2, 2),
        "splits": _tiny_splits(),
        "iteration_budgets": (2, 4),
        "null_repetitions": 1,
        "random_starts": 0,
        "random_projector_repetitions": 2,
        "workers": 1,
        "seed": 23,
    }
    first = run_convergence_audit(_tiny_heads(), **kwargs)
    second = run_convergence_audit(_tiny_heads(), **kwargs)

    assert first == second
    assert first["protocol"]["iteration_budgets"] == [2, 4]
    assert first["protocol"]["stability_omitted"] is True
    assert first["protocol"]["each_null_population_is_shared_across_splits"] is True
    assert set(first["budgets"]) == {"2", "4"}
    for budget in first["budgets"].values():
        assert set(budget["splits"]) == {"primary", "late_sensitivity"}
        for split in budget["splits"].values():
            assert len(split["smooth_null_repetitions"]) == 1
            assert set(split["real"]["population"]) == set(CONFIRMATION_METRICS)
            assert set(split["comparison"]) == set(CONFIRMATION_METRICS)
            for metric in CONFIRMATION_METRICS:
                comparison = split["comparison"][metric]
                assert comparison["real_minus_null_by_repetition"] == pytest.approx(
                    [
                        comparison["real_population_mean"]
                        - comparison["null_population_means"][0]
                    ]
                )
    for budget in first["budgets"].values():
        assert (
            budget["splits"]["primary"]["smooth_null_repetitions"][0]["trajectory_seed"]
            == budget["splits"]["late_sensitivity"]["smooth_null_repetitions"][0][
                "trajectory_seed"
            ]
        )
    assert (
        first["budgets"]["2"]["splits"]["primary"]["real_seed"]
        == first["budgets"]["4"]["splits"]["primary"]["real_seed"]
    )
    for split in first["changes_across_budgets"].values():
        for metric in split.values():
            for series in metric.values():
                assert series["maximum_absolute_pairwise_change"] >= 0.0
                assert series["maximum_relative_pairwise_change"] >= 0.0
    json.dumps(first)


def test_validation_rejects_bad_budget_and_empty_heads() -> None:
    with pytest.raises(ValueError, match="positive"):
        _validate_iteration_budgets((0, 2))
    with pytest.raises(ValueError, match="unique"):
        _validate_iteration_budgets((2, 2))
    with pytest.raises(ValueError, match="at least one"):
        run_convergence_audit(
            (),
            config=(4, 2, 2),
            splits=_tiny_splits(),
            iteration_budgets=(2,),
            null_repetitions=1,
            random_starts=0,
            random_projector_repetitions=1,
        )


def test_direct_cli_help_from_repository_root() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "audit_qk_reducing_convergence.py"),
            "--help",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--iteration-budgets" in result.stdout
