from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")


def _load_plot_module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "plot_qk_reducing_subspaces.py"
    spec = importlib.util.spec_from_file_location("plot_qk_reducing_subspaces", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(mean: float, count: int = 4) -> dict[str, float | int]:
    return {
        "count": count,
        "mean": mean,
        "standard_deviation": 0.01,
        "minimum": mean - 0.02,
        "q25": mean - 0.01,
        "median": mean,
        "q75": mean + 0.01,
        "maximum": mean + 0.02,
    }


def _synthetic_report() -> dict[str, Any]:
    configurations = [
        (32, 8, 8),
        (32, 8, 16),
        (32, 16, 8),
        (32, 16, 16),
        (64, 16, 16),
        (64, 16, 32),
        (64, 32, 16),
        (64, 32, 32),
        (96, 24, 24),
        (96, 24, 48),
        (96, 48, 24),
        (96, 48, 48),
    ]
    keys = [f"d{d:03d}_p{p:03d}_q{q:03d}" for d, p, q in configurations]
    primary_population = {
        key: {
            "validation": {"gain_over_random": _summary(0.02 + index / 1000)},
            "confirmation": {"gain_over_random": _summary(0.018 + index / 1200)},
        }
        for index, key in enumerate(keys)
    }
    heads = [
        {"key": f"L{layer}H{head}", "layer": layer, "head": head}
        for layer in range(2)
        for head in range(2)
    ]
    observed: dict[str, Any] = {}
    for split_index, split in enumerate(("primary", "late_sensitivity")):
        per_head = []
        for index, head in enumerate(heads):
            item = {
                **head,
                "confirmation_active_support_energy_fraction": 0.7 + 0.01 * index,
                "confirmation_gain_over_random": 0.08 + 0.01 * index - 0.005 * split_index,
            }
            if split == "primary":
                item["stability_overlap"] = 0.72 + 0.02 * index
            per_head.append(item)
        observed[split] = {
            "per_head": per_head,
            "population": {
                "confirmation_active_support_energy_fraction": _summary(0.72),
                "confirmation_gain_over_random": _summary(0.09),
                **({"stability_overlap": _summary(0.75)} if split == "primary" else {}),
            },
        }

    nulls: dict[str, Any] = {}
    null_names = (
        "independent_spectrum_haar",
        "within_layer_side_trajectory_pairing",
        "smooth_singular_frame_trajectory",
    )
    for null_index, null_name in enumerate(null_names):
        repetitions = []
        for repetition in range(3):
            record: dict[str, Any] = {"repetition": repetition, "seed": repetition}
            for split_index, split in enumerate(("primary", "late_sensitivity")):
                per_head = []
                for index, head in enumerate(heads):
                    item = {
                        **head,
                        "confirmation_active_support_energy_fraction": (
                            0.55 + 0.03 * null_index + 0.002 * repetition
                        ),
                        "confirmation_gain_over_random": (0.01 + 0.03 * null_index + 0.002 * index),
                    }
                    if split == "primary":
                        item["stability_overlap"] = 0.3 + 0.12 * null_index + 0.01 * index
                    per_head.append(item)
                record[split] = {"per_head": per_head, "population": {}}
            repetitions.append(record)

        comparisons = {}
        for split in ("primary", "late_sensitivity"):
            comparisons[split] = {}
            for metric, null_base in (
                ("confirmation_active_support_energy_fraction", 0.55 + 0.03 * null_index),
                ("confirmation_gain_over_random", 0.01 + 0.03 * null_index),
            ):
                null_values = [null_base + 0.002 * repetition for repetition in range(3)]
                effect = 0.09 - float(np.mean(null_values))
                comparisons[split][metric] = {
                    "observed_population_mean": 0.09,
                    "null_population_means": null_values,
                    "null_mean": float(np.mean(null_values)),
                    "null_standard_deviation": float(np.std(null_values)),
                    "observed_minus_null_mean": effect,
                    "upper_tail_p_value": 0.05 if null_index < 2 else 0.2,
                    "direction_of_structural_evidence": "higher",
                }
            comparisons[split]["stability_overlap"] = {
                "observed_population_mean": 0.75,
                "null_population_means": [0.3 + 0.12 * null_index] * 3,
                "null_mean": 0.3 + 0.12 * null_index,
                "null_standard_deviation": 0.0,
                "observed_minus_null_mean": 0.45 - 0.12 * null_index,
                "upper_tail_p_value": 0.05 if null_index < 2 else 0.2,
                "direction_of_structural_evidence": "higher",
            }
        nulls[null_name] = {
            "preservation": "synthetic",
            "repetitions": repetitions,
            "comparisons": comparisons,
        }

    return {
        "schema": "qk-reducing-subspaces-v1",
        "protocol": {"grid": keys, "fixed_primary": keys[7]},
        "real": {
            "analysis_roles": {
                "validation_selected": {"configuration": keys[-1]},
            },
            "splits": {
                "primary": {"population": primary_population},
                "late_sensitivity": {"population": primary_population},
            },
        },
        "confirmatory_observed": observed,
        "nulls": nulls,
    }


def test_decision_gate_requires_both_splits() -> None:
    module = _load_plot_module()
    passed, p_values = module.decision_matrix(_synthetic_report())

    assert passed.shape == (3, 3)
    assert p_values.shape == (2, 3)
    assert passed[-1].tolist() == [True, True, False]


def test_synthetic_report_renders_complete_png(tmp_path: Path) -> None:
    module = _load_plot_module()
    output = tmp_path / "reducing_subspaces.png"

    result = module.plot_report(_synthetic_report(), output)

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 20_000
