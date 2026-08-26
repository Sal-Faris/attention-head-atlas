"""Audit optimizer-budget robustness of the fixed QK reducing-subspace result.

This intentionally narrow audit reloads the real QK trajectories only once,
uses the preregistered primary resolution, omits stability fitting, and compares
the complete population with fresh smooth singular-frame null populations at
each iteration budget.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.analyze_qk_reducing_subspaces import (
        PRIMARY_CONFIG,
        SPLITS,
        HeadTrajectory,
        _fixed_split_analysis,
        _seed,
        config_id,
        generate_null_heads,
        load_qk_trajectories,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    # Direct execution makes this file's directory importable rather than the
    # repository root. Keep that supported alongside ``python -m scripts...``.
    from analyze_qk_reducing_subspaces import (  # type: ignore[no-redef]
        PRIMARY_CONFIG,
        SPLITS,
        HeadTrajectory,
        _fixed_split_analysis,
        _seed,
        config_id,
        generate_null_heads,
        load_qk_trajectories,
    )

DEFAULT_ITERATION_BUDGETS = (60, 120, 240)
DEFAULT_NULL_REPETITIONS = 3
DEFAULT_RANDOM_STARTS = 1
DEFAULT_RANDOM_PROJECTOR_REPETITIONS = 16

CONFIRMATION_METRICS = (
    "confirmation_active_support_energy_fraction",
    "confirmation_block_diagonal_concentration",
    "confirmation_gain_over_random",
)


def _validate_iteration_budgets(values: Sequence[int]) -> tuple[int, ...]:
    budgets = tuple(int(value) for value in values)
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("iteration budgets must be nonempty and positive")
    if len(set(budgets)) != len(budgets):
        raise ValueError("iteration budgets must be unique")
    return tuple(sorted(budgets))


def _confirmation_population(statistic: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the requested confirmation summaries from a fixed-split report."""

    population = statistic["population"]
    return {metric: population[metric] for metric in CONFIRMATION_METRICS}


def _population_means(statistic: Mapping[str, Any]) -> dict[str, float]:
    population = _confirmation_population(statistic)
    return {metric: float(summary["mean"]) for metric, summary in population.items()}


def evaluate_population(
    heads: Sequence[HeadTrajectory],
    split: dict[str, tuple[str, ...]],
    config: tuple[int, int, int],
    *,
    random_starts: int,
    max_iterations: int,
    random_projector_repetitions: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Fit one full population and return confirmation-only summaries."""

    statistic = _fixed_split_analysis(
        heads,
        split,
        config,
        random_starts=random_starts,
        max_iterations=max_iterations,
        random_projector_repetitions=random_projector_repetitions,
        workers=workers,
        seed=seed,
        include_stability=False,
    )
    return {
        "head_count": len(heads),
        "population": _confirmation_population(statistic),
    }


def _comparison(
    real: Mapping[str, float], null_repetitions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = {}
    for metric in CONFIRMATION_METRICS:
        real_mean = float(real[metric])
        null_values = [float(repetition["population_means"][metric]) for repetition in null_repetitions]
        null_mean = float(np.mean(null_values))
        differences = [real_mean - value for value in null_values]
        result[metric] = {
            "real_population_mean": real_mean,
            "null_population_means": null_values,
            "null_population_mean": null_mean,
            "real_minus_null_by_repetition": differences,
            "real_minus_null_mean": real_mean - null_mean,
        }
    return result


def _series_change(values_by_budget: Mapping[str, float], budgets: Sequence[int]) -> dict[str, Any]:
    values = np.asarray([values_by_budget[str(budget)] for budget in budgets], dtype=np.float64)
    reference = float(values[-1])
    maximum_absolute = float(np.max(values) - np.min(values))
    scale = max(abs(reference), np.finfo(np.float64).eps)
    return {
        "values_by_iteration_budget": {
            str(budget): float(value) for budget, value in zip(budgets, values, strict=True)
        },
        "largest_budget_reference": reference,
        "maximum_absolute_pairwise_change": maximum_absolute,
        "maximum_relative_pairwise_change": maximum_absolute / scale,
        "relative_change_denominator": "absolute value at largest iteration budget",
    }


def summarize_budget_changes(
    budget_reports: Mapping[str, Any],
    budgets: Sequence[int],
    split_names: Sequence[str],
) -> dict[str, Any]:
    """Summarize sensitivity of real, null, and effect-size means to budget."""

    result: dict[str, Any] = {}
    for split_name in split_names:
        result[split_name] = {}
        for metric in CONFIRMATION_METRICS:
            result[split_name][metric] = {}
            for series_name, field in (
                ("real_population_mean", "real_population_mean"),
                ("null_population_mean", "null_population_mean"),
                ("real_minus_null_mean", "real_minus_null_mean"),
            ):
                values = {
                    str(budget): float(
                        budget_reports[str(budget)]["splits"][split_name]["comparison"][metric][
                            field
                        ]
                    )
                    for budget in budgets
                }
                result[split_name][metric][series_name] = _series_change(values, budgets)
    return result


def run_convergence_audit(
    heads: Sequence[HeadTrajectory],
    *,
    config: tuple[int, int, int] = PRIMARY_CONFIG,
    splits: dict[str, dict[str, tuple[str, ...]]] = SPLITS,
    iteration_budgets: Sequence[int] = DEFAULT_ITERATION_BUDGETS,
    null_repetitions: int = DEFAULT_NULL_REPETITIONS,
    random_starts: int = DEFAULT_RANDOM_STARTS,
    random_projector_repetitions: int = DEFAULT_RANDOM_PROJECTOR_REPETITIONS,
    workers: int = 1,
    seed: int = 20260826,
) -> dict[str, Any]:
    """Run the complete real-versus-smooth-null convergence audit."""

    heads = tuple(heads)
    budgets = _validate_iteration_budgets(iteration_budgets)
    if not heads:
        raise ValueError("at least one head trajectory is required")
    if null_repetitions < 1:
        raise ValueError("null_repetitions must be positive")
    if random_starts < 0:
        raise ValueError("random_starts must be nonnegative")
    if random_projector_repetitions < 1 or workers < 1:
        raise ValueError("random projector repetitions and workers must be positive")

    budget_reports: dict[str, Any] = {}
    for budget_index, budget in enumerate(budgets):
        real_by_split: dict[str, tuple[int, dict[str, Any]]] = {}
        for split_index, (split_name, split) in enumerate(splits.items()):
            # Hold starts and random-projector baselines fixed across budgets so
            # the real-data change isolates the iteration cap.
            real_seed = _seed(seed, 0, split_index)
            real_report = evaluate_population(
                heads,
                split,
                config,
                random_starts=random_starts,
                max_iterations=budget,
                random_projector_repetitions=random_projector_repetitions,
                workers=workers,
                seed=real_seed,
            )
            real_by_split[split_name] = (real_seed, real_report)

        # A repetition is one fresh complete null population shared by both
        # splits, mirroring the fact that both splits inspect the same models.
        null_by_split: dict[str, list[dict[str, Any]]] = {
            split_name: [] for split_name in splits
        }
        for repetition in range(null_repetitions):
            null_seed = _seed(seed, 1, budget_index, repetition)
            null_heads = generate_null_heads(
                heads,
                "smooth_singular_frame_trajectory",
                seed=null_seed,
            )
            for split_index, (split_name, split) in enumerate(splits.items()):
                fit_seed = _seed(seed, 2, split_index, repetition)
                null_report = evaluate_population(
                    null_heads,
                    split,
                    config,
                    random_starts=random_starts,
                    max_iterations=budget,
                    random_projector_repetitions=random_projector_repetitions,
                    workers=workers,
                    seed=fit_seed,
                )
                null_by_split[split_name].append(
                    {
                        "repetition": repetition,
                        "trajectory_seed": null_seed,
                        "fit_seed": fit_seed,
                        "population_means": {
                            metric: float(summary["mean"])
                            for metric, summary in null_report["population"].items()
                        },
                    }
                )

        split_reports: dict[str, Any] = {}
        for split_name in splits:
            real_seed, real_report = real_by_split[split_name]
            real_means = {
                metric: float(summary["mean"])
                for metric, summary in real_report["population"].items()
            }
            null_reports = null_by_split[split_name]
            split_reports[split_name] = {
                "real_seed": real_seed,
                "real": real_report,
                "smooth_null_repetitions": null_reports,
                "comparison": _comparison(real_means, null_reports),
            }
        budget_reports[str(budget)] = {
            "max_iterations": budget,
            "splits": split_reports,
        }

    return {
        "schema": "qk-reducing-convergence-audit-v1",
        "protocol": {
            "configuration": config_id(config),
            "iteration_budgets": list(budgets),
            "random_starts": random_starts,
            "random_projector_repetitions": random_projector_repetitions,
            "null_family": "smooth_singular_frame_trajectory",
            "null_repetitions_per_budget": null_repetitions,
            "each_null_population_is_shared_across_splits": True,
            "null_populations_are_fresh_across_budgets": True,
            "stability_omitted": True,
            "seed": seed,
            "workers": workers,
        },
        "budgets": budget_reports,
        "changes_across_budgets": summarize_budget_changes(
            budget_reports, budgets, tuple(splits)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--iteration-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_ITERATION_BUDGETS),
    )
    parser.add_argument("--random-starts", type=int, default=DEFAULT_RANDOM_STARTS)
    parser.add_argument(
        "--random-projector-repetitions",
        type=int,
        default=DEFAULT_RANDOM_PROJECTOR_REPETITIONS,
    )
    parser.add_argument("--null-repetitions", type=int, default=DEFAULT_NULL_REPETITIONS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/qk_reducing_convergence_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heads, load_report = load_qk_trajectories(args.manifest)
    print(f"Loaded {len(heads)} QK trajectories once")
    audit = run_convergence_audit(
        heads,
        iteration_budgets=args.iteration_budgets,
        null_repetitions=args.null_repetitions,
        random_starts=args.random_starts,
        random_projector_repetitions=args.random_projector_repetitions,
        workers=args.workers,
        seed=args.seed,
    )
    report = {**audit, "load": load_report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
