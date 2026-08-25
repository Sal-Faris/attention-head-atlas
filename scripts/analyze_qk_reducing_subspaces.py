"""Run the frozen weight-only QK reducing-subspace experiment.

Discovery uses only training checkpoints. One resolution is chosen from the
primary validation checkpoint at the population level; confirmation checkpoints
never affect fitting or selection. The preregistered d=64, p=q=32 statistic is
reported separately from this exploratory selection.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_singular_components
from head_atlas.reducing_subspaces import (
    ReducingSubspaceFit,
    fit_reducing_subspaces_from_blocks,
    held_out_block_metrics_from_blocks,
    reducing_pair_overlap,
)
from head_atlas.trajectory_nulls import (
    SVDTrajectory,
    independent_spectrum_haar_null,
    smooth_singular_frame_trajectory_null,
    within_layer_side_trajectory_pairing_null,
)

Array = np.ndarray
T = TypeVar("T")
R = TypeVar("R")

CHECKPOINTS = (
    "step0",
    "step64",
    "step512",
    "step1000",
    "step4000",
    "step16000",
    "step64000",
    "step143000",
)

SPLITS = {
    "primary": {
        "train": ("step0", "step64", "step512", "step1000", "step4000"),
        "validation": ("step16000",),
        "confirmation": ("step64000", "step143000"),
        "stability_a": ("step0", "step512", "step4000"),
        "stability_b": ("step64", "step1000"),
    },
    "late_sensitivity": {
        "train": ("step1000", "step4000", "step16000"),
        "validation": ("step64000",),
        "confirmation": ("step143000",),
        "stability_a": ("step1000", "step16000"),
        "stability_b": ("step4000",),
    },
}

GRID = (
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
)

PRIMARY_CONFIG = (64, 32, 32)
SELECTION_METRIC = "gain_over_random"
DYNAMIC_METADATA_KEYS = {"revision", "snapshot_commit"}


@dataclass(frozen=True)
class HeadTrajectory:
    """One head's normalized compact QK trajectory and identifying metadata."""

    layer: int
    head: int
    trajectory: SVDTrajectory
    raw_frobenius_norms: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.layer < 0 or self.head < 0:
            raise ValueError("layer and head must be nonnegative")
        if not isinstance(self.trajectory, SVDTrajectory):
            raise TypeError("trajectory must be an SVDTrajectory")
        if self.trajectory.checkpoint_count != len(CHECKPOINTS):
            raise ValueError("trajectory must contain every frozen checkpoint")
        norms = np.asarray(self.raw_frobenius_norms, dtype=np.float64)
        if norms.shape != (len(CHECKPOINTS),) or np.any(~np.isfinite(norms)):
            raise ValueError("raw norms must be finite with one value per checkpoint")
        if np.any(norms <= 0.0):
            raise ValueError("raw norms must be positive")

    @property
    def key(self) -> str:
        return f"L{self.layer}H{self.head}"


def config_id(config: tuple[int, int, int]) -> str:
    """Return a stable JSON key for (support dimension, output rank, input rank)."""

    dimension, output_rank, input_rank = config
    return f"d{dimension:03d}_p{output_rank:03d}_q{input_rank:03d}"


def _seed(base_seed: int, *coordinates: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), *(int(value) for value in coordinates)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _parallel_map(function: Callable[[T], R], values: Sequence[T], workers: int) -> list[R]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, values))


def _resolve_factor_path(path: str | Path, manifest_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    choices = (candidate, manifest_path.parent / candidate, manifest_path.parent.parent / candidate)
    for choice in choices:
        if choice.exists():
            return choice
    return choices[-1]


def load_qk_trajectories(
    manifest_path: str | Path,
) -> tuple[tuple[HeadTrajectory, ...], dict[str, Any]]:
    """Load, verify, exactly decompose, and unit-normalize all frozen checkpoints."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_revision = {str(record["revision"]): record for record in manifest["records"]}
    missing = [revision for revision in CHECKPOINTS if revision not in by_revision]
    if missing:
        raise ValueError(f"manifest is missing frozen checkpoints: {missing}")

    checkpoint_components: list[list[tuple[Array, Array, Array]]] = []
    checkpoint_norms: list[list[float]] = []
    reference_order: tuple[tuple[int, int, str, tuple[int, int], tuple[int, int]], ...] | None = (
        None
    )
    reference_metadata: dict[str, Any] | None = None
    metadata_by_checkpoint: dict[str, dict[str, Any]] = {}

    for revision in CHECKPOINTS:
        record = by_revision[revision]
        factor_path = _resolve_factor_path(record["factors"]["QK"]["path"], path)
        operators, metadata = load_factor_bundle(factor_path)
        order = tuple(
            (
                operator.layer,
                operator.head,
                operator.kind,
                operator.left.shape,
                operator.right.shape,
            )
            for operator in operators
        )
        static_metadata = {
            key: value for key, value in metadata.items() if key not in DYNAMIC_METADATA_KEYS
        }
        if reference_order is None:
            reference_order = order
            reference_metadata = static_metadata
        elif order != reference_order:
            raise ValueError(f"head ordering or factor shapes differ at {revision}")
        elif static_metadata != reference_metadata:
            raise ValueError(f"structural factor metadata differ at {revision}")
        if metadata.get("revision") != revision or any(item[2] != "QK" for item in order):
            raise ValueError(f"factor bundle at {revision} is not the requested QK revision")

        components = [factorized_singular_components(operator) for operator in operators]
        norms = [float(np.linalg.norm(spectrum)) for _, spectrum, _ in components]
        if any(not np.isfinite(norm) or norm <= 0.0 for norm in norms):
            raise ValueError(f"QK bundle at {revision} contains a zero or non-finite operator")
        checkpoint_components.append(components)
        checkpoint_norms.append(norms)
        metadata_by_checkpoint[revision] = metadata

    assert reference_order is not None and reference_metadata is not None
    heads = []
    for operator_index, (layer, head, _, _, _) in enumerate(reference_order):
        left = np.stack([components[operator_index][0] for components in checkpoint_components])
        spectrum = np.stack(
            [
                components[operator_index][1] / checkpoint_norms[index][operator_index]
                for index, components in enumerate(checkpoint_components)
            ]
        )
        right = np.stack([components[operator_index][2] for components in checkpoint_components])
        heads.append(
            HeadTrajectory(
                layer=layer,
                head=head,
                trajectory=SVDTrajectory(left, spectrum, right),
                raw_frobenius_norms=tuple(
                    checkpoint_norms[index][operator_index] for index in range(len(CHECKPOINTS))
                ),
            )
        )
    load_report = {
        "manifest": str(path),
        "model": manifest.get("model"),
        "experiment_id": manifest.get("experiment_id"),
        "checkpoints": list(CHECKPOINTS),
        "head_count": len(heads),
        "structural_metadata": reference_metadata,
        "checkpoint_metadata": metadata_by_checkpoint,
        "normalization": "each complete QK operator divided by its Frobenius norm",
        "svd": "exact compact SVD from skinny factors; no ambient matrix SVD",
    }
    return tuple(heads), load_report


def _indices(split: dict[str, tuple[str, ...]], part: str) -> Array:
    return np.asarray([CHECKPOINTS.index(name) for name in split[part]], dtype=np.int64)


def _metrics_json(metrics: dict[str, Array | float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in metrics.items():
        array = np.asarray(value)
        result[name] = float(array) if array.ndim == 0 else array.astype(float).tolist()
    return result


def active_support_from_trajectory(
    trajectory: SVDTrajectory, indices: Array, support_dimension: int
) -> tuple[Array, Array]:
    """Compute exact active supports directly from compact singular factors."""

    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("indices must be a nonempty vector")
    if np.any(selected < 0) or np.any(selected >= trajectory.checkpoint_count):
        raise ValueError("indices contain an invalid checkpoint")
    if support_dimension < 1:
        raise ValueError("support_dimension must be positive")
    output_design = np.concatenate(
        [
            trajectory.left[index] * trajectory.singular_values[index][None, :]
            for index in selected
        ],
        axis=1,
    )
    input_design = np.concatenate(
        [
            trajectory.right[index] * trajectory.singular_values[index][None, :]
            for index in selected
        ],
        axis=1,
    )
    output_basis, output_values, _ = np.linalg.svd(output_design, full_matrices=False)
    input_basis, input_values, _ = np.linalg.svd(input_design, full_matrices=False)
    output_rank = int(np.count_nonzero(output_values > 1e-6 * output_values[0]))
    input_rank = int(np.count_nonzero(input_values > 1e-6 * input_values[0]))
    if support_dimension > output_rank or support_dimension > input_rank:
        raise ValueError("support_dimension exceeds the active trajectory rank")
    return output_basis[:, :support_dimension], input_basis[:, :support_dimension]


def trajectory_blocks(
    trajectory: SVDTrajectory,
    indices: Array,
    support: tuple[Array, Array],
) -> tuple[Array, Array]:
    """Project compact operators into support coordinates without materialization."""

    selected = np.asarray(indices, dtype=np.int64)
    output_basis, input_basis = support
    left_coordinates = np.einsum(
        "od,tor->tdr", output_basis, trajectory.left[selected], optimize=True
    )
    right_coordinates = np.einsum(
        "id,tir->tdr", input_basis, trajectory.right[selected], optimize=True
    )
    spectra = trajectory.singular_values[selected]
    blocks = np.einsum(
        "tdr,tr,ter->tde", left_coordinates, spectra, right_coordinates, optimize=True
    )
    full_energy = np.einsum("tr,tr->t", spectra, spectra, optimize=True)
    return blocks, full_energy


def _fit(
    training_blocks: Array,
    config: tuple[int, int, int],
    support: tuple[Array, Array],
    *,
    random_starts: int,
    max_iterations: int,
    seed: int,
) -> ReducingSubspaceFit:
    _, output_rank, input_rank = config
    return fit_reducing_subspaces_from_blocks(
        training_blocks,
        support[0],
        support[1],
        output_rank,
        input_rank,
        random_starts=random_starts,
        max_iterations=max_iterations,
        seed=seed,
    )


def evaluate_head_grid(
    head: HeadTrajectory,
    split: dict[str, tuple[str, ...]],
    grid: Sequence[tuple[int, int, int]],
    *,
    random_starts: int,
    max_iterations: int,
    random_projector_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Fit every requested resolution for one head, caching each active support."""

    train = _indices(split, "train")
    validation = _indices(split, "validation")
    confirmation = _indices(split, "confirmation")
    dimensions = sorted({config[0] for config in grid})
    supports = {
        dimension: active_support_from_trajectory(head.trajectory, train, dimension)
        for dimension in dimensions
    }
    projected = {
        dimension: {
            "train": trajectory_blocks(head.trajectory, train, supports[dimension]),
            "validation": trajectory_blocks(head.trajectory, validation, supports[dimension]),
            "confirmation": trajectory_blocks(head.trajectory, confirmation, supports[dimension]),
        }
        for dimension in dimensions
    }
    configurations = {}
    for config_index, config in enumerate(grid):
        dimension = config[0]
        fit = _fit(
            projected[dimension]["train"][0],
            config,
            supports[dimension],
            random_starts=random_starts,
            max_iterations=max_iterations,
            seed=_seed(seed, config_index, 0),
        )
        configurations[config_id(config)] = {
            "configuration": {
                "support_dimension": config[0],
                "output_rank": config[1],
                "input_rank": config[2],
            },
            "training_objective": fit.objective,
            "iterations": fit.iterations,
            "validation": _metrics_json(
                held_out_block_metrics_from_blocks(
                    *projected[dimension]["validation"],
                    fit,
                    random_repetitions=random_projector_repetitions,
                    seed=_seed(seed, config_index, 1),
                )
            ),
            "confirmation": _metrics_json(
                held_out_block_metrics_from_blocks(
                    *projected[dimension]["confirmation"],
                    fit,
                    random_repetitions=random_projector_repetitions,
                    seed=_seed(seed, config_index, 2),
                )
            ),
        }
    return {
        "key": head.key,
        "layer": head.layer,
        "head": head.head,
        "raw_frobenius_norms": dict(zip(CHECKPOINTS, head.raw_frobenius_norms, strict=True)),
        "configurations": configurations,
    }


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("population summaries require nonempty finite values")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def aggregate_grid(head_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all head/checkpoint metric values without hiding distributions."""

    if not head_reports:
        raise ValueError("at least one head report is required")
    keys = tuple(head_reports[0]["configurations"])
    if any(tuple(report["configurations"]) != keys for report in head_reports):
        raise ValueError("head reports have inconsistent configuration grids")
    population = {}
    for key in keys:
        phases = {}
        for phase in ("validation", "confirmation"):
            metric_names = tuple(head_reports[0]["configurations"][key][phase])
            phases[phase] = {
                metric: _summary(
                    value
                    for report in head_reports
                    for value in np.atleast_1d(report["configurations"][key][phase][metric]).astype(
                        float
                    )
                )
                for metric in metric_names
            }
        population[key] = phases
    return population


def select_validation_config(population: dict[str, Any]) -> str:
    """Select once on primary mean validation gain; never inspect confirmation."""

    if not population:
        raise ValueError("population grid cannot be empty")

    def selection_key(key: str) -> tuple[float, int, str]:
        value = population[key]["validation"][SELECTION_METRIC]["mean"]
        coordinate_count = int(key[1:4]) ** 2
        return float(value), -coordinate_count, key

    return max(population, key=selection_key)


def evaluate_stability(
    head: HeadTrajectory,
    split: dict[str, tuple[str, ...]],
    config: tuple[int, int, int],
    *,
    random_starts: int,
    max_iterations: int,
    seed: int,
) -> float:
    """Fit disjoint checkpoint subsets and compare ambient reducing pairs."""

    first_indices = _indices(split, "stability_a")
    second_indices = _indices(split, "stability_b")
    dimension = config[0]
    first_support = active_support_from_trajectory(
        head.trajectory, first_indices, dimension
    )
    second_support = active_support_from_trajectory(
        head.trajectory, second_indices, dimension
    )
    first = _fit(
        trajectory_blocks(head.trajectory, first_indices, first_support)[0],
        config,
        first_support,
        random_starts=random_starts,
        max_iterations=max_iterations,
        seed=_seed(seed, 0),
    )
    second = _fit(
        trajectory_blocks(head.trajectory, second_indices, second_support)[0],
        config,
        second_support,
        random_starts=random_starts,
        max_iterations=max_iterations,
        seed=_seed(seed, 1),
    )
    return reducing_pair_overlap(first, second)


def _evaluate_stability_population(
    heads: Sequence[HeadTrajectory],
    split: dict[str, tuple[str, ...]],
    config: tuple[int, int, int],
    *,
    random_starts: int,
    max_iterations: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    def evaluate(item: tuple[int, HeadTrajectory]) -> dict[str, Any]:
        index, head = item
        value = evaluate_stability(
            head,
            split,
            config,
            random_starts=random_starts,
            max_iterations=max_iterations,
            seed=_seed(seed, index),
        )
        return {"key": head.key, "layer": head.layer, "head": head.head, "overlap": value}

    per_head = _parallel_map(evaluate, list(enumerate(heads)), workers)
    return {"per_head": per_head, "population": _summary(item["overlap"] for item in per_head)}


def run_real_analysis(
    heads: Sequence[HeadTrajectory],
    *,
    grid: Sequence[tuple[int, int, int]] = GRID,
    primary_config: tuple[int, int, int] = PRIMARY_CONFIG,
    splits: dict[str, dict[str, tuple[str, ...]]] = SPLITS,
    random_starts: int,
    max_iterations: int,
    random_projector_repetitions: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Run both real-data splits, select once, and audit fixed-primary stability."""

    if primary_config not in grid:
        raise ValueError("the fixed primary configuration must be in the grid")
    split_reports = {}
    for split_index, (split_name, split) in enumerate(splits.items()):

        def evaluate(
            item: tuple[int, HeadTrajectory],
            active_split: dict[str, tuple[str, ...]] = split,
            active_split_index: int = split_index,
        ) -> dict[str, Any]:
            head_index, head = item
            return evaluate_head_grid(
                head,
                active_split,
                grid,
                random_starts=random_starts,
                max_iterations=max_iterations,
                random_projector_repetitions=random_projector_repetitions,
                seed=_seed(seed, active_split_index, head_index),
            )

        per_head = _parallel_map(evaluate, list(enumerate(heads)), workers)
        split_reports[split_name] = {
            "checkpoint_split": {key: list(value) for key, value in split.items()},
            "per_head": per_head,
            "population": aggregate_grid(per_head),
        }

    selected_key = select_validation_config(split_reports["primary"]["population"])
    stability = {
        "fixed_primary": {
            "primary": _evaluate_stability_population(
                heads,
                splits["primary"],
                primary_config,
                random_starts=random_starts,
                max_iterations=max_iterations,
                workers=workers,
                seed=_seed(seed, 700),
            )
        }
    }
    readouts = {
        "fixed_primary": {
            split_name: report["population"][config_id(primary_config)]
            for split_name, report in split_reports.items()
        },
        "validation_selected": {
            split_name: report["population"][selected_key]
            for split_name, report in split_reports.items()
        },
    }
    return {
        "analysis_roles": {
            "fixed_primary": {
                "configuration": config_id(primary_config),
                "role": "preregistered confirmatory real/null statistic",
            },
            "validation_selected": {
                "configuration": selected_key,
                "role": "exploratory population-wide selection on primary validation only",
                "criterion": f"maximum mean validation {SELECTION_METRIC}",
                "confirmation_used_for_selection": False,
            },
        },
        "splits": split_reports,
        "headline_readouts": readouts,
        "stability": stability,
    }


def _head_statistic(
    head_report: dict[str, Any], config: tuple[int, int, int], stability: float
) -> dict[str, Any]:
    configuration = head_report["configurations"][config_id(config)]
    result = {
        "key": head_report["key"],
        "layer": head_report["layer"],
        "head": head_report["head"],
        "stability_overlap": float(stability),
    }
    metric_names = (
        "active_support_energy_fraction",
        "block_diagonal_energy_fraction",
        "block_diagonal_concentration",
        "cross_leakage_fraction",
        "excess_concentration",
        "gain_over_random",
    )
    for phase in ("validation", "confirmation"):
        for metric in metric_names:
            result[f"{phase}_{metric}"] = float(np.mean(configuration[phase][metric]))
    return result


def summarize_fixed_statistic(
    head_reports: Sequence[dict[str, Any]],
    stability_report: dict[str, Any] | None,
    config: tuple[int, int, int],
) -> dict[str, Any]:
    if stability_report is None:
        stability = dict.fromkeys((report["key"] for report in head_reports), np.nan)
    else:
        stability = {item["key"]: item["overlap"] for item in stability_report["per_head"]}
    per_head = []
    for report in head_reports:
        item = _head_statistic(report, config, stability[report["key"]])
        if stability_report is None:
            del item["stability_overlap"]
        per_head.append(item)
    metric_names = [
        f"{phase}_{metric}"
        for phase in ("validation", "confirmation")
        for metric in (
            "active_support_energy_fraction",
            "block_diagonal_energy_fraction",
            "block_diagonal_concentration",
            "cross_leakage_fraction",
            "excess_concentration",
            "gain_over_random",
        )
    ]
    if stability_report is not None:
        metric_names.append("stability_overlap")
    return {
        "per_head": per_head,
        "population": {
            metric: _summary(item[metric] for item in per_head) for metric in metric_names
        },
    }


def generate_null_heads(
    heads: Sequence[HeadTrajectory], null_name: str, *, seed: int
) -> tuple[HeadTrajectory, ...]:
    """Regenerate one complete matched-null population deterministically."""

    rng = np.random.default_rng(seed)
    trajectories = tuple(head.trajectory for head in heads)
    if null_name == "independent_spectrum_haar":
        generated = tuple(independent_spectrum_haar_null(item, rng) for item in trajectories)
    elif null_name == "within_layer_side_trajectory_pairing":
        generated, _ = within_layer_side_trajectory_pairing_null(
            trajectories, [head.layer for head in heads], rng
        )
    elif null_name == "smooth_singular_frame_trajectory":
        generated = tuple(smooth_singular_frame_trajectory_null(item, rng) for item in trajectories)
    else:
        raise ValueError(f"unknown null: {null_name}")
    return tuple(
        HeadTrajectory(
            layer=head.layer,
            head=head.head,
            trajectory=trajectory,
            raw_frobenius_norms=head.raw_frobenius_norms,
        )
        for head, trajectory in zip(heads, generated, strict=True)
    )


def finite_upper_tail_p_value(observed: float, null_values: Sequence[float]) -> float:
    """Return the add-one finite-null upper-tail p-value."""

    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("null_values must be a nonempty finite vector")
    if not np.isfinite(observed):
        raise ValueError("observed must be finite")
    return float((1 + np.count_nonzero(values >= observed)) / (1 + values.size))


def finite_lower_tail_p_value(observed: float, null_values: Sequence[float]) -> float:
    """Return the add-one finite-null lower-tail p-value."""

    values = np.asarray(null_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("null_values must be a nonempty finite vector")
    if not np.isfinite(observed):
        raise ValueError("observed must be finite")
    return float((1 + np.count_nonzero(values <= observed)) / (1 + values.size))


def _fixed_split_analysis(
    heads: Sequence[HeadTrajectory],
    split: dict[str, tuple[str, ...]],
    config: tuple[int, int, int],
    *,
    random_starts: int,
    max_iterations: int,
    random_projector_repetitions: int,
    workers: int,
    seed: int,
    include_stability: bool,
) -> dict[str, Any]:
    def evaluate(item: tuple[int, HeadTrajectory]) -> dict[str, Any]:
        index, head = item
        return evaluate_head_grid(
            head,
            split,
            (config,),
            random_starts=random_starts,
            max_iterations=max_iterations,
            random_projector_repetitions=random_projector_repetitions,
            seed=_seed(seed, index, 0),
        )

    reports = _parallel_map(evaluate, list(enumerate(heads)), workers)
    stability = None
    if include_stability:
        stability = _evaluate_stability_population(
            heads,
            split,
            config,
            random_starts=random_starts,
            max_iterations=max_iterations,
            workers=workers,
            seed=_seed(seed, 1),
        )
    return summarize_fixed_statistic(reports, stability, config)


def _null_comparison(
    observed: dict[str, Any], repetitions: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    comparisons = {}
    for split_name, observed_split in observed.items():
        comparisons[split_name] = {}
        for metric, summary in observed_split["population"].items():
            null_values = [item[split_name]["population"][metric]["mean"] for item in repetitions]
            null_array = np.asarray(null_values, dtype=np.float64)
            comparisons[split_name][metric] = {
                "observed_population_mean": summary["mean"],
                "null_population_means": null_values,
                "null_mean": float(np.mean(null_array)),
                "null_standard_deviation": float(np.std(null_array)),
                "observed_minus_null_mean": float(summary["mean"] - np.mean(null_array)),
                "upper_tail_p_value": finite_upper_tail_p_value(summary["mean"], null_values),
            }
            if metric.endswith("cross_leakage_fraction"):
                comparisons[split_name][metric].update(
                    {
                        "direction_of_structural_evidence": "lower",
                        "lower_tail_p_value": finite_lower_tail_p_value(
                            summary["mean"], null_values
                        ),
                    }
                )
            else:
                comparisons[split_name][metric]["direction_of_structural_evidence"] = "higher"
    return comparisons


def run_null_analysis(
    heads: Sequence[HeadTrajectory],
    observed_real: dict[str, Any],
    *,
    primary_config: tuple[int, int, int] = PRIMARY_CONFIG,
    splits: dict[str, dict[str, tuple[str, ...]]] = SPLITS,
    null_repetitions: int,
    random_starts: int,
    max_iterations: int,
    random_projector_repetitions: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Regenerate and refit all three fixed-primary null hierarchies end to end."""

    if null_repetitions < 1:
        raise ValueError("null_repetitions must be positive")
    null_names = (
        "independent_spectrum_haar",
        "within_layer_side_trajectory_pairing",
        "smooth_singular_frame_trajectory",
    )
    results = {}
    for null_index, null_name in enumerate(null_names):
        repetitions = []
        for repetition in range(null_repetitions):
            repetition_seed = _seed(seed, null_index, repetition)
            null_heads = generate_null_heads(heads, null_name, seed=repetition_seed)
            split_results = {
                split_name: _fixed_split_analysis(
                    null_heads,
                    split,
                    primary_config,
                    random_starts=random_starts,
                    max_iterations=max_iterations,
                    random_projector_repetitions=random_projector_repetitions,
                    workers=workers,
                    seed=_seed(repetition_seed, split_index),
                    include_stability=split_name == "primary",
                )
                for split_index, (split_name, split) in enumerate(splits.items())
            }
            repetitions.append({"repetition": repetition, "seed": repetition_seed, **split_results})
            print(
                f"{null_name}: repetition {repetition + 1}/{null_repetitions}",
                flush=True,
            )
        results[null_name] = {
            "preservation": {
                "independent_spectrum_haar": "checkpoint spectra only",
                "within_layer_side_trajectory_pairing": (
                    "left/spectrum and right trajectories, layer, constant donor pairing"
                ),
                "smooth_singular_frame_trajectory": (
                    "checkpoint spectra and exact adjacent left/right frame overlaps"
                ),
            }[null_name],
            "repetitions": repetitions,
            "comparisons": _null_comparison(observed_real, repetitions),
        }
    return results


def observed_fixed_reports(
    real_analysis: dict[str, Any], primary_config: tuple[int, int, int]
) -> dict[str, Any]:
    """Extract the preregistered per-head statistic from the real grid."""

    return {
        split_name: summarize_fixed_statistic(
            split_report["per_head"],
            real_analysis["stability"]["fixed_primary"].get(split_name),
            primary_config,
        )
        for split_name, split_report in real_analysis["splits"].items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--random-starts", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--random-projector-repetitions", type=int, default=128)
    parser.add_argument("--null-repetitions", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_reducing_subspaces_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.random_starts < 0 or args.max_iterations < 1:
        raise ValueError("workers/max_iterations must be positive and random_starts nonnegative")
    if args.random_projector_repetitions < 1 or args.null_repetitions < 1:
        raise ValueError("random-projector and null repetitions must be positive")

    heads, load_report = load_qk_trajectories(args.manifest)
    print(f"Loaded {len(heads)} QK trajectories across {len(CHECKPOINTS)} checkpoints")
    real = run_real_analysis(
        heads,
        random_starts=args.random_starts,
        max_iterations=args.max_iterations,
        random_projector_repetitions=args.random_projector_repetitions,
        workers=args.workers,
        seed=args.seed,
    )
    observed = observed_fixed_reports(real, PRIMARY_CONFIG)
    nulls = run_null_analysis(
        heads,
        observed,
        null_repetitions=args.null_repetitions,
        random_starts=args.random_starts,
        max_iterations=args.max_iterations,
        random_projector_repetitions=args.random_projector_repetitions,
        workers=args.workers,
        seed=_seed(args.seed, 999),
    )
    report = {
        "schema": "qk-reducing-subspaces-v1",
        "scientific_scope": (
            "weight-only fixed within-head QK reducing compartments across training"
        ),
        "load": load_report,
        "protocol": {
            "normalization": "unit Frobenius per complete checkpoint operator",
            "grid": [config_id(config) for config in GRID],
            "fixed_primary": config_id(PRIMARY_CONFIG),
            "selection_metric": SELECTION_METRIC,
            "selection_scope": "one global choice from primary validation population mean",
            "confirmation_used_for_selection": False,
            "random_starts": args.random_starts,
            "max_iterations": args.max_iterations,
            "random_projector_repetitions": args.random_projector_repetitions,
            "null_repetitions": args.null_repetitions,
            "finite_null_minimum_p_value": 1.0 / (args.null_repetitions + 1),
            "seed": args.seed,
            "workers": args.workers,
        },
        "real": real,
        "confirmatory_observed": observed,
        "nulls": nulls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
