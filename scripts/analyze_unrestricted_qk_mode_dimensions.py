"""Discover intrinsic dimensions of unrestricted recurring QK population modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_shared_operator_compression import pairing_shuffled, spectrum_matched_haar
from scipy.optimize import linear_sum_assignment

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import FactorizedHeadOperator
from head_atlas.operator_modes import (
    dictionary_variance_recovered,
    principal_operator_modes,
    singular_values_dimension_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/pythia-70m-deduped-pilot.json")
    )
    parser.add_argument("--model-revision", default="step143000")
    parser.add_argument("--component-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--truncation-ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--primary-components", type=int, default=4)
    parser.add_argument("--primary-truncation-rank", type=int, default=64)
    parser.add_argument("--null-repetitions", type=int, default=9)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/unrestricted_qk_mode_dimensions_v1.json"),
    )
    return parser.parse_args()


def normalized_population(operators: list[FactorizedHeadOperator]) -> np.ndarray:
    matrices = np.stack([operator.materialize(dtype=np.float32) for operator in operators])
    norms = np.linalg.norm(matrices, axis=(1, 2), keepdims=True)
    return matrices / np.maximum(norms, 1e-12)


def split_folds(
    operators: list[FactorizedHeadOperator], scheme: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(
        [operator.head if scheme == "head_parity" else operator.layer for operator in operators]
    )
    return [
        (np.flatnonzero(labels % 2 != parity), np.flatnonzero(labels % 2 == parity))
        for parity in (0, 1)
    ]


def fold_report(
    matrices: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    component_counts: list[int],
    truncation_ranks: list[int],
) -> dict[str, object]:
    maximum = max(component_counts)
    modes = principal_operator_modes(matrices[train_indices], maximum)
    decompositions = [np.linalg.svd(mode, full_matrices=False) for mode in modes]
    dimensions = [
        singular_values_dimension_summary(singular)
        for _, singular, _ in decompositions
    ]
    test = matrices[test_indices]
    curves = {}
    for count in component_counts:
        selected = modes[: min(count, len(modes))]
        by_rank = {}
        for rank in truncation_ranks:
            truncated = np.stack(
                [
                    (left[:, :rank] * singular[:rank]) @ right[:rank]
                    for left, singular, right in decompositions[: len(selected)]
                ]
            )
            by_rank[str(rank)] = dictionary_variance_recovered(test, truncated)
        curves[str(count)] = {
            "unrestricted_full_rank_variance_recovered": dictionary_variance_recovered(
                test, selected
            ),
            "truncated_variance_recovered": by_rank,
        }
    return {"curves": curves, "mode_dimensions": dimensions}


def aggregate_reports(folds: list[dict[str, object]]) -> dict[str, object]:
    component_keys = list(folds[0]["curves"])
    curve_report = {}
    for count in component_keys:
        rank_keys = list(folds[0]["curves"][count]["truncated_variance_recovered"])
        curve_report[count] = {
            "unrestricted_full_rank_variance_recovered": float(
                np.mean(
                    [fold["curves"][count]["unrestricted_full_rank_variance_recovered"] for fold in folds]
                )
            ),
            "truncated_variance_recovered": {
                rank: float(
                    np.mean(
                        [fold["curves"][count]["truncated_variance_recovered"][rank] for fold in folds]
                    )
                )
                for rank in rank_keys
            },
        }
    summary_keys = list(folds[0]["mode_dimensions"][0])
    mode_count = min(len(fold["mode_dimensions"]) for fold in folds)
    dimensions = [
        {
            key: float(np.mean([fold["mode_dimensions"][index][key] for fold in folds]))
            for key in summary_keys
        }
        for index in range(mode_count)
    ]
    return {"curves": curve_report, "mean_mode_dimensions_across_folds": dimensions}


def cross_validated(
    operators: list[FactorizedHeadOperator],
    scheme: str,
    component_counts: list[int],
    truncation_ranks: list[int],
) -> dict[str, object]:
    matrices = normalized_population(operators)
    splits = split_folds(operators, scheme)
    folds = [
        fold_report(matrices, train, test, component_counts, truncation_ranks)
        for train, test in splits
    ]
    result = aggregate_reports(folds)
    stability = {}
    for count in component_counts:
        first = principal_operator_modes(matrices[splits[0][0]], count)
        second = principal_operator_modes(matrices[splits[1][0]], count)
        overlap = np.einsum("kij,lij->kl", first, second, optimize=True)
        similarity = np.abs(overlap)
        rows, columns = linear_sum_assignment(-similarity)
        matched = similarity[rows, columns]
        canonical = np.linalg.svd(overlap, compute_uv=False)
        stability[str(count)] = {
            "matched_mode_cosines": matched.tolist(),
            "mean_matched_mode_cosine": float(np.mean(matched)),
            "median_matched_mode_cosine": float(np.median(matched)),
            "canonical_cosines_between_mode_spans": canonical.tolist(),
            "mode_span_overlap_fraction": float(np.sum(canonical**2) / count),
        }
    result["disjoint_training_mode_stability"] = stability
    return result


def primary_values(report: dict[str, object], components: int, rank: int) -> dict[str, float]:
    curve = report["curves"][str(components)]
    dimensions = report["mean_mode_dimensions_across_folds"][:components]
    return {
        "unrestricted_variance_recovered": float(
            curve["unrestricted_full_rank_variance_recovered"]
        ),
        "rank_truncated_variance_recovered": float(
            curve["truncated_variance_recovered"][str(rank)]
        ),
        "median_entropy_rank": float(
            np.median([mode["entropy_rank"] for mode in dimensions])
        ),
        "median_rank_90_percent_energy": float(
            np.median([mode["rank_90_percent_energy"] for mode in dimensions])
        ),
        "median_disjoint_training_mode_cosine": float(
            report["disjoint_training_mode_stability"][str(components)][
                "median_matched_mode_cosine"
            ]
        ),
        "disjoint_training_mode_span_overlap_fraction": float(
            report["disjoint_training_mode_stability"][str(components)][
                "mode_span_overlap_fraction"
            ]
        ),
    }


def null_summary(observed: float, samples: list[float], tail: str) -> dict[str, float]:
    values = np.asarray(samples)
    if tail == "upper":
        count = np.sum(values >= observed)
    else:
        count = np.sum(values <= observed)
    return {
        "null_mean": float(np.mean(values)),
        "null_standard_deviation": float(np.std(values)),
        "observed_minus_null_mean": float(observed - np.mean(values)),
        f"{tail}_tail_p_value": float((1 + count) / (1 + len(values))),
    }


def main() -> None:
    args = parse_args()
    component_counts = sorted(set(args.component_counts))
    truncation_ranks = sorted(set(args.truncation_ranks))
    if args.primary_components not in component_counts:
        raise ValueError("primary component count must be in component counts")
    if args.primary_truncation_rank not in truncation_ranks:
        raise ValueError("primary truncation rank must be in truncation ranks")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == args.model_revision)
    operators, metadata = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    rng = np.random.default_rng(args.seed)
    schemes = {}
    for scheme in ("head_parity", "layer_parity"):
        observed = cross_validated(operators, scheme, component_counts, truncation_ranks)
        observed_primary = primary_values(
            observed, args.primary_components, args.primary_truncation_rank
        )
        nulls = {
            name: {key: [] for key in observed_primary}
            for name in ("spectrum_matched_haar", "within_layer_side_pairing_shuffle")
        }
        for repetition in range(args.null_repetitions):
            populations = {
                "spectrum_matched_haar": spectrum_matched_haar(operators, rng),
                "within_layer_side_pairing_shuffle": pairing_shuffled(operators, rng),
            }
            for name, population in populations.items():
                report = cross_validated(
                    population,
                    scheme,
                    [args.primary_components],
                    [args.primary_truncation_rank],
                )
                values = primary_values(
                    report, args.primary_components, args.primary_truncation_rank
                )
                for key, value in values.items():
                    nulls[name][key].append(value)
            print(f"{scheme} null {repetition + 1}/{args.null_repetitions}", flush=True)
        comparisons = {}
        for name, metrics in nulls.items():
            comparisons[name] = {
                key: null_summary(
                    observed_primary[key],
                    samples,
                    "lower" if "rank" in key and "recovered" not in key else "upper",
                )
                for key, samples in metrics.items()
            }
        schemes[scheme] = {
            **observed,
            "primary": {
                "components": args.primary_components,
                "truncation_rank": args.primary_truncation_rank,
                "observed": observed_primary,
                "null_comparisons": comparisons,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "unrestricted QK population modes followed by emergent-rank audit",
                "model": manifest["model"],
                "model_revision": args.model_revision,
                "operator_metadata": metadata,
                "normalization": "unit Frobenius norm per complete QK operator",
                "method": "uncentered full-operator PCA; SVD is applied only after modes are learned",
                "component_counts": component_counts,
                "truncation_ranks": truncation_ranks,
                "schemes": schemes,
                "null_repetitions": args.null_repetitions,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
