"""Test whether shared read/write structure compresses entirely held-out heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import (
    FactorizedHeadOperator,
    factorized_frobenius_norm,
    factorized_singular_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/pythia-70m-deduped-pilot.json")
    )
    parser.add_argument("--model-revision", default="step143000")
    parser.add_argument("--support-dimensions", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--core-components", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--null-repetitions", type=int, default=19)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/shared_operator_compression_v1.json"),
    )
    return parser.parse_args()


def shared_basis(operators: list[FactorizedHeadOperator], indices: np.ndarray, side: str, maximum: int) -> np.ndarray:
    width = operators[0].d_model
    covariance = np.zeros((width, width), dtype=np.float64)
    for index in indices:
        operator = operators[int(index)]
        norm = factorized_frobenius_norm(operator)
        first = np.asarray(operator.left if side == "left" else operator.right, dtype=np.float64)
        second = np.asarray(operator.right if side == "left" else operator.left, dtype=np.float64)
        covariance += first @ (second.T @ second) @ first.T / max(norm**2, 1e-24)
    values, vectors = np.linalg.eigh(covariance)
    return vectors[:, np.argsort(values)[-maximum:][::-1]]


def normalized_matrix(operator: FactorizedHeadOperator) -> np.ndarray:
    matrix = operator.materialize(dtype=np.float64)
    return matrix / max(np.linalg.norm(matrix), 1e-12)


def fold_metrics(
    operators: list[FactorizedHeadOperator],
    train: np.ndarray,
    test: np.ndarray,
    support_dimensions: list[int],
    core_components: list[int],
) -> dict[str, dict[str, float]]:
    maximum = max(support_dimensions)
    left_full = shared_basis(operators, train, "left", maximum)
    right_full = shared_basis(operators, train, "right", maximum)
    train_matrices = [normalized_matrix(operators[int(index)]) for index in train]
    test_matrices = [normalized_matrix(operators[int(index)]) for index in test]
    result = {}
    for dimension in support_dimensions:
        left = left_full[:, :dimension]
        right = right_full[:, :dimension]
        train_cores = np.stack([left.T @ matrix @ right for matrix in train_matrices])
        test_cores = np.stack([left.T @ matrix @ right for matrix in test_matrices])
        train_vectors = train_cores.reshape(len(train_cores), -1)
        test_vectors = test_cores.reshape(len(test_cores), -1)
        mean = np.mean(train_vectors, axis=0, keepdims=True)
        _, _, directions = np.linalg.svd(train_vectors - mean, full_matrices=False)
        captured = np.sum(test_vectors**2, axis=1)
        metrics = {
            "shared_support_variance_recovered": float(np.mean(captured)),
            "mean_core_full_operator_variance_recovered": float(
                np.mean(captured - np.sum((test_vectors - mean) ** 2, axis=1))
            ),
        }
        for components in core_components:
            count = min(components, len(directions))
            basis = directions[:count]
            reconstruction = mean + (test_vectors - mean) @ basis.T @ basis
            core_error = np.sum((test_vectors - reconstruction) ** 2, axis=1)
            metrics[f"pca_{components}_full_operator_variance_recovered"] = float(
                np.mean(captured - core_error)
            )
        result[str(dimension)] = metrics
    return result


def cross_validated_metrics(
    operators: list[FactorizedHeadOperator],
    support_dimensions: list[int],
    core_components: list[int],
) -> dict[str, dict[str, float]]:
    heads = np.asarray([operator.head for operator in operators])
    folds = []
    for parity in (0, 1):
        test = np.flatnonzero(heads % 2 == parity)
        train = np.flatnonzero(heads % 2 != parity)
        folds.append(fold_metrics(operators, train, test, support_dimensions, core_components))
    return {
        dimension: {
            metric: float(np.mean([fold[dimension][metric] for fold in folds]))
            for metric in folds[0][dimension]
        }
        for dimension in folds[0]
    }


def spectrum_matched_haar(
    operators: list[FactorizedHeadOperator], rng: np.random.Generator
) -> list[FactorizedHeadOperator]:
    result = []
    for operator in operators:
        _, spectrum, _ = factorized_singular_components(operator)
        left, _ = np.linalg.qr(rng.standard_normal(operator.left.shape), mode="reduced")
        right, _ = np.linalg.qr(rng.standard_normal(operator.right.shape), mode="reduced")
        result.append(
            FactorizedHeadOperator(
                operator.layer,
                operator.head,
                operator.kind,
                left * spectrum,
                right,
            )
        )
    return result


def pairing_shuffled(
    operators: list[FactorizedHeadOperator], rng: np.random.Generator
) -> list[FactorizedHeadOperator]:
    decomposed = [factorized_singular_components(operator) for operator in operators]
    result = []
    for layer in sorted({operator.layer for operator in operators}):
        indices = [index for index, operator in enumerate(operators) if operator.layer == layer]
        permutation = rng.permutation(indices)
        for index, right_index in zip(indices, permutation, strict=True):
            operator = operators[index]
            left, spectrum, _ = decomposed[index]
            right = decomposed[int(right_index)][2]
            result.append(
                FactorizedHeadOperator(
                    operator.layer,
                    operator.head,
                    operator.kind,
                    left * spectrum,
                    right,
                )
            )
    result.sort(key=lambda operator: (operator.layer, operator.head))
    return result


def primary_value(report: dict[str, dict[str, float]], dimension: int, components: int) -> float:
    return report[str(dimension)][f"pca_{components}_full_operator_variance_recovered"]


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == args.model_revision)
    support_dimensions = sorted(set(args.support_dimensions))
    core_components = sorted(set(args.core_components))
    primary_dimension = 64
    primary_components = 16
    if primary_dimension not in support_dimensions or primary_components not in core_components:
        raise ValueError("the frozen primary comparison requires support=64 and core PCA=16")
    rng = np.random.default_rng(args.seed)
    views = {}
    for kind in ("QK", "OV"):
        operators, _ = load_factor_bundle(Path(record["factors"][kind]["path"]))
        observed = cross_validated_metrics(operators, support_dimensions, core_components)
        nulls = {"spectrum_matched_haar": [], "within_layer_side_pairing_shuffle": []}
        for repetition in range(args.null_repetitions):
            nulls["spectrum_matched_haar"].append(
                primary_value(
                    cross_validated_metrics(
                        spectrum_matched_haar(operators, rng),
                        [primary_dimension],
                        [primary_components],
                    ),
                    primary_dimension,
                    primary_components,
                )
            )
            nulls["within_layer_side_pairing_shuffle"].append(
                primary_value(
                    cross_validated_metrics(
                        pairing_shuffled(operators, rng),
                        [primary_dimension],
                        [primary_components],
                    ),
                    primary_dimension,
                    primary_components,
                )
            )
            print(f"{kind} null {repetition + 1}/{args.null_repetitions}", flush=True)
        observed_primary = primary_value(observed, primary_dimension, primary_components)
        comparisons = {}
        for name, samples in nulls.items():
            values = np.asarray(samples)
            comparisons[name] = {
                "null_mean": float(np.mean(values)),
                "null_standard_deviation": float(np.std(values)),
                "observed_minus_null_mean": float(observed_primary - np.mean(values)),
                "upper_tail_p_value": float(
                    (1 + np.sum(values >= observed_primary)) / (1 + len(values))
                ),
            }
        views[kind] = {
            "held_out_head_metrics": observed,
            "primary": {
                "support_dimension": primary_dimension,
                "core_pca_components": primary_components,
                "observed_full_operator_variance_recovered": observed_primary,
                "null_comparisons": comparisons,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "leave-complete-heads-out shared read/write support and coupling compression",
                "splits": "two parity folds holding out four of eight heads in every layer",
                "normalization": "each operator has unit Frobenius norm",
                "views": views,
                "null_repetitions": args.null_repetitions,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
