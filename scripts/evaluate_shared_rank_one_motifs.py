"""Test whether reusable rank-one QK channels compress unseen heads and layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_shared_operator_compression import (
    normalized_matrix,
    pairing_shuffled,
    shared_basis,
    spectrum_matched_haar,
)
from scipy.optimize import linear_sum_assignment

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import FactorizedHeadOperator
from head_atlas.tensor_motifs import (
    encode_rank_one_motifs,
    fit_rank_one_motifs,
    reconstruct_rank_one_motifs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/pythia-70m-deduped-pilot.json")
    )
    parser.add_argument("--model-revision", default="step143000")
    parser.add_argument("--support-dimension", type=int, default=64)
    parser.add_argument("--motif-counts", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--primary-motif-count", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--null-repetitions", type=int, default=19)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--stability-seeds", type=int, nargs="+", default=[23, 71, 131])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/shared_rank_one_qk_motifs_v1.json"),
    )
    parser.add_argument(
        "--motif-output",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/shared_rank_one_qk_motifs_v1.npz"),
    )
    return parser.parse_args()


def split_folds(operators: list[FactorizedHeadOperator], scheme: str) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(
        [operator.head if scheme == "head_parity" else operator.layer for operator in operators]
    )
    return [
        (np.flatnonzero(labels % 2 != parity), np.flatnonzero(labels % 2 == parity))
        for parity in (0, 1)
    ]


def dense_pca_reconstruction(train: np.ndarray, test: np.ndarray, components: int) -> np.ndarray:
    train_vectors = train.reshape(len(train), -1)
    test_vectors = test.reshape(len(test), -1)
    _, _, directions = np.linalg.svd(train_vectors, full_matrices=False)
    basis = directions[: min(components, len(directions))]
    return (test_vectors @ basis.T @ basis).reshape(test.shape)


def recovered_full_variance(test: np.ndarray, reconstruction: np.ndarray) -> float:
    captured = np.sum(test**2, axis=(1, 2))
    core_error = np.sum((test - reconstruction) ** 2, axis=(1, 2))
    return float(np.mean(captured - core_error))


def evaluate_fold(
    operators: list[FactorizedHeadOperator],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    dimension: int,
    motif_counts: list[int],
    iterations: int,
    restarts: int,
    seed: int,
) -> dict[str, object]:
    left = shared_basis(operators, train_indices, "left", dimension)
    right = shared_basis(operators, train_indices, "right", dimension)
    train = np.stack(
        [left.T @ normalized_matrix(operators[int(index)]) @ right for index in train_indices]
    )
    test = np.stack(
        [left.T @ normalized_matrix(operators[int(index)]) @ right for index in test_indices]
    )
    captured = float(np.mean(np.sum(test**2, axis=(1, 2))))
    by_count = {}
    for offset, count in enumerate(motif_counts):
        motifs = fit_rank_one_motifs(
            train,
            count,
            seed=seed + 1009 * offset,
            iterations=iterations,
            restarts=restarts,
        )
        cp_reconstruction = reconstruct_rank_one_motifs(test, motifs)
        dense_reconstruction = dense_pca_reconstruction(train, test, count)
        matched_components = max(1, int(np.floor(2 * dimension * count / dimension**2)))
        matched_reconstruction = dense_pca_reconstruction(train, test, matched_components)
        cp_recovered = recovered_full_variance(test, cp_reconstruction)
        by_count[str(count)] = {
            "rank_one_motif_full_variance_recovered": cp_recovered,
            "fraction_of_shared_support_recovered": cp_recovered / max(captured, 1e-12),
            "dense_pca_same_component_count_full_variance_recovered": recovered_full_variance(
                test, dense_reconstruction
            ),
            "parameter_matched_dense_pca_components": matched_components,
            "parameter_matched_dense_pca_full_variance_recovered": recovered_full_variance(
                test, matched_reconstruction
            ),
            "motif_dictionary_parameters": 2 * dimension * count,
            "dense_pca_dictionary_parameters": dimension**2 * count,
            "training_relative_squared_error": motifs.training_loss,
            "iterations_completed": motifs.iterations,
        }
    return {"shared_support_variance_recovered": captured, "by_motif_count": by_count}


def cross_validated(
    operators: list[FactorizedHeadOperator],
    scheme: str,
    dimension: int,
    motif_counts: list[int],
    iterations: int,
    restarts: int,
    seed: int,
) -> dict[str, object]:
    folds = [
        evaluate_fold(
            operators,
            train,
            test,
            dimension,
            motif_counts,
            iterations,
            restarts,
            seed + fold_index * 100_003,
        )
        for fold_index, (train, test) in enumerate(split_folds(operators, scheme))
    ]
    counts = [str(count) for count in motif_counts]
    return {
        "shared_support_variance_recovered": float(
            np.mean([fold["shared_support_variance_recovered"] for fold in folds])
        ),
        "by_motif_count": {
            count: {
                key: float(np.mean([fold["by_motif_count"][count][key] for fold in folds]))
                if isinstance(folds[0]["by_motif_count"][count][key], float)
                else folds[0]["by_motif_count"][count][key]
                for key in folds[0]["by_motif_count"][count]
            }
            for count in counts
        },
    }


def primary_value(report: dict[str, object], count: int) -> float:
    return float(report["by_motif_count"][str(count)]["rank_one_motif_full_variance_recovered"])


def summarize_null(observed: float, samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples)
    return {
        "null_mean": float(np.mean(values)),
        "null_standard_deviation": float(np.std(values)),
        "observed_minus_null_mean": float(observed - np.mean(values)),
        "upper_tail_p_value": float((1 + np.sum(values >= observed)) / (1 + len(values))),
    }


def full_population_motifs(
    operators: list[FactorizedHeadOperator],
    dimension: int,
    count: int,
    iterations: int,
    restarts: int,
    seeds: list[int],
    output: Path,
) -> dict[str, object]:
    indices = np.arange(len(operators))
    support_left = shared_basis(operators, indices, "left", dimension)
    support_right = shared_basis(operators, indices, "right", dimension)
    matrices = [normalized_matrix(operator) for operator in operators]
    cores = np.stack(
        [support_left.T @ matrix @ support_right for matrix in matrices]
    )
    fits = [
        fit_rank_one_motifs(
            cores,
            count,
            seed=seed,
            iterations=iterations,
            restarts=restarts,
        )
        for seed in seeds
    ]
    reference = fits[0]
    stability = []
    for seed, candidate in zip(seeds[1:], fits[1:], strict=True):
        similarity = np.abs(reference.left.T @ candidate.left) * np.abs(
            reference.right.T @ candidate.right
        )
        rows, columns = linear_sum_assignment(-similarity)
        matched = similarity[rows, columns]
        stability.append(
            {
                "seed": seed,
                "mean_matched_atom_cosine": float(np.mean(matched)),
                "median_matched_atom_cosine": float(np.median(matched)),
                "atoms_above_0.8": int(np.sum(matched >= 0.8)),
                "atoms_above_0.5": int(np.sum(matched >= 0.5)),
            }
        )

    coefficients = encode_rank_one_motifs(cores, reference)
    absolute = np.abs(coefficients)
    thresholds = 0.25 * np.max(absolute, axis=0, keepdims=True)
    active = absolute >= thresholds
    layers = np.asarray([operator.layer for operator in operators])
    active_layers = [
        len(set(layers[active[:, index]].tolist())) for index in range(count)
    ]
    effective_heads = (np.sum(absolute, axis=0) ** 2) / np.maximum(
        np.sum(absolute**2, axis=0), 1e-12
    )
    residual_left = support_left @ reference.left
    residual_right = support_right @ reference.right
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        left_motifs=residual_left.astype(np.float32),
        right_motifs=residual_right.astype(np.float32),
        coefficients=coefficients.astype(np.float32),
        layers=layers,
        heads=np.asarray([operator.head for operator in operators]),
        kind=np.asarray("QK"),
        normalization=np.asarray("unit Frobenius norm per operator; unit-norm motif sides"),
    )
    return {
        "stability_against_reference_seed": seeds[0],
        "stability": stability,
        "activity_threshold": "absolute coefficient >= 25% of that motif's population maximum",
        "active_head_count_per_motif": np.sum(active, axis=0).astype(int).tolist(),
        "active_layer_count_per_motif": active_layers,
        "effective_head_count_per_motif": effective_heads.tolist(),
        "median_active_heads": float(np.median(np.sum(active, axis=0))),
        "median_active_layers": float(np.median(active_layers)),
        "median_effective_heads": float(np.median(effective_heads)),
        "artifact": str(output),
    }


def main() -> None:
    args = parse_args()
    counts = sorted(set(args.motif_counts))
    if args.primary_motif_count not in counts:
        raise ValueError("primary motif count must be included in motif counts")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == args.model_revision)
    operators, metadata = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    rng = np.random.default_rng(args.seed)
    reports = {}
    for scheme_index, scheme in enumerate(("head_parity", "layer_parity")):
        observed = cross_validated(
            operators,
            scheme,
            args.support_dimension,
            counts,
            args.iterations,
            args.restarts,
            args.seed + scheme_index * 1_000_003,
        )
        null_samples = {"spectrum_matched_haar": [], "within_layer_side_pairing_shuffle": []}
        for repetition in range(args.null_repetitions):
            for null_index, (name, population) in enumerate(
                (
                    ("spectrum_matched_haar", spectrum_matched_haar(operators, rng)),
                    ("within_layer_side_pairing_shuffle", pairing_shuffled(operators, rng)),
                )
            ):
                null_report = cross_validated(
                    population,
                    scheme,
                    args.support_dimension,
                    [args.primary_motif_count],
                    args.iterations,
                    1,
                    args.seed + 10_000_019 * (repetition + 1) + null_index,
                )
                null_samples[name].append(primary_value(null_report, args.primary_motif_count))
            print(
                f"{scheme} null {repetition + 1}/{args.null_repetitions}", flush=True
            )
        observed_primary = primary_value(observed, args.primary_motif_count)
        reports[scheme] = {
            **observed,
            "primary": {
                "motif_count": args.primary_motif_count,
                "observed_full_variance_recovered": observed_primary,
                "null_comparisons": {
                    name: summarize_null(observed_primary, samples)
                    for name, samples in null_samples.items()
                },
            },
        }
    heldout_robustness = {}
    for scheme_index, scheme in enumerate(("head_parity", "layer_parity")):
        values = [
            {
                "seed": args.seed,
                "full_variance_recovered": primary_value(
                    reports[scheme], args.primary_motif_count
                ),
            }
        ]
        for seed in args.stability_seeds:
            if seed == args.seed:
                continue
            repeated = cross_validated(
                operators,
                scheme,
                args.support_dimension,
                [args.primary_motif_count],
                args.iterations,
                args.restarts,
                seed + scheme_index * 1_000_003,
            )
            values.append(
                {
                    "seed": seed,
                    "full_variance_recovered": primary_value(
                        repeated, args.primary_motif_count
                    ),
                }
            )
        heldout_robustness[scheme] = values
    motif_analysis = full_population_motifs(
        operators,
        args.support_dimension,
        args.primary_motif_count,
        args.iterations,
        args.restarts,
        args.stability_seeds,
        args.motif_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "shared rank-one QK motif compression of unseen heads and layers",
                "model": manifest["model"],
                "model_revision": args.model_revision,
                "operator_metadata": metadata,
                "normalization": "unit Frobenius norm per complete QK operator",
                "support_dimension": args.support_dimension,
                "algorithm": {
                    "motif_counts": counts,
                    "primary_motif_count": args.primary_motif_count,
                    "als_iterations": args.iterations,
                    "observed_restarts": args.restarts,
                    "null_restarts": 1,
                    "stability_seeds": args.stability_seeds,
                },
                "reports": reports,
                "heldout_initialization_robustness": heldout_robustness,
                "full_population_motif_analysis": motif_analysis,
                "null_repetitions": args.null_repetitions,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
