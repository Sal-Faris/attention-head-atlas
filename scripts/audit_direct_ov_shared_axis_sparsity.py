"""Test whether the direct OV block signal reduces to shared-axis sparsity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pilot_direct_ov_restricted_maps import TOTAL_BUDGETS, unstructured_sparse_energy

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_frobenius_norm
from head_atlas.restricted_maps import (
    population_operator_bases,
    project_operator,
    shared_basis_scalar_cost,
    spectrum_matched_rotation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ov", type=Path, default=Path("artifacts/pythia-70m-deduped/step143000/ov_factors.npz")
    )
    parser.add_argument("--basis-dimension", type=int, default=128)
    parser.add_argument("--null-repetitions", type=int, default=99)
    parser.add_argument("--seed", type=int, default=17320)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/direct_ov_shared_axis_sparsity_v1.json"
        ),
    )
    return parser.parse_args()


def upper_tail(observed: float, null: list[float]) -> float:
    values = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(values >= observed)) / (1 + len(values)))


def main() -> None:
    args = parse_args()
    operators, metadata = load_factor_bundle(args.ov)
    rng = np.random.default_rng(args.seed)
    splits = []
    for training_parity in (0, 1):
        training = [item for item in operators if item.head % 2 == training_parity]
        evaluation = [item for item in operators if item.head % 2 != training_parity]
        read_basis, write_basis = population_operator_bases(training, args.basis_dimension)
        dictionary_cost = shared_basis_scalar_cost(
            operators[0].d_model, args.basis_dimension, len(evaluation)
        )
        coefficients = [
            project_operator(item, read_basis, write_basis)
            / max(factorized_frobenius_norm(item), 1e-12)
            for item in evaluation
        ]
        observed = [
            float(
                np.mean(
                    [
                        unstructured_sparse_energy(values, budget - dictionary_cost)
                        for values in coefficients
                    ]
                )
            )
            for budget in TOTAL_BUDGETS
        ]
        null_samples = [[] for _ in TOTAL_BUDGETS]
        for repetition in range(args.null_repetitions):
            rotated = [spectrum_matched_rotation(values, rng) for values in coefficients]
            for budget_index, budget in enumerate(TOTAL_BUDGETS):
                null_samples[budget_index].append(
                    float(
                        np.mean(
                            [
                                unstructured_sparse_energy(
                                    values, budget - dictionary_cost
                                )
                                for values in rotated
                            ]
                        )
                    )
                )
            print(
                f"parity {training_parity} null {repetition + 1}/{args.null_repetitions}",
                flush=True,
            )
        splits.append(
            {
                "training_head_parity": training_parity,
                "dictionary_scalar_cost_per_evaluation_head": dictionary_cost,
                "budgets": [
                    {
                        "total_scalar_budget": budget,
                        "observed_sparse_energy_fraction": observed[index],
                        "rotation_null_mean": float(np.mean(null_samples[index])),
                        "rotation_null_samples": null_samples[index],
                        "upper_tail_p": upper_tail(observed[index], null_samples[index]),
                    }
                    for index, budget in enumerate(TOTAL_BUDGETS)
                ],
            }
        )
    report = {
        "status": "shared-axis sparsity audit for direct OV restricted-map pilot",
        "model": metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": metadata.get("revision", "step143000"),
        "basis_dimension": args.basis_dimension,
        "definition": (
            "retain individually largest coefficients in a population basis learned "
            "from complementary heads; compare with projected-spectrum-matched rotations"
        ),
        "splits": splits,
        "null_repetitions": args.null_repetitions,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved sparsity audit to {args.output}")


if __name__ == "__main__":
    main()
