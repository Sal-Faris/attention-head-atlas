"""Small real pilot of direct variable-rank restricted maps in OV weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_frobenius_norm, factorized_singular_components
from head_atlas.restricted_maps import (
    fit_restricted_map_path,
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
    parser.add_argument("--basis-dimension", type=int, default=64)
    parser.add_argument("--training-head-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--maximum-blocks", type=int, default=6)
    parser.add_argument("--null-repetitions", type=int, default=7)
    parser.add_argument("--synthetic-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=14142)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/direct_ov_restricted_map_pilot_v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/direct_ov_restricted_map_pilot_v1.png"),
    )
    return parser.parse_args()


PENALTIES = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
TOTAL_BUDGETS = (4_000.0, 8_000.0, 16_000.0, 32_000.0)
SYNTHETIC_BUDGETS = (100.0, 250.0, 500.0, 1_000.0)


def fit_frontier(coefficients: np.ndarray, maximum_blocks: int) -> list[dict[str, object]]:
    """Fit the block model over a fixed complexity grid."""

    frontier: list[dict[str, object]] = [{"scalar_cost": 0.0, "captured_energy": 0.0, "blocks": []}]
    support_sizes = tuple(size for size in (4, 8, 16, 32, 64) if size <= len(coefficients))
    for penalty in PENALTIES:
        fits = fit_restricted_map_path(
            coefficients,
            complexity_penalty=penalty,
            support_sizes=support_sizes,
            maximum_blocks=maximum_blocks,
        )
        for fit in fits[1:]:
            frontier.append(
                {
                    "penalty": penalty,
                    "scalar_cost": fit.scalar_cost,
                    "captured_energy": fit.captured_energy,
                    "blocks": [
                        {
                            "read_dimension": len(block.read_indices),
                            "write_dimension": len(block.write_indices),
                            "core_rank": block.core_rank,
                            "scalar_cost": block.scalar_cost,
                            "energy_gain": block.energy_gain,
                        }
                        for block in fit.blocks
                    ],
                }
            )
    return frontier


def select_at_budget(frontier: list[dict[str, object]], available_cost: float) -> dict[str, object]:
    eligible = [item for item in frontier if item["scalar_cost"] <= max(available_cost, 0.0)]
    return max(eligible, key=lambda item: (item["captured_energy"], -item["scalar_cost"]))


def unstructured_sparse_energy(
    coefficients: np.ndarray,
    available_cost: float,
    *,
    float_bits: int = 16,
) -> float:
    dimension = len(coefficients)
    scalar_cost = 1.0 + 2.0 * np.log2(dimension) / float_bits
    retained = min(int(max(available_cost, 0.0) // scalar_cost), coefficients.size)
    if retained == 0:
        return 0.0
    energy = np.sort(np.ravel(coefficients**2))[::-1]
    return float(np.sum(energy[:retained]))


def projected_dense_low_rank_energy(coefficients: np.ndarray, available_cost: float) -> float:
    dimension = len(coefficients)
    singular_values = np.linalg.svd(coefficients, compute_uv=False)
    selected = 0
    for rank in range(1, dimension + 1):
        cost = rank * (2 * dimension - rank) + rank + 1
        if cost <= max(available_cost, 0.0):
            selected = rank
    return float(np.sum(singular_values[:selected] ** 2))


def full_svd_energy(operator: object, budget: float) -> float:
    _, singular_values, _ = factorized_singular_components(operator)
    total = max(factorized_frobenius_norm(operator) ** 2, 1e-20)
    width = operator.d_model
    selected = 0
    for rank in range(1, len(singular_values) + 1):
        cost = rank * (2 * width - rank) + rank
        if cost <= budget:
            selected = rank
    return float(np.sum(singular_values[:selected] ** 2) / total)


def synthetic_matrix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    values = np.zeros((dimension, dimension), dtype=np.float64)
    permutation_rows = rng.permutation(dimension)
    permutation_columns = rng.permutation(dimension)
    specifications = ((8, 8, 2), (16, 16, 3), (8, 16, 2))
    row_start = 0
    column_start = 0
    for read_size, write_size, rank in specifications:
        rows = permutation_rows[row_start : row_start + read_size]
        columns = permutation_columns[column_start : column_start + write_size]
        core = rng.standard_normal((read_size, rank)) @ rng.standard_normal((rank, write_size))
        values[np.ix_(rows, columns)] += core
        row_start += read_size
        column_start += write_size
    values += 0.015 * np.std(values) * rng.standard_normal(values.shape)
    return values / np.linalg.norm(values)


def run_synthetic_calibration(
    rng: np.random.Generator,
    *,
    dimension: int,
    samples: int,
    maximum_blocks: int,
) -> dict[str, object]:
    planted = [[] for _ in SYNTHETIC_BUDGETS]
    rotated = [[] for _ in SYNTHETIC_BUDGETS]
    dense = [[] for _ in SYNTHETIC_BUDGETS]
    for _ in range(samples):
        values = synthetic_matrix(rng, dimension)
        null = spectrum_matched_rotation(values, rng)
        planted_frontier = fit_frontier(values, maximum_blocks)
        null_frontier = fit_frontier(null, maximum_blocks)
        for index, budget in enumerate(SYNTHETIC_BUDGETS):
            planted[index].append(select_at_budget(planted_frontier, budget)["captured_energy"])
            rotated[index].append(select_at_budget(null_frontier, budget)["captured_energy"])
            dense[index].append(projected_dense_low_rank_energy(values, budget))
    return {
        "definition": "three variable-size planted restricted maps plus small dense noise",
        "budgets": list(SYNTHETIC_BUDGETS),
        "planted_mean_energy": [float(np.mean(values)) for values in planted],
        "spectrum_matched_rotation_mean_energy": [float(np.mean(values)) for values in rotated],
        "dense_low_rank_mean_energy": [float(np.mean(values)) for values in dense],
        "samples": samples,
    }


def summarize_real_heads(
    operators: list[object],
    read_basis: np.ndarray,
    write_basis: np.ndarray,
    *,
    dictionary_cost: float,
    maximum_blocks: int,
    null_repetitions: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    method_values = {
        name: [[] for _ in TOTAL_BUDGETS]
        for name in (
            "restricted_blocks",
            "projected_dense_low_rank",
            "unstructured_sparse",
            "full_svd",
        )
    }
    null_values = [[] for _ in TOTAL_BUDGETS]
    per_head = []
    for operator_index, operator in enumerate(operators):
        total_energy = max(factorized_frobenius_norm(operator) ** 2, 1e-20)
        coefficients = project_operator(operator, read_basis, write_basis) / np.sqrt(total_energy)
        frontier = fit_frontier(coefficients, maximum_blocks)
        head_record = {"layer": operator.layer, "head": operator.head, "budgets": []}
        null_frontiers = [
            fit_frontier(spectrum_matched_rotation(coefficients, rng), maximum_blocks)
            for _ in range(null_repetitions)
        ]
        for budget_index, budget in enumerate(TOTAL_BUDGETS):
            available = budget - dictionary_cost
            selected = select_at_budget(frontier, available)
            block_energy = float(selected["captured_energy"])
            dense_energy = projected_dense_low_rank_energy(coefficients, available)
            sparse_energy = unstructured_sparse_energy(coefficients, available)
            svd_energy = full_svd_energy(operator, budget)
            null_samples = [
                float(select_at_budget(null_frontier, available)["captured_energy"])
                for null_frontier in null_frontiers
            ]
            method_values["restricted_blocks"][budget_index].append(block_energy)
            method_values["projected_dense_low_rank"][budget_index].append(dense_energy)
            method_values["unstructured_sparse"][budget_index].append(sparse_energy)
            method_values["full_svd"][budget_index].append(svd_energy)
            null_values[budget_index].extend(null_samples)
            head_record["budgets"].append(
                {
                    "total_scalar_budget": budget,
                    "restricted_block_energy_fraction": block_energy,
                    "projected_dense_low_rank_energy_fraction": dense_energy,
                    "unstructured_sparse_energy_fraction": sparse_energy,
                    "full_svd_energy_fraction": svd_energy,
                    "null_mean_energy_fraction": float(np.mean(null_samples)),
                    "null_samples": null_samples,
                    "selected_scalar_cost_excluding_dictionary": selected["scalar_cost"],
                    "selected_blocks": selected["blocks"],
                }
            )
        per_head.append(head_record)
        print(
            f"real head {operator_index + 1}/{len(operators)}: L{operator.layer}H{operator.head}",
            flush=True,
        )
    population = {
        "total_scalar_budgets": list(TOTAL_BUDGETS),
        "dictionary_scalar_cost_per_evaluation_head": dictionary_cost,
        "mean_energy_fraction": {
            name: [float(np.mean(values)) for values in budget_values]
            for name, budget_values in method_values.items()
        },
        "spectrum_matched_rotation_mean_energy_fraction": [
            float(np.mean(values)) for values in null_values
        ],
        "restricted_minus_rotation": [
            float(np.mean(method_values["restricted_blocks"][index]) - np.mean(null_values[index]))
            for index in range(len(TOTAL_BUDGETS))
        ],
    }
    return population, per_head


def plot_report(report: dict[str, object], output: Path) -> None:
    synthetic = report["synthetic_calibration"]
    population = report["held_out_real_population"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    axes[0].plot(
        synthetic["budgets"], synthetic["planted_mean_energy"], marker="o", label="Planted blocks"
    )
    axes[0].plot(
        synthetic["budgets"],
        synthetic["spectrum_matched_rotation_mean_energy"],
        marker="o",
        label="Spectrum-matched rotation",
    )
    axes[0].plot(
        synthetic["budgets"],
        synthetic["dense_low_rank_mean_energy"],
        marker="o",
        label="One dense low-rank map",
    )
    axes[0].set_xlabel("Scalar-equivalent budget")
    axes[0].set_ylabel("Explained operator energy")
    axes[0].set_title("A  Can the method recover planted blocks?", loc="left")
    axes[0].legend()

    labels = {
        "restricted_blocks": "Restricted blocks",
        "projected_dense_low_rank": "One projected low-rank map",
        "unstructured_sparse": "Unstructured sparse coefficients",
        "full_svd": "Full-space truncated SVD",
    }
    for name, values in population["mean_energy_fraction"].items():
        axes[1].plot(population["total_scalar_budgets"], values, marker="o", label=labels[name])
    axes[1].plot(
        population["total_scalar_budgets"],
        population["spectrum_matched_rotation_mean_energy_fraction"],
        marker="o",
        linestyle="--",
        label="Rotated-null blocks",
    )
    axes[1].set_xlabel("Total scalar-equivalent budget per head")
    axes[1].set_ylabel("Mean explained OV energy")
    axes[1].set_title("B  Held-out real OV rate-distortion", loc="left")
    axes[1].legend(fontsize=8)

    primary_index = 1
    head_differences = []
    layers = []
    for head in report["held_out_heads"]:
        record = head["budgets"][primary_index]
        head_differences.append(
            record["restricted_block_energy_fraction"] - record["null_mean_energy_fraction"]
        )
        layers.append(head["layer"])
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].scatter(layers, head_differences)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Real minus rotated-null energy")
    axes[2].set_title("C  Per-head advantage at budget 8,000", loc="left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    operators, metadata = load_factor_bundle(args.ov)
    training = [
        operator for operator in operators if operator.head % 2 == args.training_head_parity
    ]
    evaluation = [
        operator for operator in operators if operator.head % 2 != args.training_head_parity
    ]
    read_basis, write_basis = population_operator_bases(training, args.basis_dimension)
    dictionary_cost = shared_basis_scalar_cost(
        operators[0].d_model, args.basis_dimension, len(evaluation)
    )
    rng = np.random.default_rng(args.seed)
    synthetic = run_synthetic_calibration(
        rng,
        dimension=args.basis_dimension,
        samples=args.synthetic_samples,
        maximum_blocks=args.maximum_blocks,
    )
    population, per_head = summarize_real_heads(
        evaluation,
        read_basis,
        write_basis,
        dictionary_cost=dictionary_cost,
        maximum_blocks=args.maximum_blocks,
        null_repetitions=args.null_repetitions,
        rng=rng,
    )
    report = {
        "status": "direct variable-rank restricted-map pilot",
        "model": metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": metadata.get("revision", "step143000"),
        "operator_kind": "OV",
        "discovery": "weights only; no tokens, prompts, activations, or semantic labels",
        "training_heads": [{"layer": item.layer, "head": item.head} for item in training],
        "evaluation_heads": [{"layer": item.layer, "head": item.head} for item in evaluation],
        "training_head_parity": args.training_head_parity,
        "basis_dimension": args.basis_dimension,
        "dictionary_learning": (
            "top eigenspaces of summed M M^T and M^T M over even-numbered training heads"
        ),
        "restricted_map": (
            "sum of variable read-support, write-support, and learned low-rank core blocks "
            "plus an explicit residual"
        ),
        "complexity": (
            "shared Stiefel basis degrees amortized conservatively over held-out heads; "
            "per-block support code plus arbitrary low-rank core parameters"
        ),
        "synthetic_calibration": synthetic,
        "held_out_real_population": population,
        "held_out_heads": per_head,
        "null_repetitions_per_head": args.null_repetitions,
        "penalties": list(PENALTIES),
        "maximum_blocks": args.maximum_blocks,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved result to {args.output}")
    print(f"saved figure to {args.figure}")


if __name__ == "__main__":
    main()
