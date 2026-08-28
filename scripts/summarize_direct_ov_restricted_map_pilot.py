"""Consolidate reciprocal direct OV restricted-map pilot results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_frobenius_norm
from head_atlas.restricted_maps import (
    population_operator_bases,
    project_operator,
    shared_basis_scalar_cost,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/direct_ov_restricted_map_pilot_dim128_v1.json"
        ),
    )
    parser.add_argument(
        "--reciprocal",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/"
            "direct_ov_restricted_map_pilot_dim128_reciprocal_v1.json"
        ),
    )
    parser.add_argument(
        "--sparsity-audit",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/direct_ov_shared_axis_sparsity_v1.json"
        ),
    )
    parser.add_argument(
        "--ov", type=Path, default=Path("artifacts/pythia-70m-deduped/step143000/ov_factors.npz")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/direct_ov_restricted_map_summary_v1.json"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/direct_ov_restricted_map_summary_v1.png"
        ),
    )
    return parser.parse_args()


def population_null(report: dict[str, object], budget_index: int) -> list[float]:
    repetitions = report["null_repetitions_per_head"]
    return [
        float(
            np.mean(
                [head["budgets"][budget_index]["null_samples"][index] for head in report["held_out_heads"]]
            )
        )
        for index in range(repetitions)
    ]


def upper_tail(observed: float, null: list[float]) -> float:
    values = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(values >= observed)) / (1 + len(values)))


def split_summary(
    report: dict[str, object], sparsity: dict[str, object], budget_index: int
) -> dict[str, object]:
    population = report["held_out_real_population"]
    block = population["mean_energy_fraction"]["restricted_blocks"][budget_index]
    block_null = population_null(report, budget_index)
    sparse_record = sparsity["budgets"][budget_index]
    selected_blocks = [
        block_record
        for head in report["held_out_heads"]
        for block_record in head["budgets"][budget_index]["selected_blocks"]
    ]
    return {
        "training_head_parity": report["training_head_parity"],
        "total_scalar_budget": population["total_scalar_budgets"][budget_index],
        "restricted_block_energy_fraction": block,
        "restricted_block_rotation_null_mean": float(np.mean(block_null)),
        "restricted_block_rotation_upper_tail_p": upper_tail(block, block_null),
        "unstructured_sparse_energy_fraction": sparse_record[
            "observed_sparse_energy_fraction"
        ],
        "unstructured_sparse_rotation_null_mean": sparse_record["rotation_null_mean"],
        "unstructured_sparse_rotation_upper_tail_p": sparse_record["upper_tail_p"],
        "projected_dense_low_rank_energy_fraction": population["mean_energy_fraction"][
            "projected_dense_low_rank"
        ][budget_index],
        "full_svd_energy_fraction": population["mean_energy_fraction"]["full_svd"][
            budget_index
        ],
        "selected_block_count": len(selected_blocks),
        "fraction_blocks_with_read_or_write_dimension_64": float(
            np.mean(
                [
                    block_record["read_dimension"] == 64
                    or block_record["write_dimension"] == 64
                    for block_record in selected_blocks
                ]
            )
        ),
    }


def plot_report(summary: dict[str, object], output: Path) -> None:
    synthetic = summary["synthetic_calibration"]
    splits = summary["reciprocal_splits_at_budget_8000"]
    sensitivity = summary["population_basis_sensitivity"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    axes[0].plot(synthetic["budgets"], synthetic["planted_mean_energy"], marker="o", label="Planted")
    axes[0].plot(
        synthetic["budgets"],
        synthetic["spectrum_matched_rotation_mean_energy"],
        marker="o",
        label="Rotated null",
    )
    axes[0].plot(
        synthetic["budgets"],
        synthetic["dense_low_rank_mean_energy"],
        marker="o",
        label="One dense map",
    )
    axes[0].set_xlabel("Scalar-equivalent budget")
    axes[0].set_ylabel("Explained energy")
    axes[0].set_title("A  Planted-block calibration", loc="left")
    axes[0].legend()

    labels = ("Block", "Block null", "Sparse", "Sparse null", "Dense map", "Full SVD")
    fields = (
        "restricted_block_energy_fraction",
        "restricted_block_rotation_null_mean",
        "unstructured_sparse_energy_fraction",
        "unstructured_sparse_rotation_null_mean",
        "projected_dense_low_rank_energy_fraction",
        "full_svd_energy_fraction",
    )
    positions = np.arange(len(labels))
    width = 0.36
    for index, split in enumerate(splits):
        axes[1].bar(
            positions + (index - 0.5) * width,
            [split[field] for field in fields],
            width,
            label=f"Training parity {index}",
        )
    axes[1].set_xticks(positions, labels, rotation=25, ha="right")
    axes[1].set_ylabel("Held-out OV energy")
    axes[1].set_title("B  What explains the real signal?", loc="left")
    axes[1].legend()

    axes[2].plot(
        sensitivity["basis_dimensions"],
        sensitivity["held_out_projection_energy"],
        marker="o",
    )
    axes[2].set_xlabel("Population basis dimension per side")
    axes[2].set_ylabel("Held-out projected OV energy")
    axes[2].set_title("C  Shared subspaces are gradual, not compact", loc="left")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    reports = [
        json.loads(args.primary.read_text(encoding="utf-8")),
        json.loads(args.reciprocal.read_text(encoding="utf-8")),
    ]
    sparsity = json.loads(args.sparsity_audit.read_text(encoding="utf-8"))
    operators, metadata = load_factor_bundle(args.ov)
    training = [item for item in operators if item.head % 2 == 0]
    evaluation = [item for item in operators if item.head % 2 == 1]
    read_basis, write_basis = population_operator_bases(training, operators[0].d_model)
    dimensions = (32, 64, 128, 256, 384, 512)
    projection = [
        float(
            np.mean(
                [
                    np.linalg.norm(
                        project_operator(item, read_basis[:, :dimension], write_basis[:, :dimension])
                    )
                    ** 2
                    / max(factorized_frobenius_norm(item) ** 2, 1e-20)
                    for item in evaluation
                ]
            )
        )
        for dimension in dimensions
    ]
    summary = {
        "status": "consolidated direct OV restricted-map pilot",
        "model": metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": metadata.get("revision", "step143000"),
        "synthetic_calibration": reports[0]["synthetic_calibration"],
        "reciprocal_splits_at_budget_8000": [
            split_summary(report, sparsity["splits"][index], 1)
            for index, report in enumerate(reports)
        ],
        "population_basis_sensitivity": {
            "basis_dimensions": list(dimensions),
            "held_out_projection_energy": projection,
            "amortized_scalar_cost_over_24_evaluation_heads": [
                shared_basis_scalar_cost(operators[0].d_model, dimension, len(evaluation))
                for dimension in dimensions
            ],
        },
        "interpretation_gate": (
            "block model must beat both rotated blocks and unstructured sparse coefficients; "
            "only the first condition is met"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_report(summary, args.figure)
    print(f"saved summary to {args.output}")
    print(f"saved figure to {args.figure}")


if __name__ == "__main__":
    main()
