"""Test direct OV compartments in independently anchored architectural coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pilot_direct_ov_restricted_maps import (
    fit_frontier,
    full_svd_energy,
    projected_dense_low_rank_energy,
    select_at_budget,
    unstructured_sparse_energy,
)

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_frobenius_norm
from head_atlas.restricted_maps import (
    architectural_operator_bases,
    project_operator,
    shared_basis_scalar_cost,
    spectrum_matched_rotation,
)

BUDGETS = (1_000.0, 2_000.0, 4_000.0, 8_000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ov",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/step143000/ov_factors.npz"),
    )
    parser.add_argument(
        "--qk",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/step143000/qk_factors.npz"),
    )
    parser.add_argument("--target-layers", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--basis-dimension", type=int, default=128)
    parser.add_argument("--maximum-blocks", type=int, default=6)
    parser.add_argument("--null-repetitions", type=int, default=19)
    parser.add_argument("--seed", type=int, default=27182)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_ov_restricted_map_pilot_v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_ov_restricted_map_pilot_v1.png"),
    )
    return parser.parse_args()


def projection_fraction(operator: object, read: np.ndarray, write: np.ndarray) -> float:
    total = max(factorized_frobenius_norm(operator) ** 2, 1e-20)
    return float(np.sum(project_operator(operator, read, write) ** 2) / total)


def projection_sensitivity(
    ov: list[object],
    qk: list[object],
    target_layers: tuple[int, ...],
) -> dict[str, object]:
    width = ov[0].d_model
    dimensions = tuple(value for value in (32, 64, 128, 256, 384) if value <= width)
    by_parity: dict[str, list[float]] = {}
    for parity in (0, 1):
        means = []
        for dimension in dimensions:
            fractions = []
            for layer in target_layers:
                read, write = architectural_operator_bases(
                    ov,
                    qk,
                    target_layer=layer,
                    anchor_head_parity=parity,
                    dimension=dimension,
                )
                fractions.extend(
                    projection_fraction(operator, read, write)
                    for operator in ov
                    if operator.layer == layer
                )
            means.append(float(np.mean(fractions)))
        by_parity[str(parity)] = means
    return {"dimensions": list(dimensions), "mean_energy_fraction_by_anchor_parity": by_parity}


def analyze_anchor_split(
    ov: list[object],
    qk: list[object],
    *,
    target_layers: tuple[int, ...],
    anchor_parity: int,
    basis_dimension: int,
    maximum_blocks: int,
    null_repetitions: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    method_samples = {
        name: [[] for _ in BUDGETS]
        for name in (
            "restricted_blocks",
            "unstructured_sparse",
            "projected_dense_low_rank",
            "full_svd",
        )
    }
    null_block_by_repetition = [[[] for _ in BUDGETS] for _ in range(null_repetitions)]
    null_sparse_by_repetition = [[[] for _ in BUDGETS] for _ in range(null_repetitions)]
    per_head = []
    layer_projection: dict[str, list[float]] = {}
    selected_supports: list[dict[str, int]] = []

    for layer in target_layers:
        read, write = architectural_operator_bases(
            ov,
            qk,
            target_layer=layer,
            anchor_head_parity=anchor_parity,
            dimension=basis_dimension,
        )
        targets = [operator for operator in ov if operator.layer == layer]
        layer_projection[str(layer)] = []
        for operator in targets:
            total = max(factorized_frobenius_norm(operator) ** 2, 1e-20)
            coefficients = project_operator(operator, read, write) / np.sqrt(total)
            projected_energy = float(np.sum(coefficients**2))
            layer_projection[str(layer)].append(projected_energy)
            frontier = fit_frontier(coefficients, maximum_blocks)
            null_coefficients = [
                spectrum_matched_rotation(coefficients, rng) for _ in range(null_repetitions)
            ]
            null_frontiers = [fit_frontier(values, maximum_blocks) for values in null_coefficients]
            head_budgets = []
            for budget_index, budget in enumerate(BUDGETS):
                selected = select_at_budget(frontier, budget)
                real_values = {
                    "restricted_blocks": float(selected["captured_energy"]),
                    "unstructured_sparse": unstructured_sparse_energy(coefficients, budget),
                    "projected_dense_low_rank": projected_dense_low_rank_energy(
                        coefficients, budget
                    ),
                    "full_svd": full_svd_energy(operator, budget),
                }
                for name, value in real_values.items():
                    method_samples[name][budget_index].append(value)
                null_blocks = []
                null_sparse = []
                for repetition, (values, null_frontier) in enumerate(
                    zip(null_coefficients, null_frontiers, strict=True)
                ):
                    block_value = float(select_at_budget(null_frontier, budget)["captured_energy"])
                    sparse_value = unstructured_sparse_energy(values, budget)
                    null_blocks.append(block_value)
                    null_sparse.append(sparse_value)
                    null_block_by_repetition[repetition][budget_index].append(block_value)
                    null_sparse_by_repetition[repetition][budget_index].append(sparse_value)
                if budget == 4_000.0:
                    selected_supports.extend(selected["blocks"])
                head_budgets.append(
                    {
                        "scalar_budget": budget,
                        **{f"{name}_energy_fraction": value for name, value in real_values.items()},
                        "rotation_block_mean_energy_fraction": float(np.mean(null_blocks)),
                        "rotation_sparse_mean_energy_fraction": float(np.mean(null_sparse)),
                        "selected_blocks": selected["blocks"],
                    }
                )
            per_head.append(
                {
                    "layer": operator.layer,
                    "head": operator.head,
                    "architectural_projection_energy_fraction": projected_energy,
                    "budgets": head_budgets,
                }
            )
            print(
                f"anchor parity {anchor_parity}: L{operator.layer}H{operator.head}",
                flush=True,
            )

    means = {
        name: [float(np.mean(values)) for values in budget_values]
        for name, budget_values in method_samples.items()
    }
    null_block_means = np.asarray(
        [
            [float(np.mean(repetition[index])) for index in range(len(BUDGETS))]
            for repetition in null_block_by_repetition
        ]
    )
    null_sparse_means = np.asarray(
        [
            [float(np.mean(repetition[index])) for index in range(len(BUDGETS))]
            for repetition in null_sparse_by_repetition
        ]
    )
    p_values = []
    for index in range(len(BUDGETS)):
        observed = means["restricted_blocks"][index]
        p_values.append(
            float((1 + np.sum(null_block_means[:, index] >= observed)) / (null_repetitions + 1))
        )
    primary = BUDGETS.index(4_000.0)
    gate = {
        "beats_spectrum_rotation": means["restricted_blocks"][primary]
        > float(np.mean(null_block_means[:, primary])),
        "beats_unstructured_sparsity": means["restricted_blocks"][primary]
        > means["unstructured_sparse"][primary],
        "beats_one_dense_low_rank_map": means["restricted_blocks"][primary]
        > means["projected_dense_low_rank"][primary],
        "rotation_p_value": p_values[primary],
        "passes_all_at_budget_4000": False,
    }
    gate["passes_all_at_budget_4000"] = bool(
        gate["beats_spectrum_rotation"]
        and gate["beats_unstructured_sparsity"]
        and gate["beats_one_dense_low_rank_map"]
        and gate["rotation_p_value"] <= 0.05
    )
    conservative_cost = shared_basis_scalar_cost(
        ov[0].d_model,
        basis_dimension,
        sum(operator.layer in target_layers for operator in ov),
    )
    return {
        "anchor_head_parity": anchor_parity,
        "target_layers": list(target_layers),
        "target_head_count": len(per_head),
        "basis_dimension": basis_dimension,
        "conditional_basis_cost": 0.0,
        "conservative_amortized_basis_cost": conservative_cost,
        "basis_cost_note": (
            "primary comparison treats coordinates as side information deterministically derived "
            "from other model weights; conservative Stiefel cost is reported but not subtracted"
        ),
        "mean_projection_energy_fraction_by_layer": {
            layer: float(np.mean(values)) for layer, values in layer_projection.items()
        },
        "budgets": list(BUDGETS),
        "mean_energy_fraction": means,
        "rotation_block_population_means": [
            float(np.mean(null_block_means[:, index])) for index in range(len(BUDGETS))
        ],
        "rotation_sparse_population_means": [
            float(np.mean(null_sparse_means[:, index])) for index in range(len(BUDGETS))
        ],
        "rotation_block_population_p_values": p_values,
        "selected_block_shapes_at_budget_4000": selected_supports,
        "decision_gate_at_budget_4000": gate,
        "heads": per_head,
    }


def plot_report(report: dict[str, object], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    sensitivity = report["projection_sensitivity"]
    for parity, values in sensitivity["mean_energy_fraction_by_anchor_parity"].items():
        axes[0].plot(sensitivity["dimensions"], values, marker="o", label=f"Anchor parity {parity}")
    axes[0].set_xlabel("Architectural basis dimension")
    axes[0].set_ylabel("Mean fraction of full OV energy projected")
    axes[0].set_title("A  How much OV action is architecturally reachable?", loc="left")
    axes[0].legend()

    colors = ("tab:blue", "tab:orange")
    for split, color in zip(report["anchor_splits"], colors, strict=True):
        axes[1].plot(
            split["budgets"],
            split["mean_energy_fraction"]["restricted_blocks"],
            marker="o",
            color=color,
            label=f"Blocks, anchors {split['anchor_head_parity']}",
        )
        axes[1].plot(
            split["budgets"],
            split["rotation_block_population_means"],
            linestyle="--",
            color=color,
            label=f"Rotated null, anchors {split['anchor_head_parity']}",
        )
    axes[1].set_xlabel("Conditional scalar-equivalent budget")
    axes[1].set_ylabel("Mean fraction of full OV energy")
    axes[1].set_title("B  Do architectural blocks beat rotations?", loc="left")
    axes[1].legend(fontsize=8)

    primary = BUDGETS.index(4_000.0)
    labels = ("Blocks", "Sparse", "Dense low-rank", "Full SVD")
    keys = ("restricted_blocks", "unstructured_sparse", "projected_dense_low_rank", "full_svd")
    x = np.arange(len(labels))
    width = 0.36
    for offset, split in zip((-width / 2, width / 2), report["anchor_splits"], strict=True):
        values = [split["mean_energy_fraction"][key][primary] for key in keys]
        axes[2].bar(
            x + offset,
            values,
            width,
            label=f"Anchor parity {split['anchor_head_parity']}",
        )
    axes[2].set_xticks(x, labels, rotation=20, ha="right")
    axes[2].set_ylabel("Mean fraction of full OV energy")
    axes[2].set_title("C  Does compartment structure win at budget 4,000?", loc="left")
    axes[2].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ov, ov_metadata = load_factor_bundle(args.ov)
    qk, qk_metadata = load_factor_bundle(args.qk)
    target_layers = tuple(sorted(set(args.target_layers)))
    rng = np.random.default_rng(args.seed)
    sensitivity = projection_sensitivity(ov, qk, target_layers)
    splits = [
        analyze_anchor_split(
            ov,
            qk,
            target_layers=target_layers,
            anchor_parity=parity,
            basis_dimension=args.basis_dimension,
            maximum_blocks=args.maximum_blocks,
            null_repetitions=args.null_repetitions,
            rng=rng,
        )
        for parity in (0, 1)
    ]
    report = {
        "status": "direct architectural-coordinate OV restricted-map pilot",
        "model": ov_metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": ov_metadata.get("revision", "step143000"),
        "operator_kind": "OV",
        "discovery": "weights only; no tokens, prompts, activations, or semantic labels",
        "coordinate_definition": {
            "read": "unit-trace covariances of exact outputs M_OV^T M_OV from earlier-layer anchor heads",
            "write": "unit-trace Q, K, and V reader covariances from later-layer anchor heads",
            "leakage_control": "target-layer matrices never define either basis",
            "reciprocal_control": "the complete pilot is repeated with even and odd anchor heads",
        },
        "operator_metadata_agree": {
            key: ov_metadata.get(key) == qk_metadata.get(key)
            for key in ("n_layers", "n_heads", "d_model", "d_head")
        },
        "projection_sensitivity": sensitivity,
        "anchor_splits": splits,
        "compartment_gate": (
            "At budget 4000, restricted blocks must significantly beat spectrum-matched "
            "rotations, unstructured coefficient sparsity, and one dense projected low-rank map "
            "under both reciprocal anchor parities."
        ),
        "null_repetitions": args.null_repetitions,
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
