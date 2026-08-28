"""Test whether OV maps concentrate in externally discovered architectural module pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from head_atlas.architectural_modules import (
    axis_usage_profiles,
    module_pair_energy,
    partition_axis_profiles,
    top_pair_statistics,
)
from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_frobenius_norm
from head_atlas.restricted_maps import (
    architectural_anchor_covariances,
    architectural_operator_bases,
    project_operator,
)

RESOLUTIONS = (2, 3, 4, 6, 8)


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
    parser.add_argument("--null-repetitions", type=int, default=99)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_module_pairs_v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_module_pairs_v1.png"),
    )
    return parser.parse_args()


def rotated_matrix(
    singular_values: np.ndarray,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample independent left/right orientations with a fixed nonzero spectrum."""

    rank = len(singular_values)
    left, _ = np.linalg.qr(rng.standard_normal((dimension, rank)), mode="reduced")
    right, _ = np.linalg.qr(rng.standard_normal((dimension, rank)), mode="reduced")
    return (left * singular_values[None, :]) @ right.T


def discover_layer_modules(
    ov: list[object],
    qk: list[object],
    *,
    layer: int,
    parity: int,
    dimension: int,
    seed: int,
) -> dict[str, object]:
    read_covariances, write_covariances = architectural_anchor_covariances(
        ov,
        qk,
        target_layer=layer,
        anchor_head_parity=parity,
    )
    read_basis, write_basis = architectural_operator_bases(
        ov,
        qk,
        target_layer=layer,
        anchor_head_parity=parity,
        dimension=dimension,
    )
    read_profiles = axis_usage_profiles(read_covariances, read_basis)
    write_profiles = axis_usage_profiles(write_covariances, write_basis)
    partitions = {}
    for resolution in RESOLUTIONS:
        read_labels = partition_axis_profiles(
            read_profiles,
            resolution,
            seed=seed + 100 * layer + 10 * parity + resolution,
        )
        write_labels = partition_axis_profiles(
            write_profiles,
            resolution,
            seed=seed + 10_000 + 100 * layer + 10 * parity + resolution,
        )
        partitions[str(resolution)] = {
            "read_labels": read_labels,
            "write_labels": write_labels,
            "read_sizes": np.bincount(read_labels, minlength=resolution),
            "write_sizes": np.bincount(write_labels, minlength=resolution),
        }
    return {
        "read_basis": read_basis,
        "write_basis": write_basis,
        "read_anchor_count": len(read_covariances),
        "write_anchor_count": len(write_covariances),
        "partitions": partitions,
    }


def concentration(
    coefficients: np.ndarray,
    read_labels: np.ndarray,
    write_labels: np.ndarray,
    resolution: int,
) -> dict[str, float | np.ndarray]:
    energy, area = module_pair_energy(coefficients, read_labels, write_labels, resolution)
    return top_pair_statistics(energy, area, pair_count=resolution)


def max_corrected_test(
    observed: np.ndarray,
    null_samples: np.ndarray,
) -> dict[str, object]:
    """Correct selection over resolutions using a centered max-statistic null."""

    observed_population = np.mean(observed, axis=0)
    null_population = np.mean(null_samples, axis=0)
    null_center = np.mean(null_population, axis=0)
    advantages = observed_population - null_center
    selected_index = int(np.argmax(advantages))
    observed_max = float(advantages[selected_index])
    null_maxima = np.max(null_population - null_center[None, :], axis=1)
    p_value = float((1 + np.sum(null_maxima >= observed_max)) / (len(null_maxima) + 1))
    return {
        "observed_population_by_resolution": observed_population.tolist(),
        "null_population_mean_by_resolution": null_center.tolist(),
        "advantage_by_resolution": advantages.tolist(),
        "selected_resolution": RESOLUTIONS[selected_index],
        "selected_index": selected_index,
        "selection_corrected_p_value": p_value,
        "observed_selected_advantage": observed_max,
    }


def layer_resolution_corrected_test(
    observed: np.ndarray,
    null_samples: np.ndarray,
    target_layers: tuple[int, ...],
    targets: list[object],
) -> dict[str, object]:
    """Correct a search over both target layer and module resolution."""

    observed_configurations = []
    null_configurations = []
    labels = []
    target_layer_array = np.asarray([operator.layer for operator in targets])
    for layer in target_layers:
        selected = target_layer_array == layer
        for resolution_index, resolution in enumerate(RESOLUTIONS):
            observed_configurations.append(float(np.mean(observed[selected, resolution_index])))
            null_configurations.append(np.mean(null_samples[selected, :, resolution_index], axis=0))
            labels.append({"layer": layer, "resolution": resolution})
    observed_values = np.asarray(observed_configurations)
    null_values = np.stack(null_configurations, axis=1)
    null_center = np.mean(null_values, axis=0)
    advantages = observed_values - null_center
    selected_index = int(np.argmax(advantages))
    observed_max = float(advantages[selected_index])
    null_maxima = np.max(null_values - null_center[None, :], axis=1)
    p_value = float((1 + np.sum(null_maxima >= observed_max)) / (len(null_maxima) + 1))
    return {
        "configurations": labels,
        "observed_concentration": observed_values.tolist(),
        "null_mean_concentration": null_center.tolist(),
        "advantage": advantages.tolist(),
        "selected": labels[selected_index],
        "selected_advantage": observed_max,
        "selection_corrected_p_value": p_value,
    }


def matched_partition_overlap(
    first_basis: np.ndarray,
    first_labels: np.ndarray,
    second_basis: np.ndarray,
    second_labels: np.ndarray,
    resolution: int,
) -> float:
    """Match two ambient subspace partitions and return dimension-weighted overlap."""

    overlap = np.zeros((resolution, resolution), dtype=np.float64)
    for first_label in range(resolution):
        first = first_basis[:, first_labels == first_label]
        for second_label in range(resolution):
            second = second_basis[:, second_labels == second_label]
            overlap[first_label, second_label] = float(np.sum((first.T @ second) ** 2))
    rows, columns = linear_sum_assignment(-overlap)
    return float(np.sum(overlap[rows, columns]) / first_basis.shape[1])


def cross_anchor_stability(
    ov: list[object],
    qk: list[object],
    *,
    target_layers: tuple[int, ...],
    dimension: int,
    repetitions: int,
    rng: np.random.Generator,
    seed: int,
) -> dict[str, object]:
    observed = np.zeros((len(target_layers), len(RESOLUTIONS)))
    shuffled = np.zeros((len(target_layers), repetitions, len(RESOLUTIONS)))
    layer_records = []
    for layer_index, layer in enumerate(target_layers):
        first = discover_layer_modules(
            ov, qk, layer=layer, parity=0, dimension=dimension, seed=seed
        )
        second = discover_layer_modules(
            ov, qk, layer=layer, parity=1, dimension=dimension, seed=seed
        )
        resolution_records = []
        for resolution_index, resolution in enumerate(RESOLUTIONS):
            first_partition = first["partitions"][str(resolution)]
            second_partition = second["partitions"][str(resolution)]
            observed[layer_index, resolution_index] = matched_partition_overlap(
                first["read_basis"],
                first_partition["read_labels"],
                second["read_basis"],
                second_partition["read_labels"],
                resolution,
            )
            write_observed = matched_partition_overlap(
                first["write_basis"],
                first_partition["write_labels"],
                second["write_basis"],
                second_partition["write_labels"],
                resolution,
            )
            observed[layer_index, resolution_index] = (
                observed[layer_index, resolution_index] + write_observed
            ) / 2.0
            for repetition in range(repetitions):
                read_null = matched_partition_overlap(
                    first["read_basis"],
                    rng.permutation(first_partition["read_labels"]),
                    second["read_basis"],
                    rng.permutation(second_partition["read_labels"]),
                    resolution,
                )
                write_null = matched_partition_overlap(
                    first["write_basis"],
                    rng.permutation(first_partition["write_labels"]),
                    second["write_basis"],
                    rng.permutation(second_partition["write_labels"]),
                    resolution,
                )
                shuffled[layer_index, repetition, resolution_index] = (read_null + write_null) / 2.0
            resolution_records.append(
                {
                    "resolution": resolution,
                    "mean_read_write_overlap": observed[layer_index, resolution_index],
                    "shuffle_mean": float(np.mean(shuffled[layer_index, :, resolution_index])),
                }
            )
        layer_records.append({"layer": layer, "resolutions": resolution_records})
    test = max_corrected_test(observed, shuffled)
    test["layers"] = layer_records
    return test


def analyze_parity(
    ov: list[object],
    qk: list[object],
    *,
    target_layers: tuple[int, ...],
    parity: int,
    dimension: int,
    repetitions: int,
    rng: np.random.Generator,
    seed: int,
) -> dict[str, object]:
    layer_modules = {
        layer: discover_layer_modules(
            ov,
            qk,
            layer=layer,
            parity=parity,
            dimension=dimension,
            seed=seed,
        )
        for layer in target_layers
    }
    targets = [operator for operator in ov if operator.layer in target_layers]
    observed = np.zeros((len(targets), len(RESOLUTIONS)))
    area = np.zeros_like(observed)
    enrichment = np.zeros_like(observed)
    effective_pairs = np.zeros_like(observed)
    projection = np.zeros(len(targets))
    rotation = np.zeros((len(targets), repetitions, len(RESOLUTIONS)))
    shuffled = np.zeros_like(rotation)
    per_head = []

    for head_index, operator in enumerate(targets):
        modules = layer_modules[operator.layer]
        total = max(factorized_frobenius_norm(operator) ** 2, 1e-20)
        coefficients = project_operator(
            operator,
            modules["read_basis"],
            modules["write_basis"],
        ) / np.sqrt(total)
        projection[head_index] = float(np.sum(coefficients**2))
        singular_values = np.linalg.svd(coefficients, compute_uv=False)
        threshold = max(float(singular_values[0]) * 1e-12, 1e-15)
        singular_values = singular_values[singular_values > threshold]
        rotated = [rotated_matrix(singular_values, dimension, rng) for _ in range(repetitions)]
        head_resolutions = []
        for resolution_index, resolution in enumerate(RESOLUTIONS):
            partition = modules["partitions"][str(resolution)]
            real = concentration(
                coefficients,
                partition["read_labels"],
                partition["write_labels"],
                resolution,
            )
            observed[head_index, resolution_index] = real["energy_fraction"]
            area[head_index, resolution_index] = real["area_fraction"]
            enrichment[head_index, resolution_index] = real["energy_enrichment"]
            effective_pairs[head_index, resolution_index] = real["effective_pair_count"]
            for repetition in range(repetitions):
                rotation[head_index, repetition, resolution_index] = concentration(
                    rotated[repetition],
                    partition["read_labels"],
                    partition["write_labels"],
                    resolution,
                )["energy_fraction"]
                shuffled_read = rng.permutation(partition["read_labels"])
                shuffled_write = rng.permutation(partition["write_labels"])
                shuffled[head_index, repetition, resolution_index] = concentration(
                    coefficients,
                    shuffled_read,
                    shuffled_write,
                    resolution,
                )["energy_fraction"]
            head_resolutions.append(
                {
                    "resolution": resolution,
                    "top_pair_count": resolution,
                    "energy_fraction_of_projection": float(real["energy_fraction"]),
                    "area_fraction": float(real["area_fraction"]),
                    "energy_enrichment": float(real["energy_enrichment"]),
                    "effective_pair_count": float(real["effective_pair_count"]),
                    "rotation_mean": float(np.mean(rotation[head_index, :, resolution_index])),
                    "partition_shuffle_mean": float(
                        np.mean(shuffled[head_index, :, resolution_index])
                    ),
                }
            )
        per_head.append(
            {
                "layer": operator.layer,
                "head": operator.head,
                "projection_energy_fraction": projection[head_index],
                "resolutions": head_resolutions,
            }
        )
        print(f"anchor parity {parity}: L{operator.layer}H{operator.head}", flush=True)

    rotation_test = max_corrected_test(observed, rotation)
    shuffle_test = max_corrected_test(observed, shuffled)
    layer_rotation_test = layer_resolution_corrected_test(
        observed, rotation, target_layers, targets
    )
    layer_shuffle_test = layer_resolution_corrected_test(observed, shuffled, target_layers, targets)
    selected_index = int(rotation_test["selected_index"])
    selected_resolution = RESOLUTIONS[selected_index]
    captured_full = observed[:, selected_index] * projection
    layer_description = {}
    for layer, modules in layer_modules.items():
        layer_description[str(layer)] = {
            "read_anchor_count": modules["read_anchor_count"],
            "write_anchor_count": modules["write_anchor_count"],
            "partitions": {
                key: {
                    "read_sizes": value["read_sizes"].tolist(),
                    "write_sizes": value["write_sizes"].tolist(),
                }
                for key, value in modules["partitions"].items()
            },
        }
    return {
        "anchor_head_parity": parity,
        "target_head_count": len(targets),
        "layer_modules": layer_description,
        "resolutions": list(RESOLUTIONS),
        "mean_projection_energy_fraction": float(np.mean(projection)),
        "mean_energy_fraction_of_projection_by_resolution": np.mean(observed, axis=0).tolist(),
        "mean_selected_area_fraction_by_resolution": np.mean(area, axis=0).tolist(),
        "mean_energy_enrichment_by_resolution": np.mean(enrichment, axis=0).tolist(),
        "mean_effective_pair_count_by_resolution": np.mean(effective_pairs, axis=0).tolist(),
        "spectrum_rotation_test": rotation_test,
        "matched_partition_shuffle_test": shuffle_test,
        "layer_and_resolution_corrected_spectrum_test": layer_rotation_test,
        "layer_and_resolution_corrected_partition_test": layer_shuffle_test,
        "primary_selected_resolution": selected_resolution,
        "mean_full_ov_energy_in_selected_pairs": float(np.mean(captured_full)),
        "fraction_heads_above_rotation_mean_at_selected_resolution": float(
            np.mean(observed[:, selected_index] > np.mean(rotation[:, :, selected_index], axis=1))
        ),
        "fraction_heads_above_partition_shuffle_mean_at_selected_resolution": float(
            np.mean(observed[:, selected_index] > np.mean(shuffled[:, :, selected_index], axis=1))
        ),
        "heads": per_head,
    }


def plot_report(report: dict[str, object], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes = axes.ravel()
    colors = ("tab:blue", "tab:orange")
    for split, color in zip(report["anchor_splits"], colors, strict=True):
        resolutions = split["resolutions"]
        axes[0].plot(
            resolutions,
            split["mean_energy_fraction_of_projection_by_resolution"],
            marker="o",
            color=color,
            label=f"Real, anchors {split['anchor_head_parity']}",
        )
        axes[0].plot(
            resolutions,
            split["spectrum_rotation_test"]["null_population_mean_by_resolution"],
            linestyle="--",
            color=color,
            label=f"Rotated, anchors {split['anchor_head_parity']}",
        )
        axes[1].plot(
            resolutions,
            split["spectrum_rotation_test"]["advantage_by_resolution"],
            marker="o",
            color=color,
            label=f"Spectrum null, anchors {split['anchor_head_parity']}",
        )
        axes[1].plot(
            resolutions,
            split["matched_partition_shuffle_test"]["advantage_by_resolution"],
            marker="s",
            linestyle=":",
            color=color,
            label=f"Partition shuffle, anchors {split['anchor_head_parity']}",
        )
    axes[0].set_xlabel("Architectural modules per side")
    axes[0].set_ylabel("Energy in strongest k module pairs / projected energy")
    axes[0].set_title("A  Do external modules localize OV action?", loc="left")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Architectural modules per side")
    axes[1].set_ylabel("Real minus null concentration")
    axes[1].set_title("B  Resolution-corrected advantages", loc="left")
    axes[1].legend(fontsize=7)

    labels = ("Projected OV", "In selected pairs")
    x = np.arange(2)
    width = 0.36
    for offset, split in zip((-width / 2, width / 2), report["anchor_splits"], strict=True):
        values = [
            split["mean_projection_energy_fraction"],
            split["mean_full_ov_energy_in_selected_pairs"],
        ]
        axes[2].bar(
            x + offset,
            values,
            width,
            label=(
                f"Anchors {split['anchor_head_parity']}, k={split['primary_selected_resolution']}"
            ),
        )
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Fraction of full OV energy")
    axes[2].set_title("C  Absolute variance explained", loc="left")
    axes[2].legend(fontsize=8)

    stability = report["cross_anchor_partition_stability"]
    axes[3].plot(
        RESOLUTIONS,
        stability["observed_population_by_resolution"],
        marker="o",
        label="Even/odd matched overlap",
    )
    axes[3].plot(
        RESOLUTIONS,
        stability["null_population_mean_by_resolution"],
        marker="o",
        linestyle="--",
        label="Size-matched permutation",
    )
    axes[3].set_xlabel("Architectural modules per side")
    axes[3].set_ylabel("Dimension-weighted matched subspace overlap")
    axes[3].set_title("D  Are module boundaries reproducible?", loc="left")
    axes[3].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ov, metadata = load_factor_bundle(args.ov)
    qk, _ = load_factor_bundle(args.qk)
    target_layers = tuple(sorted(set(args.target_layers)))
    rng = np.random.default_rng(args.seed)
    splits = [
        analyze_parity(
            ov,
            qk,
            target_layers=target_layers,
            parity=parity,
            dimension=args.basis_dimension,
            repetitions=args.null_repetitions,
            rng=rng,
            seed=args.seed,
        )
        for parity in (0, 1)
    ]
    stability = cross_anchor_stability(
        ov,
        qk,
        target_layers=target_layers,
        dimension=args.basis_dimension,
        repetitions=args.null_repetitions,
        rng=rng,
        seed=args.seed,
    )
    report = {
        "status": "target-independent architectural module-pair pilot",
        "model": metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": metadata.get("revision", "step143000"),
        "operator_kind": "OV",
        "discovery": "weights only; target matrices do not define module boundaries",
        "target_layers": list(target_layers),
        "basis_dimension": args.basis_dimension,
        "module_definition": (
            "cluster residual axes by normalized diagonal usage profiles across individual "
            "earlier OV-writer or later Q/K/V-reader covariance anchors"
        ),
        "compartment_statistic": (
            "fraction of projected OV energy in the strongest k of k-squared externally "
            "defined read-module/write-module pairs"
        ),
        "selection_control": (
            "choose k by maximum population advantage and compare with the centered maximum "
            "over the same resolutions in every null repetition"
        ),
        "nulls": {
            "spectrum_rotation": "independent orientations preserving projected singular values",
            "matched_partition_shuffle": (
                "permute read and write module memberships while preserving every group size"
            ),
        },
        "anchor_splits": splits,
        "cross_anchor_partition_stability": stability,
        "null_repetitions": args.null_repetitions,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved result to {args.output}")
    print(f"saved figure to {args.figure}")


if __name__ == "__main__":
    main()
