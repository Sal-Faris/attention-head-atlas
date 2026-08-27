"""Discover OV channel compartments from prompt-independent architectural fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.architectural_compartments import (
    confirmation_r2,
    factor_overlap,
    fit_compartments,
    weighted_subspace_overlap,
)
from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_singular_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ov", type=Path, default=Path("artifacts/pythia-70m-deduped/step143000/ov_factors.npz")
    )
    parser.add_argument(
        "--qk", type=Path, default=Path("artifacts/pythia-70m-deduped/step143000/qk_factors.npz")
    )
    parser.add_argument("--maximum-components", type=int, default=6)
    parser.add_argument("--discovery-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--pairing-nulls", type=int, default=19)
    parser.add_argument("--label-nulls", type=int, default=99)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_ov_compartments_v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_ov_compartments_v1.png"),
    )
    return parser.parse_args()


def location_index(operators: list[object]) -> dict[tuple[int, int], int]:
    return {(operator.layer, operator.head): index for index, operator in enumerate(operators)}


def fingerprints(
    index: int,
    ov: list[object],
    qk: list[object],
    decomposed_ov: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    write_permutation: np.ndarray | None = None,
    discovery_parity: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return discovery/confirmation channel fingerprints and singular gains."""

    operator = ov[index]
    read_modes, spectrum, write_modes = decomposed_ov[index]
    if write_permutation is not None:
        write_modes = write_modes[:, write_permutation]
    lookup = location_index(ov)
    layer_count = max(item.layer for item in ov) + 1
    head_count = max(item.head for item in ov) + 1
    feature_halves: dict[int, list[np.ndarray]] = {0: [], 1: []}

    for source_layer in range(operator.layer):
        for source_head in range(head_count):
            source_index = lookup[(source_layer, source_head)]
            _, source_spectrum, source_writes = decomposed_ov[source_index]
            feature_halves[source_head % 2].append(
                weighted_subspace_overlap(read_modes, source_writes, source_spectrum)
            )

    for target_layer in range(operator.layer + 1, layer_count):
        for target_head in range(head_count):
            target_index = lookup[(target_layer, target_head)]
            parity = target_head % 2
            feature_halves[parity].extend(
                [
                    factor_overlap(write_modes, qk[target_index].left),
                    factor_overlap(write_modes, qk[target_index].right),
                    factor_overlap(write_modes, ov[target_index].left),
                ]
            )

    normalized_gain = spectrum / max(np.linalg.norm(spectrum), 1e-12)
    discovery_columns = feature_halves[discovery_parity] + [normalized_gain]
    confirmation_columns = feature_halves[1 - discovery_parity]
    if not confirmation_columns:
        raise RuntimeError("architectural confirmation fingerprint is empty")
    discovery = np.stack(discovery_columns, axis=1)
    confirmation = np.stack(confirmation_columns, axis=1)
    return discovery, confirmation, normalized_gain


def fit_population(
    ov: list[object],
    qk: list[object],
    decomposed_ov: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    maximum_components: int,
    seed: int,
    permutations: list[np.ndarray] | None = None,
    discovery_parity: int = 0,
) -> tuple[list[dict[str, object]], float]:
    records = []
    scores = []
    for index, operator in enumerate(ov):
        permutation = None if permutations is None else permutations[index]
        discovery, confirmation, spectrum = fingerprints(
            index,
            ov,
            qk,
            decomposed_ov,
            write_permutation=permutation,
            discovery_parity=discovery_parity,
        )
        fit = fit_compartments(
            discovery,
            confirmation,
            maximum_components=maximum_components,
            seed=seed + index,
        )
        sizes = np.bincount(fit.labels, minlength=fit.component_count)
        energies = np.asarray(
            [np.sum(spectrum[fit.labels == label] ** 2) for label in range(fit.component_count)]
        )
        records.append(
            {
                "layer": operator.layer,
                "head": operator.head,
                "component_count": fit.component_count,
                "component_dimensions": sizes.tolist(),
                "component_energy_fractions": energies.tolist(),
                "confirmation_r2": fit.confirmation_r2,
                "bic": fit.bic,
                "labels": fit.labels.tolist(),
            }
        )
        scores.append(fit.confirmation_r2)
    return records, float(np.mean(scores))


def upper_tail(observed: float, null: list[float]) -> float:
    values = np.asarray(null)
    return float((1 + np.sum(values >= observed)) / (1 + len(values)))


def plot_report(report: dict[str, object], output: Path) -> None:
    heads = report["heads"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    counts = np.asarray([record["component_count"] for record in heads])
    axes[0].hist(counts, bins=np.arange(0.5, counts.max() + 1.5), rwidth=0.8)
    axes[0].set_xlabel("BIC-selected compartments per OV head")
    axes[0].set_ylabel("Head count")
    axes[0].set_title("A  Variable-size partitions", loc="left")

    layers = np.asarray([record["layer"] for record in heads])
    scores = np.asarray([record["confirmation_r2"] for record in heads])
    axes[1].scatter(layers + np.random.default_rng(0).normal(0, 0.04, len(layers)), scores)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Held-out architectural R2")
    axes[1].set_title("B  Discovery transfers across anchor heads", loc="left")

    null = np.asarray(report["read_write_pairing_null"]["population_mean_samples"])
    axes[2].hist(null, bins=10, alpha=0.7, label="Re-paired channel null")
    axes[2].axvline(report["population_mean_confirmation_r2"], color="red", label="Real")
    axes[2].set_xlabel("Population mean held-out R2")
    axes[2].set_ylabel("Null repetitions")
    axes[2].set_title("C  Does read/write pairing matter?", loc="left")
    axes[2].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ov, ov_metadata = load_factor_bundle(args.ov)
    qk, _ = load_factor_bundle(args.qk)
    if [(item.layer, item.head) for item in ov] != [(item.layer, item.head) for item in qk]:
        raise ValueError("OV and QK locations do not align")
    decomposed = [factorized_singular_components(operator) for operator in ov]
    observed, observed_mean = fit_population(
        ov,
        qk,
        decomposed,
        maximum_components=args.maximum_components,
        seed=args.seed,
        discovery_parity=args.discovery_parity,
    )

    rng = np.random.default_rng(args.seed)
    pairing_null = []
    for repetition in range(args.pairing_nulls):
        permutations = [rng.permutation(operator.d_head) for operator in ov]
        _, score = fit_population(
            ov,
            qk,
            decomposed,
            maximum_components=args.maximum_components,
            seed=args.seed + 1000 * (repetition + 1),
            permutations=permutations,
            discovery_parity=args.discovery_parity,
        )
        pairing_null.append(score)
        print(f"pairing null {repetition + 1}/{args.pairing_nulls}", flush=True)

    label_null = []
    confirmations = [
        fingerprints(
            i, ov, qk, decomposed, discovery_parity=args.discovery_parity
        )[1]
        for i in range(len(ov))
    ]
    for _ in range(args.label_nulls):
        values = []
        for record, confirmation in zip(observed, confirmations, strict=True):
            labels = rng.permutation(np.asarray(record["labels"], dtype=np.int64))
            values.append(confirmation_r2(confirmation, labels))
        label_null.append(float(np.mean(values)))

    report = {
        "status": "prompt-independent architecture-connected OV channel compartments",
        "model": ov_metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": ov_metadata.get("revision", "step143000"),
        "discovery_anchor_head_parity": args.discovery_parity,
        "confirmation_anchor_head_parity": 1 - args.discovery_parity,
        "channel_definition": "exact OV singular read/write pair",
        "discovery_fingerprint": (
            "normalized gain plus overlaps with discovery-parity earlier OV writers and later Q/K/V readers"
        ),
        "confirmation_fingerprint": (
            "overlaps with opposite-parity held-out earlier OV writers and later Q/K/V readers"
        ),
        "selection": "diagonal Gaussian mixture in PCA fingerprint coordinates; component count by BIC",
        "population_mean_confirmation_r2": observed_mean,
        "component_count_summary": {
            "mean": float(np.mean([record["component_count"] for record in observed])),
            "median": float(np.median([record["component_count"] for record in observed])),
            "minimum": int(np.min([record["component_count"] for record in observed])),
            "maximum": int(np.max([record["component_count"] for record in observed])),
        },
        "read_write_pairing_null": {
            "definition": "independently permute write singular modes within every head before refitting",
            "population_mean_samples": pairing_null,
            "mean": float(np.mean(pairing_null)),
            "observed_minus_mean": float(observed_mean - np.mean(pairing_null)),
            "upper_tail_p_value": upper_tail(observed_mean, pairing_null),
        },
        "held_out_anchor_label_null": {
            "definition": "permute each real fitted partition across channels",
            "population_mean_samples": label_null,
            "mean": float(np.mean(label_null)),
            "observed_minus_mean": float(observed_mean - np.mean(label_null)),
            "upper_tail_p_value": upper_tail(observed_mean, label_null),
        },
        "heads": observed,
        "pairing_null_repetitions": args.pairing_nulls,
        "label_null_repetitions": args.label_nulls,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved result to {args.output}")
    print(f"saved figure to {args.figure}")


if __name__ == "__main__":
    main()
