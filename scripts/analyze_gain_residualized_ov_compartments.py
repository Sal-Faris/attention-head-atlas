"""Test OV architectural channel organization after removing singular gain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_architectural_ov_compartments import fingerprints
from sklearn.metrics import adjusted_rand_score

from head_atlas.architectural_compartments import (
    confirmation_r2,
    fit_compartments,
    residualize_against_gain,
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
    parser.add_argument(
        "--raw-audit",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/architectural_ov_compartment_confounders_v1.json"
        ),
    )
    parser.add_argument("--maximum-components", type=int, default=6)
    parser.add_argument("--gain-knots", type=int, default=6)
    parser.add_argument("--pairing-nulls", type=int, default=19)
    parser.add_argument("--label-nulls", type=int, default=99)
    parser.add_argument("--seed", type=int, default=16180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/gain_residualized_ov_compartments_v1.json"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/gain_residualized_ov_compartments_v1.png"
        ),
    )
    return parser.parse_args()


def residual_fingerprints(
    index: int,
    ov: list[object],
    qk: list[object],
    decomposed_ov: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    discovery_parity: int,
    gain_knots: int,
    write_permutation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    discovery, confirmation, gain = fingerprints(
        index,
        ov,
        qk,
        decomposed_ov,
        discovery_parity=discovery_parity,
        write_permutation=write_permutation,
    )
    return (
        residualize_against_gain(discovery[:, :-1], gain, n_knots=gain_knots),
        residualize_against_gain(confirmation, gain, n_knots=gain_knots),
        gain,
    )


def fit_population(
    ov: list[object],
    qk: list[object],
    decomposed_ov: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    discovery_parity: int,
    gain_knots: int,
    maximum_components: int,
    seed: int,
    permutations: list[np.ndarray] | None = None,
) -> tuple[list[dict[str, object]], float, list[np.ndarray]]:
    records = []
    confirmations = []
    for index, operator in enumerate(ov):
        permutation = None if permutations is None else permutations[index]
        discovery, confirmation, gain = residual_fingerprints(
            index,
            ov,
            qk,
            decomposed_ov,
            discovery_parity=discovery_parity,
            gain_knots=gain_knots,
            write_permutation=permutation,
        )
        fit = fit_compartments(
            discovery,
            confirmation,
            maximum_components=maximum_components,
            seed=seed + index,
        )
        sizes = np.bincount(fit.labels, minlength=fit.component_count)
        energy = np.asarray(
            [np.sum(gain[fit.labels == label] ** 2) for label in range(fit.component_count)]
        )
        records.append(
            {
                "layer": operator.layer,
                "head": operator.head,
                "component_count": fit.component_count,
                "component_dimensions": sizes.tolist(),
                "component_energy_fractions": energy.tolist(),
                "confirmation_r2": fit.confirmation_r2,
                "labels": fit.labels.tolist(),
            }
        )
        confirmations.append(confirmation)
    return (
        records,
        float(np.mean([record["confirmation_r2"] for record in records])),
        confirmations,
    )


def upper_tail(observed: float, null: list[float]) -> float:
    values = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(values >= observed)) / (1 + len(values)))


def summarize_split(
    records: list[dict[str, object]],
    observed: float,
    confirmations: list[np.ndarray],
    pairing_null: list[float],
    pairing_head_nulls: list[list[float]],
    *,
    label_repetitions: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    label_null = []
    for _ in range(label_repetitions):
        values = [
            confirmation_r2(
                confirmation,
                rng.permutation(np.asarray(record["labels"], dtype=np.int64)),
            )
            for record, confirmation in zip(records, confirmations, strict=True)
        ]
        label_null.append(float(np.mean(values)))

    for record, head_null in zip(records, pairing_head_nulls, strict=True):
        observed_head = float(record["confirmation_r2"])
        record["pairing_null_mean"] = float(np.mean(head_null))
        record["observed_minus_pairing_null_mean"] = float(
            observed_head - np.mean(head_null)
        )
        record["pairing_null_upper_tail_p"] = upper_tail(observed_head, head_null)

    layer_summary = {}
    for layer in sorted({int(record["layer"]) for record in records}):
        layer_records = [record for record in records if record["layer"] == layer]
        layer_summary[str(layer)] = {
            "mean_confirmation_r2": float(
                np.mean([record["confirmation_r2"] for record in layer_records])
            ),
            "multicomponent_heads": int(
                sum(record["component_count"] > 1 for record in layer_records)
            ),
        }
    return {
        "population_mean_confirmation_r2": observed,
        "mean_component_count": float(
            np.mean([record["component_count"] for record in records])
        ),
        "multicomponent_heads": int(sum(record["component_count"] > 1 for record in records)),
        "label_null": {
            "mean": float(np.mean(label_null)),
            "samples": label_null,
            "upper_tail_p": upper_tail(observed, label_null),
        },
        "read_write_pairing_null": {
            "mean": float(np.mean(pairing_null)),
            "samples": pairing_null,
            "upper_tail_p": upper_tail(observed, pairing_null),
        },
        "exploratory_head_screen": {
            "heads_above_pairing_null_mean": int(
                sum(record["observed_minus_pairing_null_mean"] > 0 for record in records)
            ),
            "heads_above_all_pairing_nulls": int(
                sum(
                    record["pairing_null_upper_tail_p"]
                    <= 1.0 / (1.0 + len(pairing_null))
                    for record in records
                )
            ),
            "warning": (
                "exploratory only: 19 nulls give minimum p=0.05 and do not support "
                "correction across 48 heads"
            ),
        },
        "by_layer": layer_summary,
        "heads": records,
    }


def plot_report(report: dict[str, object], output: Path) -> None:
    splits = report["splits"]
    raw = report["raw_architecture_only_baseline"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    positions = np.arange(2)
    width = 0.34
    raw_values = [item["mean_confirmation_r2"] for item in raw]
    residual_values = [item["population_mean_confirmation_r2"] for item in splits]
    label_values = [item["label_null"]["mean"] for item in splits]
    axes[0].bar(positions - width / 2, raw_values, width, label="Before gain removal")
    axes[0].bar(positions + width / 2, residual_values, width, label="After gain removal")
    axes[0].scatter(positions + width / 2, label_values, color="black", marker="_", s=100)
    axes[0].set_xticks(positions, ("Parity 0", "Parity 1"))
    axes[0].set_ylabel("Held-out architectural R2")
    axes[0].set_title("A  Structure beyond spectral gain", loc="left")
    axes[0].legend()

    for parity, split in enumerate(splits):
        axes[1].hist(
            split["read_write_pairing_null"]["samples"],
            bins=10,
            alpha=0.55,
            label=f"Parity {parity} re-paired",
        )
        axes[1].axvline(
            split["population_mean_confirmation_r2"],
            color=f"C{parity}",
            linewidth=2,
        )
    axes[1].set_xlabel("Population mean residual R2")
    axes[1].set_ylabel("Null repetitions")
    axes[1].set_title("B  Does exact read/write pairing survive?", loc="left")
    axes[1].legend()

    layers = sorted(int(layer) for layer in splits[0]["by_layer"])
    for parity, split in enumerate(splits):
        values = [split["by_layer"][str(layer)]["mean_confirmation_r2"] for layer in layers]
        axes[2].plot(layers, values, marker="o", label=f"Parity {parity}")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Mean residual R2")
    axes[2].set_title("C  Localization after gain removal", loc="left")
    axes[2].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ov, metadata = load_factor_bundle(args.ov)
    qk, _ = load_factor_bundle(args.qk)
    decomposed = [factorized_singular_components(operator) for operator in ov]
    observed = []
    for parity in (0, 1):
        records, score, confirmations = fit_population(
            ov,
            qk,
            decomposed,
            discovery_parity=parity,
            gain_knots=args.gain_knots,
            maximum_components=args.maximum_components,
            seed=args.seed,
        )
        observed.append((records, score, confirmations))

    rng = np.random.default_rng(args.seed)
    pairing_nulls = [[], []]
    pairing_head_nulls = [
        [[] for _ in ov],
        [[] for _ in ov],
    ]
    for repetition in range(args.pairing_nulls):
        permutations = [rng.permutation(operator.d_head) for operator in ov]
        for parity in (0, 1):
            null_records, score, _ = fit_population(
                ov,
                qk,
                decomposed,
                discovery_parity=parity,
                gain_knots=args.gain_knots,
                maximum_components=args.maximum_components,
                seed=args.seed + 1000 * (repetition + 1),
                permutations=permutations,
            )
            pairing_nulls[parity].append(score)
            for index, record in enumerate(null_records):
                pairing_head_nulls[parity][index].append(record["confirmation_r2"])
        print(f"pairing null {repetition + 1}/{args.pairing_nulls}", flush=True)

    splits = [
        summarize_split(
            records,
            score,
            confirmations,
            pairing_nulls[parity],
            pairing_head_nulls[parity],
            label_repetitions=args.label_nulls,
            seed=args.seed + 10_000 * parity,
        )
        for parity, (records, score, confirmations) in enumerate(observed)
    ]
    first_heads = splits[0]["heads"]
    second_heads = splits[1]["heads"]
    reciprocal = []
    reciprocal_candidates = []
    for first, second in zip(first_heads, second_heads, strict=True):
        if first["component_count"] > 1 and second["component_count"] > 1:
            reciprocal.append(float(adjusted_rand_score(first["labels"], second["labels"])))
        if (
            first["pairing_null_upper_tail_p"] <= 1.0 / (1.0 + args.pairing_nulls)
            and second["pairing_null_upper_tail_p"] <= 1.0 / (1.0 + args.pairing_nulls)
        ):
            reciprocal_candidates.append({"layer": first["layer"], "head": first["head"]})

    raw_audit = json.loads(args.raw_audit.read_text(encoding="utf-8"))
    raw_baseline = [
        {
            "discovery_parity": split["discovery_parity"],
            "mean_confirmation_r2": split["population"]["architecture_only"][
                "mean_confirmation_r2"
            ],
        }
        for split in raw_audit["splits"]
    ]
    report = {
        "status": "OV architectural compartment test after nonlinear singular-gain removal",
        "model": metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": metadata.get("revision", "step143000"),
        "residualization": (
            f"cubic quantile spline of log normalized gain with {args.gain_knots} knots; "
            "fit independently to every architectural feature within every head"
        ),
        "raw_architecture_only_baseline": raw_baseline,
        "splits": splits,
        "reciprocal_stability": {
            "heads_multicomponent_in_both": len(reciprocal),
            "mean_adjusted_rand": float(np.mean(reciprocal)) if reciprocal else None,
            "median_adjusted_rand": float(np.median(reciprocal)) if reciprocal else None,
            "heads_adjusted_rand_at_least_0_5": int(
                np.sum(np.asarray(reciprocal) >= 0.5)
            ),
            "exploratory_heads_above_all_pairing_nulls_in_both_splits": (
                reciprocal_candidates
            ),
        },
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
