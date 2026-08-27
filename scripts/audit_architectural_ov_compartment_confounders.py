"""Test whether OV compartments reduce to singular gain or channel rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from analyze_architectural_ov_compartments import fingerprints
from sklearn.metrics import adjusted_rand_score

from head_atlas.architectural_compartments import confirmation_r2, fit_compartments
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
    parser.add_argument("--label-nulls", type=int, default=99)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/architectural_ov_compartment_confounders_v1.json"
        ),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/architectural_ov_compartment_confounders_v1.png"
        ),
    )
    return parser.parse_args()


def plot_report(report: dict[str, object], output: Path) -> None:
    """Plot predictive strength and the remaining gain confound."""

    splits = report["splits"]
    variants = ("full", "architecture_only", "gain_only", "rank_only")
    labels = ("Architecture + gain", "Architecture only", "Gain only", "Rank only")
    positions = np.arange(len(variants), dtype=np.float64)
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for parity, split in enumerate(splits):
        values = [split["population"][name]["mean_confirmation_r2"] for name in variants]
        nulls = [split["population"][name]["label_null_mean"] for name in variants]
        offset = (parity - 0.5) * width
        axes[0].bar(
            positions + offset,
            values,
            width,
            label=f"Discovery parity {parity}",
            alpha=0.8,
        )
        axes[0].scatter(
            positions + offset,
            nulls,
            color="black",
            marker="_",
            s=90,
            zorder=3,
        )
    axes[0].set_xticks(positions, labels, rotation=18, ha="right")
    axes[0].set_ylabel("Held-out architectural R2")
    axes[0].set_title("A  Gain explains most, not all, of the signal", loc="left")
    axes[0].legend()

    agreements = [
        split["architecture_gain_relationship"]["mean_adjusted_rand_among_those_heads"]
        for split in splits
    ]
    gain_scores = [
        split["architecture_gain_relationship"]["mean_gain_r2_from_architecture_only_labels"]
        for split in splits
    ]
    relation_positions = np.arange(2, dtype=np.float64)
    axes[1].bar(relation_positions - width / 2, agreements, width, label="Adjusted Rand")
    axes[1].bar(
        relation_positions + width / 2,
        gain_scores,
        width,
        label="Gain R2 from architecture labels",
    )
    axes[1].set_xticks(relation_positions, ("Parity 0", "Parity 1"))
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Agreement / explained fraction")
    axes[1].set_title("B  Architectural and gain groups are related", loc="left")
    axes[1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ov, metadata = load_factor_bundle(args.ov)
    qk, _ = load_factor_bundle(args.qk)
    decomposed = [factorized_singular_components(operator) for operator in ov]
    variants = ("full", "architecture_only", "gain_only", "rank_only")
    report: dict[str, object] = {
        "status": "singular-spectrum confound audit for architectural OV compartments",
        "model": metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": metadata.get("revision", "step143000"),
        "variant_definitions": {
            "full": "architectural overlaps plus normalized singular gain",
            "architecture_only": "architectural overlaps without singular gain",
            "gain_only": "normalized singular gain only",
            "rank_only": "singular-value order only",
        },
        "splits": [],
    }

    for parity in (0, 1):
        scores = {name: [] for name in variants}
        counts = {name: [] for name in variants}
        architecture_gain_agreements = []
        architecture_label_gain_scores = []
        confirmations = []
        fitted_labels = {name: [] for name in variants}
        head_records = []
        for index, operator in enumerate(ov):
            discovery, confirmation, gain = fingerprints(
                index, ov, qk, decomposed, discovery_parity=parity
            )
            candidate_features = {
                "full": discovery,
                "architecture_only": discovery[:, :-1],
                "gain_only": gain[:, None],
                "rank_only": np.arange(len(gain), dtype=np.float64)[:, None],
            }
            confirmations.append(confirmation)
            record = {"layer": operator.layer, "head": operator.head, "variants": {}}
            fits = {}
            for name, features in candidate_features.items():
                fit = fit_compartments(
                    features,
                    confirmation,
                    maximum_components=args.maximum_components,
                    seed=args.seed + index,
                )
                scores[name].append(fit.confirmation_r2)
                counts[name].append(fit.component_count)
                fits[name] = fit
                fitted_labels[name].append(fit.labels)
                record["variants"][name] = {
                    "component_count": fit.component_count,
                    "confirmation_r2": fit.confirmation_r2,
                }
            architecture_label_gain_scores.append(
                confirmation_r2(gain[:, None], fits["architecture_only"].labels)
            )
            if (
                fits["architecture_only"].component_count > 1
                and fits["gain_only"].component_count > 1
            ):
                agreement = float(
                    adjusted_rand_score(
                        fits["architecture_only"].labels, fits["gain_only"].labels
                    )
                )
                architecture_gain_agreements.append(agreement)
                record["architecture_gain_adjusted_rand"] = agreement
            head_records.append(record)

        rng = np.random.default_rng(args.seed + 100_000 * parity)
        label_nulls = {name: [] for name in variants}
        for _ in range(args.label_nulls):
            for name in variants:
                shuffled_scores = [
                    confirmation_r2(confirmation, rng.permutation(labels))
                    for confirmation, labels in zip(
                        confirmations, fitted_labels[name], strict=True
                    )
                ]
                label_nulls[name].append(float(np.mean(shuffled_scores)))
        report["splits"].append(
            {
                "discovery_parity": parity,
                "population": {
                    name: {
                        "mean_confirmation_r2": float(np.mean(scores[name])),
                        "mean_component_count": float(np.mean(counts[name])),
                        "multicomponent_heads": int(np.sum(np.asarray(counts[name]) > 1)),
                        "label_null_mean": float(np.mean(label_nulls[name])),
                        "label_null_upper_tail_p": float(
                            (
                                1
                                + np.sum(
                                    np.asarray(label_nulls[name])
                                    >= np.mean(scores[name])
                                )
                            )
                            / (1 + args.label_nulls)
                        ),
                    }
                    for name in variants
                },
                "architecture_gain_relationship": {
                    "heads_multicomponent_in_both": len(architecture_gain_agreements),
                    "mean_adjusted_rand_among_those_heads": (
                        float(np.mean(architecture_gain_agreements))
                        if architecture_gain_agreements
                        else None
                    ),
                    "median_adjusted_rand_among_those_heads": (
                        float(np.median(architecture_gain_agreements))
                        if architecture_gain_agreements
                        else None
                    ),
                    "mean_gain_r2_from_architecture_only_labels": float(
                        np.mean(architecture_label_gain_scores)
                    ),
                },
                "heads": head_records,
            }
        )
        print(f"finished discovery parity {parity}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved confound audit to {args.output}")
    print(f"saved confound figure to {args.figure}")


if __name__ == "__main__":
    main()
