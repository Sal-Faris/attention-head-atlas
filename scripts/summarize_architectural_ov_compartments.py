"""Combine reciprocal architectural-anchor compartment splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parity-zero",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_ov_compartments_v1.json"),
    )
    parser.add_argument(
        "--parity-one",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/architectural_ov_compartments_parity1_v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/architectural_ov_compartments_summary_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [
        json.loads(args.parity_zero.read_text(encoding="utf-8")),
        json.loads(args.parity_one.read_text(encoding="utf-8")),
    ]
    first, second = reports
    first_locations = [(item["layer"], item["head"]) for item in first["heads"]]
    second_locations = [(item["layer"], item["head"]) for item in second["heads"]]
    if first_locations != second_locations:
        raise ValueError("reciprocal reports do not align")

    both_multicomponent = []
    all_agreements = []
    for first_head, second_head in zip(first["heads"], second["heads"], strict=True):
        agreement = float(adjusted_rand_score(first_head["labels"], second_head["labels"]))
        all_agreements.append(agreement)
        if first_head["component_count"] > 1 and second_head["component_count"] > 1:
            both_multicomponent.append(agreement)

    module_dimensions = []
    module_energies = []
    for report in reports:
        for head in report["heads"]:
            if head["component_count"] > 1:
                module_dimensions.extend(head["component_dimensions"])
                module_energies.extend(head["component_energy_fractions"])

    summary = {
        "status": "reciprocal-anchor summary of architecture-connected OV compartments",
        "model": first["model"],
        "revision": first["revision"],
        "split_results": [
            {
                "discovery_parity": parity,
                "population_mean_confirmation_r2": report[
                    "population_mean_confirmation_r2"
                ],
                "multicomponent_heads": int(
                    sum(item["component_count"] > 1 for item in report["heads"])
                ),
                "mean_component_count": report["component_count_summary"]["mean"],
                "pairing_null_mean": report["read_write_pairing_null"]["mean"],
                "pairing_null_p": report["read_write_pairing_null"]["upper_tail_p_value"],
                "label_null_mean": report["held_out_anchor_label_null"]["mean"],
                "label_null_p": report["held_out_anchor_label_null"]["upper_tail_p_value"],
            }
            for parity, report in enumerate(reports)
        ],
        "reciprocal_stability": {
            "heads_multicomponent_in_both_splits": len(both_multicomponent),
            "mean_adjusted_rand_among_those_heads": float(np.mean(both_multicomponent)),
            "median_adjusted_rand_among_those_heads": float(np.median(both_multicomponent)),
            "heads_with_adjusted_rand_at_least_0_5": int(
                np.sum(np.asarray(both_multicomponent) >= 0.5)
            ),
            "same_component_count_heads": int(
                sum(
                    first_head["component_count"] == second_head["component_count"]
                    for first_head, second_head in zip(
                        first["heads"], second["heads"], strict=True
                    )
                )
            ),
            "all_head_adjusted_rand": all_agreements,
        },
        "module_summary_across_both_splits": {
            "multicomponent_module_count": len(module_dimensions),
            "median_dimension": float(np.median(module_dimensions)),
            "minimum_dimension": int(np.min(module_dimensions)),
            "maximum_dimension": int(np.max(module_dimensions)),
            "median_operator_energy_fraction": float(np.median(module_energies)),
        },
        "scope_warning": (
            "all singular channels are partitioned, so this analysis measures held-out "
            "architectural coherence rather than variance explained by accepted compartments"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved reciprocal summary to {args.output}")


if __name__ == "__main__":
    main()
