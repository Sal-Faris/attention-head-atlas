"""Audit architectural OV blocks against baselines at their actual selected cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pilot_direct_ov_restricted_maps import (
    full_svd_energy,
    projected_dense_low_rank_energy,
    unstructured_sparse_energy,
)

from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_frobenius_norm
from head_atlas.restricted_maps import architectural_operator_bases, project_operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_ov_restricted_map_pilot_v1.json"),
    )
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
    parser.add_argument("--budget", type=float, default=4_000.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_ov_selected_cost_audit_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.pilot.read_text(encoding="utf-8"))
    ov, _ = load_factor_bundle(args.ov)
    qk, _ = load_factor_bundle(args.qk)
    by_location = {(operator.layer, operator.head): operator for operator in ov}
    audited_splits = []
    for split in report["anchor_splits"]:
        parity = int(split["anchor_head_parity"])
        dimension = int(split["basis_dimension"])
        bases = {
            layer: architectural_operator_bases(
                ov,
                qk,
                target_layer=layer,
                anchor_head_parity=parity,
                dimension=dimension,
            )
            for layer in split["target_layers"]
        }
        rows = []
        for head in split["heads"]:
            record = next(item for item in head["budgets"] if item["scalar_budget"] == args.budget)
            operator = by_location[(head["layer"], head["head"])]
            total = max(factorized_frobenius_norm(operator) ** 2, 1e-20)
            read, write = bases[head["layer"]]
            coefficients = project_operator(operator, read, write) / np.sqrt(total)
            selected_cost = float(sum(block["scalar_cost"] for block in record["selected_blocks"]))
            rows.append(
                {
                    "layer": head["layer"],
                    "head": head["head"],
                    "selected_block_cost": selected_cost,
                    "restricted_blocks": record["restricted_blocks_energy_fraction"],
                    "cost_matched_unstructured_sparse": unstructured_sparse_energy(
                        coefficients, selected_cost
                    ),
                    "cost_matched_projected_dense_low_rank": projected_dense_low_rank_energy(
                        coefficients, selected_cost
                    ),
                    "cost_matched_full_svd": full_svd_energy(operator, selected_cost),
                }
            )
        keys = (
            "restricted_blocks",
            "cost_matched_unstructured_sparse",
            "cost_matched_projected_dense_low_rank",
            "cost_matched_full_svd",
        )
        means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
        audited_splits.append(
            {
                "anchor_head_parity": parity,
                "budget_cap": args.budget,
                "mean_actual_selected_block_cost": float(
                    np.mean([row["selected_block_cost"] for row in rows])
                ),
                "mean_energy_fraction_at_per_head_selected_cost": means,
                "fraction_of_heads_where_blocks_beat_sparse": float(
                    np.mean(
                        [
                            row["restricted_blocks"] > row["cost_matched_unstructured_sparse"]
                            for row in rows
                        ]
                    )
                ),
                "fraction_of_heads_where_blocks_beat_dense_low_rank": float(
                    np.mean(
                        [
                            row["restricted_blocks"] > row["cost_matched_projected_dense_low_rank"]
                            for row in rows
                        ]
                    )
                ),
                "heads": rows,
            }
        )
    audit = {
        "status": "selected-cost audit of architectural OV restricted maps",
        "source_pilot": str(args.pilot),
        "reason": (
            "The block frontier can use less than its budget cap. This diagnostic gives every "
            "alternative each head's actual selected block cost; the main rate-distortion gate "
            "still uses the common budget cap."
        ),
        "splits": audited_splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"saved selected-cost audit to {args.output}")


if __name__ == "__main__":
    main()
