"""Check whether the gain-residualized OV result depends on spline flexibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_gain_residualized_ov_compartments import fit_population

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
    parser.add_argument("--gain-knots", type=int, nargs="+", default=[3, 4, 5, 6, 8])
    parser.add_argument("--maximum-components", type=int, default=6)
    parser.add_argument("--seed", type=int, default=16180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/pythia-70m-deduped/gain_residualization_sensitivity_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ov, metadata = load_factor_bundle(args.ov)
    qk, _ = load_factor_bundle(args.qk)
    decomposed = [factorized_singular_components(operator) for operator in ov]
    results = []
    for knots in args.gain_knots:
        split_results = []
        for parity in (0, 1):
            records, score, _ = fit_population(
                ov,
                qk,
                decomposed,
                discovery_parity=parity,
                gain_knots=knots,
                maximum_components=args.maximum_components,
                seed=args.seed,
            )
            split_results.append(
                {
                    "discovery_parity": parity,
                    "population_mean_confirmation_r2": score,
                    "mean_component_count": sum(
                        record["component_count"] for record in records
                    )
                    / len(records),
                    "multicomponent_heads": sum(
                        record["component_count"] > 1 for record in records
                    ),
                }
            )
        results.append({"gain_knots": knots, "splits": split_results})
        print(f"finished {knots} gain knots", flush=True)

    report = {
        "status": "gain-residualization smoothness sensitivity",
        "model": metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": metadata.get("revision", "step143000"),
        "results": results,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved sensitivity audit to {args.output}")


if __name__ == "__main__":
    main()
