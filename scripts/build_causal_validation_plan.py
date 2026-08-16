"""Build label-free, layer-matched intervention targets for shortlisted atoms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stability",
        type=Path,
        default=Path("results/pythia-70m-deduped/compact_dictionary_stability.json"),
    )
    parser.add_argument(
        "--emergence",
        type=Path,
        default=Path("results/pythia-70m-deduped/compact_atom_emergence.json"),
    )
    parser.add_argument(
        "--reuse", type=Path, default=Path("results/pythia-70m-deduped/atom_reuse.json")
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/causal_validation_plan.json"),
    )
    parser.add_argument("--targets-per-atom", type=int, default=3)
    return parser.parse_args()


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def main() -> None:
    args = parse_args()
    stability = json.loads(args.stability.read_text(encoding="utf-8"))
    emergence = json.loads(args.emergence.read_text(encoding="utf-8"))
    reuse = json.loads(args.reuse.read_text(encoding="utf-8"))
    candidates = []
    for view in ("QK", "OV", "JOINT"):
        stability_by_atom = {
            record["atom"]: record for record in stability["views"][view]["per_atom"]
        }
        emergence_by_atom = {
            record["atom"]: record for record in emergence["views"][view]["atoms"]
        }
        reuse_by_atom = {
            record["atom"]: record for record in reuse["views"][view]["compact"]["atoms"]
        }
        artifact = load_artifact(
            args.artifact_root / f"{view.lower()}_compact_dictionary.npz"
        )
        final_step = int(np.max(artifact["checkpoint_values"]))
        final_indices = np.flatnonzero(artifact["checkpoint_values"] == final_step)
        for atom, stability_record in stability_by_atom.items():
            emergence_record = emergence_by_atom[atom]
            reuse_record = reuse_by_atom[atom]
            if not (
                stability_record["passes_0.10_advantage"]
                and emergence_record["significant_monotonic_trajectory"]
                and reuse_record["passes_cross_layer_reuse_rule"]
            ):
                continue
            coefficients = artifact["codes"][final_indices, atom]
            target_order = np.argsort(np.abs(coefficients))[::-1]
            targets = []
            selected_target_indices = target_order[: args.targets_per_atom]
            target_locations = {
                (
                    int(artifact["layers"][final_indices[local_index]]),
                    int(artifact["heads"][final_indices[local_index]]),
                )
                for local_index in selected_target_indices
            }
            for local_index in selected_target_indices:
                global_index = final_indices[local_index]
                layer = int(artifact["layers"][global_index])
                head = int(artifact["heads"][global_index])
                same_layer = final_indices[artifact["layers"][final_indices] == layer]
                control_order = same_layer[
                    np.argsort(np.abs(artifact["codes"][same_layer, atom]))
                ]
                control = next(
                    index
                    for index in control_order
                    if (layer, int(artifact["heads"][index])) not in target_locations
                )
                targets.append(
                    {
                        "target": {
                            "layer": layer,
                            "head": head,
                            "atom_coefficient": float(coefficients[local_index]),
                        },
                        "same_layer_low_loading_control": {
                            "layer": layer,
                            "head": int(artifact["heads"][control]),
                            "atom_coefficient": float(artifact["codes"][control, atom]),
                        },
                    }
                )
            candidates.append(
                {
                    "view": view,
                    "atom": atom,
                    "training_correlation": emergence_record[
                        "spearman_training_correlation"
                    ],
                    "bootstrap_similarity": stability_record[
                        "trajectory_bootstrap_similarity_mean"
                    ],
                    "effective_trajectory_participation": reuse_record[
                        "effective_trajectory_participation"
                    ],
                    "effective_layer_participation": reuse_record[
                        "effective_layer_participation"
                    ],
                    "intervention_pairs": targets,
                }
            )

    plan = {
        "status": "planned, not executed",
        "checkpoint": "step143000",
        "candidate_rule": (
            "compact atom passes stability, FDR-controlled temporal change, trajectory reuse, "
            "and cross-layer reuse; no functional labels"
        ),
        "interventions": [
            "zero head result at every token position",
            "replace head result with its clean-run mean",
            "patch clean head result into a corrupted sequence",
        ],
        "layer_matched_control": "same-layer head with minimal absolute candidate-atom loading",
        "primary_metrics": [
            "change in next-token cross-entropy",
            "KL divergence from clean next-token distribution",
            "recovery under clean-to-corrupted activation patching",
        ],
        "multiple_testing": "Benjamini-Hochberg correction across candidate heads and metrics",
        "interpretation_gate": (
            "name an atom only if candidate interventions exceed layer-matched controls and "
            "the effect recurs across at least two independent heads"
        ),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved {len(candidates)} causal candidates to {args.output}")


if __name__ == "__main__":
    main()
