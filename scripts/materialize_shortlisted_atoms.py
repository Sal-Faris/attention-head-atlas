"""Select stable temporal motifs and map them back to residual-stream matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_atlas.atoms import coordinate_atom_coefficients, materialize_operator_atoms
from head_atlas.factor_io import load_factor_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=Path("results/pythia-70m-deduped/dictionary_stability.json"),
    )
    parser.add_argument(
        "--emergence",
        type=Path,
        default=Path("results/pythia-70m-deduped/atom_emergence.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/shortlisted_atoms.json"),
    )
    parser.add_argument("--per-view", type=int, default=3)
    parser.add_argument("--profile", choices=("optimal", "compact"), default="optimal")
    parser.add_argument(
        "--reuse",
        type=Path,
        default=Path("results/pythia-70m-deduped/atom_reuse.json"),
    )
    return parser.parse_args()


def select_atoms(
    stability: dict,
    emergence: dict,
    reuse: dict,
    count: int,
) -> list[int]:
    stable = {
        record["atom"]: record
        for record in stability["per_atom"]
        if record["passes_0.10_advantage"]
    }
    temporal = {record["atom"]: record for record in emergence["atoms"]}
    reusable = {
        record["atom"] for record in reuse["atoms"] if record["passes_reuse_rule"]
    }
    priority_groups = [
        [
            atom
            for atom in stable
            if atom in reusable
            if temporal[atom]["significant_monotonic_trajectory"]
            and temporal[atom]["spearman_training_correlation"] > 0
        ],
        [
            atom
            for atom in stable
            if atom in reusable
            if temporal[atom]["significant_monotonic_trajectory"]
            and temporal[atom]["spearman_training_correlation"] < 0
        ],
        [atom for atom in stable if atom in reusable],
        list(stable),
    ]
    selected = []
    for group in priority_groups:
        ordered = sorted(
            group,
            key=lambda atom: stable[atom]["trajectory_bootstrap_similarity_mean"],
            reverse=True,
        )
        for atom in ordered:
            if atom not in selected:
                selected.append(atom)
            if len(selected) == count:
                return selected
    return selected


def load_population(manifest: dict, kind: str) -> list:
    operators = []
    for record in manifest["records"]:
        checkpoint_operators, _ = load_factor_bundle(record["factors"][kind]["path"])
        operators.extend(checkpoint_operators)
    return operators


def matrix_summary(matrix: np.ndarray, kind: str) -> dict:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    squared = singular_values**2
    probabilities = squared / max(float(np.sum(squared)), 1e-12)
    nonzero = probabilities > 0
    effective_rank = float(np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))))
    result = {
        "frobenius_norm": float(np.linalg.norm(matrix)),
        "spectral_norm": float(singular_values[0]),
        "effective_rank": effective_rank,
        "top_10_singular_values": singular_values[:10].tolist(),
        "trace": float(np.trace(matrix)),
    }
    if kind == "QK":
        result["antisymmetric_fraction"] = float(
            np.linalg.norm(matrix - matrix.T) / (2.0 * np.linalg.norm(matrix))
        )
    return result


def exemplar_heads(artifact: dict[str, np.ndarray], atom: int, count: int = 5) -> list[dict]:
    final_step = int(np.max(artifact["checkpoint_values"]))
    final_indices = np.flatnonzero(artifact["checkpoint_values"] == final_step)
    coefficients = artifact["codes"][final_indices, atom]
    order = np.argsort(np.abs(coefficients))[::-1][:count]
    return [
        {
            "checkpoint": final_step,
            "layer": int(artifact["layers"][final_indices[index]]),
            "head": int(artifact["heads"][final_indices[index]]),
            "coefficient": float(coefficients[index]),
        }
        for index in order
    ]


def load_dictionary(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def main() -> None:
    args = parse_args()
    if args.per_view < 1:
        raise ValueError("per-view shortlist size must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    stability = json.loads(args.stability.read_text(encoding="utf-8"))
    emergence = json.loads(args.emergence.read_text(encoding="utf-8"))
    reuse = json.loads(args.reuse.read_text(encoding="utf-8"))
    artifacts = {
        view: load_dictionary(
            args.artifact_root
            / (
                f"{view.lower()}_compact_dictionary.npz"
                if args.profile == "compact"
                else f"{view.lower()}_dictionary.npz"
            )
        )
        for view in ("QK", "OV", "JOINT")
    }
    selected = {
        view: select_atoms(
            stability["views"][view],
            emergence["views"][view],
            reuse["views"][view][args.profile],
            args.per_view,
        )
        for view in ("QK", "OV", "JOINT")
    }

    qk_dimensions = artifacts["QK"]["coordinates"].shape[1]
    qk_coefficients = coordinate_atom_coefficients(
        artifacts["QK"]["coordinates"], artifacts["QK"]["atoms"][selected["QK"]]
    )
    joint_qk_coefficients = coordinate_atom_coefficients(
        artifacts["JOINT"]["coordinates"][:, :qk_dimensions],
        artifacts["JOINT"]["atoms"][selected["JOINT"], :qk_dimensions],
    )
    qk_operators = load_population(manifest, "QK")
    qk_matrices = materialize_operator_atoms(
        np.concatenate([qk_coefficients, joint_qk_coefficients], axis=1), qk_operators
    )
    del qk_operators

    ov_coefficients = coordinate_atom_coefficients(
        artifacts["OV"]["coordinates"], artifacts["OV"]["atoms"][selected["OV"]]
    )
    joint_ov_coefficients = coordinate_atom_coefficients(
        artifacts["JOINT"]["coordinates"][:, qk_dimensions:],
        artifacts["JOINT"]["atoms"][selected["JOINT"], qk_dimensions:],
    )
    ov_operators = load_population(manifest, "OV")
    ov_matrices = materialize_operator_atoms(
        np.concatenate([ov_coefficients, joint_ov_coefficients], axis=1), ov_operators
    )
    del ov_operators

    count = args.per_view
    matrix_groups = {
        "QK": qk_matrices[:count],
        "OV": ov_matrices[:count],
        "JOINT_QK": qk_matrices[count:],
        "JOINT_OV": ov_matrices[count:],
    }
    matrix_artifact = args.artifact_root / f"{args.profile}_shortlisted_atom_matrices.npz"
    np.savez_compressed(
        matrix_artifact,
        qk_atom_ids=np.asarray(selected["QK"]),
        qk_matrices=matrix_groups["QK"],
        ov_atom_ids=np.asarray(selected["OV"]),
        ov_matrices=matrix_groups["OV"],
        joint_atom_ids=np.asarray(selected["JOINT"]),
        joint_qk_matrices=matrix_groups["JOINT_QK"],
        joint_ov_matrices=matrix_groups["JOINT_OV"],
    )

    report_views = {}
    for view in ("QK", "OV", "JOINT"):
        stability_by_atom = {
            record["atom"]: record for record in stability["views"][view]["per_atom"]
        }
        emergence_by_atom = {
            record["atom"]: record for record in emergence["views"][view]["atoms"]
        }
        records = []
        for index, atom in enumerate(selected[view]):
            if view == "JOINT":
                matrices = {
                    "QK": matrix_summary(matrix_groups["JOINT_QK"][index], "QK"),
                    "OV": matrix_summary(matrix_groups["JOINT_OV"][index], "OV"),
                }
            else:
                matrices = matrix_summary(matrix_groups[view][index], view)
            records.append(
                {
                    "atom": atom,
                    "selection_priority": index + 1,
                    "stability": stability_by_atom[atom],
                    "emergence": emergence_by_atom[atom],
                    "matrix_summary": matrices,
                    "final_checkpoint_exemplar_heads": exemplar_heads(
                        artifacts[view], atom
                    ),
                }
            )
        report_views[view] = records

    report = {
        "analysis_status": "label-free atom shortlist",
        "profile": args.profile,
        "selection_rule": (
            "reusable and significant increasing trajectories first, then reusable "
            "significant decreasing trajectories, then other reusable atoms; within "
            "tier sort by trajectory-bootstrap stability"
        ),
        "per_view": args.per_view,
        "matrix_artifact": str(matrix_artifact),
        "views": report_views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"selected atoms {selected} and saved report to {args.output}")


if __name__ == "__main__":
    main()
