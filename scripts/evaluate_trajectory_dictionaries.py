"""Select QK, OV, and shared-code atom dictionaries without functional labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import DictionaryLearning, sparse_encode

from head_atlas.dictionary import (
    blocked_checkpoint_splits,
    cross_validated_reconstruction,
    grouped_splits,
    head_trajectory_groups,
    joint_view_coordinates,
)
from head_atlas.embedding import classical_mds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pythia70m_deduped_pilot.json")
    )
    parser.add_argument(
        "--qk-input",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/qk_trajectory_distances.npz"),
    )
    parser.add_argument(
        "--ov-input",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/ov_trajectory_distances.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/pythia-70m-deduped/dictionaries.json")
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument("--dictionary-alpha", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=500)
    return parser.parse_args()


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def select_dictionary(result: dict[str, object]) -> tuple[int, int, float]:
    candidates = []
    for component_label, record in result.items():
        for model, error in record["mean_relative_squared_error"].items():
            if model.startswith("dictionary_"):
                candidates.append(
                    (float(error), int(component_label), int(model.removeprefix("dictionary_")))
                )
    error, components, active = min(candidates)
    return components, active, error


def select_compact_dictionary(result: dict[str, object]) -> tuple[int, int, float]:
    """Choose the smallest preregistered sparse model that beats hard clusters."""

    for component_label in sorted(result, key=int):
        record = result[component_label]
        errors = record["mean_relative_squared_error"]
        for model in sorted(
            (name for name in errors if name.startswith("dictionary_")),
            key=lambda name: int(name.removeprefix("dictionary_")),
        ):
            if errors[model] < errors["kmeans"]:
                return (
                    int(component_label),
                    int(model.removeprefix("dictionary_")),
                    float(errors[model]),
                )
    raise RuntimeError("no sparse dictionary beats its matched hard-cluster baseline")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    qk = load_trajectory(args.qk_input)
    ov = load_trajectory(args.ov_input)
    metadata_keys = ("checkpoints", "checkpoint_values", "layers", "heads")
    for key in metadata_keys:
        if not np.array_equal(qk[key], ov[key]):
            raise ValueError(f"QK and OV trajectory metadata differ for {key}")

    qk_embedding = classical_mds(qk["distances"])
    ov_embedding = classical_mds(ov["distances"])
    qk_coordinates = np.asarray(qk_embedding["coordinates"], dtype=np.float64)
    ov_coordinates = np.asarray(ov_embedding["coordinates"], dtype=np.float64)
    views = {
        "QK": qk_coordinates,
        "OV": ov_coordinates,
        "JOINT": joint_view_coordinates(qk_coordinates, ov_coordinates),
    }
    minimum_step = int(config["dictionary_discovery_minimum_step"])
    discovery_mask = qk["checkpoint_values"] >= minimum_step
    discovery_indices = np.flatnonzero(discovery_mask)
    discovery_groups = head_trajectory_groups(
        qk["layers"][discovery_mask], qk["heads"][discovery_mask]
    )
    trajectory_splits = grouped_splits(
        discovery_groups, int(config["trajectory_validation_folds"])
    )
    temporal_splits = blocked_checkpoint_splits(
        qk["checkpoint_values"][discovery_mask],
        int(config["temporal_validation_blocks"]),
    )

    results = {}
    for view_name, all_coordinates in views.items():
        print(f"selecting {view_name} dictionary", flush=True)
        discovery_coordinates = all_coordinates[discovery_indices]
        trajectory_result = cross_validated_reconstruction(
            discovery_coordinates,
            trajectory_splits,
            config["dictionary_atom_counts"],
            config["active_atom_counts"],
            dictionary_alpha=args.dictionary_alpha,
            seed=int(config["random_seed"]),
            max_iter=args.max_iter,
        )
        selected_components, selected_active, selected_error = select_dictionary(
            trajectory_result
        )
        compact_components, compact_active, compact_error = select_compact_dictionary(
            trajectory_result
        )
        print(
            f"{view_name}: selected {selected_components} atoms, "
            f"{selected_active} active, relative error {selected_error:.4f}",
            flush=True,
        )
        temporal_result = cross_validated_reconstruction(
            discovery_coordinates,
            temporal_splits,
            [selected_components],
            [selected_active],
            dictionary_alpha=args.dictionary_alpha,
            seed=int(config["random_seed"]) + 100,
            max_iter=args.max_iter,
        )

        mean = np.mean(discovery_coordinates, axis=0, keepdims=True)
        model = DictionaryLearning(
            n_components=selected_components,
            alpha=args.dictionary_alpha,
            max_iter=args.max_iter,
            fit_algorithm="cd",
            random_state=int(config["random_seed"]),
        ).fit(discovery_coordinates - mean)
        codes = sparse_encode(
            all_coordinates - mean,
            model.components_,
            algorithm="omp",
            n_nonzero_coefs=selected_active,
        )
        artifact_path = args.artifact_root / f"{view_name.lower()}_dictionary.npz"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path,
            atoms=model.components_,
            codes=codes,
            coordinate_mean=mean,
            coordinates=all_coordinates,
            checkpoints=qk["checkpoints"],
            checkpoint_values=qk["checkpoint_values"],
            layers=qk["layers"],
            heads=qk["heads"],
            discovery_mask=discovery_mask,
            selected_active_atoms=np.asarray(selected_active),
            dictionary_alpha=np.asarray(args.dictionary_alpha),
        )
        compact_model = DictionaryLearning(
            n_components=compact_components,
            alpha=args.dictionary_alpha,
            max_iter=args.max_iter,
            fit_algorithm="cd",
            random_state=int(config["random_seed"]),
        ).fit(discovery_coordinates - mean)
        compact_codes = sparse_encode(
            all_coordinates - mean,
            compact_model.components_,
            algorithm="omp",
            n_nonzero_coefs=compact_active,
        )
        compact_artifact_path = (
            args.artifact_root / f"{view_name.lower()}_compact_dictionary.npz"
        )
        np.savez_compressed(
            compact_artifact_path,
            atoms=compact_model.components_,
            codes=compact_codes,
            coordinate_mean=mean,
            coordinates=all_coordinates,
            checkpoints=qk["checkpoints"],
            checkpoint_values=qk["checkpoint_values"],
            layers=qk["layers"],
            heads=qk["heads"],
            discovery_mask=discovery_mask,
            selected_active_atoms=np.asarray(compact_active),
            dictionary_alpha=np.asarray(args.dictionary_alpha),
        )
        results[view_name] = {
            "coordinate_dimensions": all_coordinates.shape[1],
            "selected_components": selected_components,
            "selected_active_atoms": selected_active,
            "selected_trajectory_relative_squared_error": selected_error,
            "trajectory_group_validation": trajectory_result,
            "blocked_temporal_validation": temporal_result,
            "artifact": str(artifact_path),
            "compact_selection": {
                "selection_rule": (
                    "smallest preregistered atom count, then smallest active count, "
                    "whose grouped error beats matched k-means"
                ),
                "components": compact_components,
                "active_atoms": compact_active,
                "trajectory_relative_squared_error": compact_error,
                "artifact": str(compact_artifact_path),
            },
        }

    report = {
        "analysis_status": "preregistered unsupervised pilot",
        "feature_space": "exact PCoA coordinates of normalized-Frobenius geometry",
        "joint_view": "equal-weight orthogonal product with one shared sparse code",
        "functional_labels_used": False,
        "dictionary_discovery_minimum_step": minimum_step,
        "discovery_observations": int(np.sum(discovery_mask)),
        "control_observations": int(np.sum(~discovery_mask)),
        "dictionary_alpha": args.dictionary_alpha,
        "max_iter": args.max_iter,
        "qk_negative_eigenvalue_mass_ratio": float(
            qk_embedding["negative_eigenvalue_mass_ratio"]
        ),
        "ov_negative_eigenvalue_mass_ratio": float(
            ov_embedding["negative_eigenvalue_mass_ratio"]
        ),
        "views": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved dictionary evaluation to {args.output}")


if __name__ == "__main__":
    main()
