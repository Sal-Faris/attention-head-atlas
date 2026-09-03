"""Test atom structure after removing each checkpoint-by-layer population mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import DictionaryLearning, sparse_encode

from head_atlas.dictionary import (
    cross_validated_reconstruction,
    grouped_splits,
    head_trajectory_groups,
    joint_view_coordinates,
    residualize_group_means,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/pythia70m_deduped_pilot.json")
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/layer_residual_dictionaries.json"),
    )
    parser.add_argument("--dictionary-alpha", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=500)
    return parser.parse_args()


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def compact_selection(result: dict[str, object]) -> tuple[int, int, float]:
    for component_label in sorted(result, key=int):
        errors = result[component_label]["mean_relative_squared_error"]
        dictionary_models = sorted(
            (model for model in errors if model.startswith("dictionary_")),
            key=lambda model: int(model.removeprefix("dictionary_")),
        )
        for model in dictionary_models:
            if errors[model] < errors["kmeans"]:
                return (
                    int(component_label),
                    int(model.removeprefix("dictionary_")),
                    float(errors[model]),
                )
    raise RuntimeError("no residual dictionary beats its hard-cluster baseline")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    qk_artifact = load_artifact(args.artifact_root / "qk_dictionary.npz")
    ov_artifact = load_artifact(args.artifact_root / "ov_dictionary.npz")
    for key in ("checkpoint_values", "layers", "heads", "discovery_mask"):
        if not np.array_equal(qk_artifact[key], ov_artifact[key]):
            raise ValueError(f"QK and OV metadata differ for {key}")
    checkpoint_values = qk_artifact["checkpoint_values"]
    layers = qk_artifact["layers"]
    heads = qk_artifact["heads"]
    checkpoint_layer_groups = checkpoint_values * (int(np.max(layers)) + 1) + layers
    raw_views = {
        "QK": np.asarray(qk_artifact["coordinates"], dtype=np.float64),
        "OV": np.asarray(ov_artifact["coordinates"], dtype=np.float64),
    }
    residual_views = {
        view: residualize_group_means(coordinates, checkpoint_layer_groups)
        for view, coordinates in raw_views.items()
    }
    residual_views["JOINT"] = joint_view_coordinates(
        residual_views["QK"], residual_views["OV"]
    )
    discovery_mask = np.asarray(qk_artifact["discovery_mask"], dtype=bool)
    discovery_groups = head_trajectory_groups(
        layers[discovery_mask], heads[discovery_mask]
    )
    splits = grouped_splits(
        discovery_groups, int(config["trajectory_validation_folds"])
    )

    results = {}
    for view, coordinates in residual_views.items():
        print(f"evaluating layer-residual {view}", flush=True)
        discovery_coordinates = coordinates[discovery_mask]
        validation = cross_validated_reconstruction(
            discovery_coordinates,
            splits,
            config["dictionary_atom_counts"],
            config["active_atom_counts"],
            dictionary_alpha=args.dictionary_alpha,
            seed=int(config["random_seed"]),
            max_iter=args.max_iter,
        )
        components, active_atoms, relative_error = compact_selection(validation)
        mean = np.mean(discovery_coordinates, axis=0, keepdims=True)
        model = DictionaryLearning(
            n_components=components,
            alpha=args.dictionary_alpha,
            max_iter=args.max_iter,
            fit_algorithm="cd",
            random_state=int(config["random_seed"]),
        ).fit(discovery_coordinates - mean)
        codes = sparse_encode(
            coordinates - mean,
            model.components_,
            algorithm="omp",
            n_nonzero_coefs=active_atoms,
        )
        artifact_path = args.artifact_root / f"{view.lower()}_residual_compact_dictionary.npz"
        np.savez_compressed(
            artifact_path,
            atoms=model.components_,
            codes=codes,
            coordinate_mean=mean,
            coordinates=coordinates,
            checkpoints=qk_artifact["checkpoints"],
            checkpoint_values=checkpoint_values,
            layers=layers,
            heads=heads,
            discovery_mask=discovery_mask,
            selected_active_atoms=np.asarray(active_atoms),
            dictionary_alpha=np.asarray(args.dictionary_alpha),
            residualization=np.asarray("checkpoint_by_layer_mean"),
        )
        raw = (
            joint_view_coordinates(raw_views["QK"], raw_views["OV"])
            if view == "JOINT"
            else raw_views[view]
        )
        raw_centered = raw - np.mean(raw, axis=0, keepdims=True)
        results[view] = {
            "variance_fraction_after_checkpoint_layer_residualization": float(
                np.sum(coordinates**2) / np.sum(raw_centered**2)
            ),
            "compact_components": components,
            "compact_active_atoms": active_atoms,
            "compact_trajectory_relative_squared_error": relative_error,
            "validation": validation,
            "artifact": str(artifact_path),
        }
        print(
            f"{view}: {components} atoms/{active_atoms} active, error={relative_error:.4f}",
            flush=True,
        )

    report = {
        "analysis_status": "exploratory layer-confound diagnosis",
        "residualization": "subtract each checkpoint-by-layer coordinate centroid",
        "functional_labels_used": False,
        "views": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved layer-residual evaluation to {args.output}")


if __name__ == "__main__":
    main()
