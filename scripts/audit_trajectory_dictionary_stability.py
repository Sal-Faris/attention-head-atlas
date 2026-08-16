"""Audit selected Pythia dictionaries across seeds and trajectory bootstraps."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import DictionaryLearning

from head_atlas.dictionary import head_trajectory_groups
from head_atlas.motifs import matched_atom_similarities, matched_atom_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("results/pythia-70m-deduped/dictionaries.json"),
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/pythia-70m-deduped")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/dictionary_stability.json"),
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def fit_dictionary(
    coordinates: np.ndarray,
    components: int,
    alpha: float,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    model = DictionaryLearning(
        n_components=components,
        alpha=alpha,
        max_iter=max_iter,
        fit_algorithm="cd",
        random_state=seed,
    ).fit(centered)
    return model.components_


def pairwise_similarities(dictionaries: list[np.ndarray]) -> list[float]:
    return [
        matched_atom_similarity(first, second)
        for first, second in itertools.combinations(dictionaries, 2)
    ]


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    if args.repetitions < 2:
        raise ValueError("repetitions must be at least two")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    results = {}
    for view_name, selection in evaluation["views"].items():
        artifact_path = args.artifact_root / f"{view_name.lower()}_dictionary.npz"
        with np.load(artifact_path, allow_pickle=False) as artifact:
            reference_atoms = np.asarray(artifact["atoms"], dtype=np.float64)
            coordinates = np.asarray(artifact["coordinates"], dtype=np.float64)
            discovery_mask = np.asarray(artifact["discovery_mask"], dtype=bool)
            layers = np.asarray(artifact["layers"], dtype=np.int64)[discovery_mask]
            heads = np.asarray(artifact["heads"], dtype=np.int64)[discovery_mask]
        discovery_coordinates = coordinates[discovery_mask]
        groups = head_trajectory_groups(layers, heads)
        unique_groups = np.unique(groups)
        components = int(selection["selected_components"])
        alpha = float(evaluation["dictionary_alpha"])

        print(f"auditing {view_name} stability", flush=True)
        initialized = [
            fit_dictionary(
                discovery_coordinates,
                components,
                alpha,
                args.max_iter,
                args.seed + repetition,
            )
            for repetition in range(args.repetitions)
        ]
        bootstrapped = []
        for repetition in range(args.repetitions):
            sampled_groups = rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )
            sampled_indices = np.concatenate(
                [np.flatnonzero(groups == group) for group in sampled_groups]
            )
            bootstrapped.append(
                fit_dictionary(
                    discovery_coordinates[sampled_indices],
                    components,
                    alpha,
                    args.max_iter,
                    args.seed + 1000 + repetition,
                )
            )
        random_similarities = []
        random_atom_matches = []
        for _ in range(max(20, args.repetitions)):
            first = rng.standard_normal((components, discovery_coordinates.shape[1]))
            second = rng.standard_normal((components, discovery_coordinates.shape[1]))
            random_similarities.append(matched_atom_similarity(first, second))
            random_atom_matches.append(matched_atom_similarities(reference_atoms, first))

        initialization = summary(pairwise_similarities(initialized))
        bootstrap = summary(pairwise_similarities(bootstrapped))
        random_baseline = summary(random_similarities)
        bootstrap_atom_matches = np.stack(
            [matched_atom_similarities(reference_atoms, atoms) for atoms in bootstrapped]
        )
        initialization_atom_matches = np.stack(
            [matched_atom_similarities(reference_atoms, atoms) for atoms in initialized]
        )
        random_atom_matches_array = np.stack(random_atom_matches)
        per_atom = []
        for atom in range(components):
            bootstrap_mean = float(np.mean(bootstrap_atom_matches[:, atom]))
            random_mean = float(np.mean(random_atom_matches_array[:, atom]))
            per_atom.append(
                {
                    "atom": atom,
                    "initialization_similarity_mean": float(
                        np.mean(initialization_atom_matches[:, atom])
                    ),
                    "trajectory_bootstrap_similarity_mean": bootstrap_mean,
                    "trajectory_bootstrap_similarity_standard_deviation": float(
                        np.std(bootstrap_atom_matches[:, atom])
                    ),
                    "random_match_similarity_mean": random_mean,
                    "bootstrap_advantage_over_random": bootstrap_mean - random_mean,
                    "passes_0.10_advantage": bootstrap_mean - random_mean >= 0.10,
                }
            )
        results[view_name] = {
            "components": components,
            "trajectory_count": len(unique_groups),
            "initialization_stability": initialization,
            "trajectory_bootstrap_stability": bootstrap,
            "random_dictionary_baseline": random_baseline,
            "bootstrap_advantage_over_random": (
                bootstrap["mean"] - random_baseline["mean"]
            ),
            "passes_preregistered_0.10_advantage": (
                bootstrap["mean"] - random_baseline["mean"] >= 0.10
            ),
            "stable_atom_count": sum(atom["passes_0.10_advantage"] for atom in per_atom),
            "per_atom": per_atom,
        }
        print(
            f"{view_name}: init={initialization['mean']:.3f}, "
            f"bootstrap={bootstrap['mean']:.3f}, random={random_baseline['mean']:.3f}",
            flush=True,
        )

    report = {
        "analysis_status": "preregistered trajectory-bootstrap stability audit",
        "matching": "Hungarian mean absolute atom cosine",
        "repetitions": args.repetitions,
        "bootstrap_unit": "complete layer/head trajectory",
        "seed": args.seed,
        "views": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved stability report to {args.output}")


if __name__ == "__main__":
    main()
