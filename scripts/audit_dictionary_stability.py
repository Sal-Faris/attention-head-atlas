"""Measure motif-dictionary stability across initializations and bootstraps."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import DictionaryLearning

from head_atlas.distances import weighted_product_distances
from head_atlas.embedding import classical_mds
from head_atlas.motifs import matched_atom_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--dictionary-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_distances(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return (
            np.asarray(bundle["distances"], dtype=np.float64),
            np.asarray(bundle["layers"], dtype=np.int64),
            np.asarray(bundle["heads"], dtype=np.int64),
        )


def fit_dictionary(
    coordinates: np.ndarray,
    components: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    initial_dictionary = rng.standard_normal((components, centered.shape[1]))
    initial_dictionary /= np.linalg.norm(initial_dictionary, axis=1, keepdims=True)
    initial_codes = 0.1 * rng.standard_normal((len(centered), components))
    model = DictionaryLearning(
        n_components=components,
        alpha=alpha,
        max_iter=1000,
        fit_algorithm="cd",
        code_init=initial_codes,
        dict_init=initial_dictionary,
        random_state=seed,
    ).fit(centered)
    return model.components_


def pairwise_similarities(dictionaries: list[np.ndarray]) -> list[float]:
    return [
        matched_atom_similarity(first, second)
        for first, second in itertools.combinations(dictionaries, 2)
    ]


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values)
    return {
        "comparison_count": len(values),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    if args.repetitions < 2:
        raise ValueError("repetitions must be at least two")
    if not 0.0 < args.sample_fraction < 1.0:
        raise ValueError("sample fraction must lie strictly between zero and one")
    qk_distances, layers, heads = load_distances(args.qk_input)
    ov_distances, ov_layers, ov_heads = load_distances(args.ov_input)
    if not np.array_equal(layers, ov_layers) or not np.array_equal(heads, ov_heads):
        raise ValueError("QK and OV bundles must contain heads in the same order")
    distance_views = {
        "QK": qk_distances,
        "OV": ov_distances,
        "JOINT": weighted_product_distances([qk_distances, ov_distances]),
    }
    rng = np.random.default_rng(args.seed)
    results = {}
    for view_name, distances in distance_views.items():
        coordinates = np.asarray(classical_mds(distances)["coordinates"])
        results[view_name] = {}
        for components in sorted(set(args.components)):
            full_dictionaries = [
                fit_dictionary(
                    coordinates,
                    components,
                    args.dictionary_alpha,
                    args.seed + repetition,
                )
                for repetition in range(args.repetitions)
            ]
            bootstrap_dictionaries = []
            sample_size = round(args.sample_fraction * len(coordinates))
            for repetition in range(args.repetitions):
                indices = rng.choice(len(coordinates), size=sample_size, replace=True)
                bootstrap_dictionaries.append(
                    fit_dictionary(
                        coordinates[indices],
                        components,
                        args.dictionary_alpha,
                        args.seed + repetition,
                    )
                )
            random_similarities = []
            for _ in range(20):
                first = rng.standard_normal((components, coordinates.shape[1]))
                second = rng.standard_normal((components, coordinates.shape[1]))
                random_similarities.append(matched_atom_similarity(first, second))
            results[view_name][str(components)] = {
                "initialization_stability": summarize(
                    pairwise_similarities(full_dictionaries)
                ),
                "bootstrap_stability": summarize(
                    pairwise_similarities(bootstrap_dictionaries)
                ),
                "random_dictionary_baseline": summarize(random_similarities),
            }
            record = results[view_name][str(components)]
            print(
                f"{view_name} k={components}: init={record['initialization_stability']['mean']:.3f}, "
                f"bootstrap={record['bootstrap_stability']['mean']:.3f}, "
                f"random={record['random_dictionary_baseline']['mean']:.3f}",
                flush=True,
            )
    result = {
        "analysis_status": "exploratory dictionary-stability audit",
        "matching": "Hungarian mean absolute atom cosine",
        "repetitions": args.repetitions,
        "sample_fraction": args.sample_fraction,
        "dictionary_alpha": args.dictionary_alpha,
        "seed": args.seed,
        "views": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved dictionary stability audit to {args.output}")


if __name__ == "__main__":
    main()
