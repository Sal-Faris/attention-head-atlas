"""Evaluate spectrum and projector-subspace views against external families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_atlas.retrieval import evaluate_family_retrieval, permutation_retrieval_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def benchmark_families(benchmark: dict[str, object]) -> dict[str, list[tuple[int, int]]]:
    return {
        record["family_id"]: [tuple(location) for location in record["primary_heads"]]
        for record in benchmark["families"]
        if record["use_for_primary_retrieval"]
        and record["inspection_status"] == "uninspected at benchmark freeze"
    }


def normalized_vector_distances(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms == 0):
        raise ValueError("cannot normalize zero feature vectors")
    normalized = features / norms[:, None]
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distances = np.sqrt(np.maximum(2.0 - 2.0 * similarities, 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def load_views(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as bundle:
        layers = np.asarray(bundle["layers"], dtype=np.int64)
        heads = np.asarray(bundle["heads"], dtype=np.int64)
        kind = str(np.asarray(bundle["kinds"])[0])
        ranks = np.asarray(bundle["ranks"], dtype=np.int64)
        views = {
            "singular_spectrum": normalized_vector_distances(bundle["singular_values"])
        }
        for rank in ranks:
            for side in ("left", "right", "joint"):
                key = f"{side}_rank_{rank}"
                views[key] = np.asarray(bundle[key], dtype=np.float64)
    return views, layers, heads, kind


def evaluate_view(
    distances: np.ndarray,
    layers: np.ndarray,
    heads: np.ndarray,
    families: dict[str, list[tuple[int, int]]],
    permutations: int,
    seed: int,
) -> dict[str, object]:
    return {
        "retrieval": evaluate_family_retrieval(distances, layers, heads, families),
        "layer_stratified_permutation": permutation_retrieval_test(
            distances,
            layers,
            heads,
            families,
            permutations=permutations,
            seed=seed,
            stratify_by_layer=True,
        ),
    }


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    families = benchmark_families(benchmark)
    qk_views, qk_layers, qk_heads, qk_kind = load_views(args.qk_input)
    ov_views, ov_layers, ov_heads, ov_kind = load_views(args.ov_input)
    if qk_kind != "QK" or ov_kind != "OV":
        raise ValueError("expected one QK bundle and one OV bundle")
    if not np.array_equal(qk_layers, ov_layers) or not np.array_equal(qk_heads, ov_heads):
        raise ValueError("QK and OV bundles must contain heads in the same order")

    result_views = {}
    for kind, views in (("QK", qk_views), ("OV", ov_views)):
        result_views[kind] = {}
        for view_name, distances in views.items():
            evaluation = evaluate_view(
                distances,
                qk_layers,
                qk_heads,
                families,
                args.permutations,
                args.seed,
            )
            result_views[kind][view_name] = evaluation
            aggregate = evaluation["retrieval"]["aggregate"]
            permutation = evaluation["layer_stratified_permutation"]
            print(
                f"{kind} {view_name}: mAP={aggregate['mean_average_precision']:.3f}, "
                f"layer-p={permutation['upper_tail_p_value']:.4g}"
            )

    result = {
        "benchmark_id": benchmark["benchmark_id"],
        "analysis_status": "exploratory metric and rank comparison",
        "labels_used_to_compare_metrics": True,
        "permutations": args.permutations,
        "seed": args.seed,
        "views": result_views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved subspace retrieval audit to {args.output}")


if __name__ == "__main__":
    main()
