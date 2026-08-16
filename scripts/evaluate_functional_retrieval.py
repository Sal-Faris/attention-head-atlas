"""Evaluate frozen operator distances against published GPT-2 head families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_atlas.distances import weighted_product_distances
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


def load_distance_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as bundle:
        distances = np.asarray(bundle["distances"], dtype=np.float64)
        layers = np.asarray(bundle["layers"], dtype=np.int64)
        heads = np.asarray(bundle["heads"], dtype=np.int64)
        kinds = np.asarray(bundle["kinds"])
        metric = str(bundle["metric"].item())
    if np.unique(kinds).size != 1:
        raise ValueError(f"operator kinds in {path} must be uniform")
    if metric != "normalized_frobenius":
        raise ValueError(f"unsupported metric in {path}: {metric}")
    return distances, layers, heads, str(kinds[0])


def benchmark_families(
    benchmark: dict[str, object],
    include_inspected: bool,
) -> dict[str, list[tuple[int, int]]]:
    families = {}
    for record in benchmark["families"]:
        if not record["use_for_primary_retrieval"]:
            continue
        if not include_inspected and record["inspection_status"] != "uninspected at benchmark freeze":
            continue
        locations = [tuple(location) for location in record["primary_heads"]]
        if len(locations) >= 2:
            families[record["family_id"]] = locations
    return families


def evaluate_view(
    distances: np.ndarray,
    layers: np.ndarray,
    heads: np.ndarray,
    all_families: dict[str, list[tuple[int, int]]],
    uninspected_families: dict[str, list[tuple[int, int]]],
    permutations: int,
    seed: int,
) -> dict[str, object]:
    leave_one_family_out = {}
    for excluded_family in uninspected_families:
        retained = {
            family_id: locations
            for family_id, locations in uninspected_families.items()
            if family_id != excluded_family
        }
        leave_one_family_out[excluded_family] = {
            "retrieval": evaluate_family_retrieval(distances, layers, heads, retained),
            "label_permutation": permutation_retrieval_test(
                distances,
                layers,
                heads,
                retained,
                permutations=permutations,
                seed=seed,
            ),
            "layer_stratified_permutation": permutation_retrieval_test(
                distances,
                layers,
                heads,
                retained,
                permutations=permutations,
                seed=seed,
                stratify_by_layer=True,
            ),
        }

    return {
        "all_primary_families_descriptive": evaluate_family_retrieval(
            distances, layers, heads, all_families
        ),
        "uninspected_families": evaluate_family_retrieval(
            distances, layers, heads, uninspected_families
        ),
        "uninspected_label_permutation": permutation_retrieval_test(
            distances,
            layers,
            heads,
            uninspected_families,
            permutations=permutations,
            seed=seed,
        ),
        "uninspected_layer_stratified_permutation": permutation_retrieval_test(
            distances,
            layers,
            heads,
            uninspected_families,
            permutations=permutations,
            seed=seed,
            stratify_by_layer=True,
        ),
        "leave_one_family_out": leave_one_family_out,
    }


def main() -> None:
    args = parse_args()
    if args.permutations < 1:
        raise ValueError("permutations must be positive")

    qk_distances, qk_layers, qk_heads, qk_kind = load_distance_bundle(args.qk_input)
    ov_distances, ov_layers, ov_heads, ov_kind = load_distance_bundle(args.ov_input)
    if not np.array_equal(qk_layers, ov_layers) or not np.array_equal(qk_heads, ov_heads):
        raise ValueError("QK and OV bundles must contain heads in the same order")
    if qk_kind != "QK" or ov_kind != "OV":
        raise ValueError("expected one QK bundle and one OV bundle")

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    all_families = benchmark_families(benchmark, include_inspected=True)
    uninspected_families = benchmark_families(benchmark, include_inspected=False)
    joint_distances = weighted_product_distances([qk_distances, ov_distances])

    result = {
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_source": str(args.benchmark),
        "metric": "normalized_frobenius",
        "joint_metric": "equal-weight-euclidean-product-of-qk-and-ov",
        "permutations": args.permutations,
        "seed": args.seed,
        "views": {
            "QK": evaluate_view(
                qk_distances,
                qk_layers,
                qk_heads,
                all_families,
                uninspected_families,
                args.permutations,
                args.seed,
            ),
            "OV": evaluate_view(
                ov_distances,
                ov_layers,
                ov_heads,
                all_families,
                uninspected_families,
                args.permutations,
                args.seed,
            ),
            "JOINT": evaluate_view(
                joint_distances,
                qk_layers,
                qk_heads,
                all_families,
                uninspected_families,
                args.permutations,
                args.seed,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved functional retrieval audit to {args.output}")
    for view_name, view_result in result["views"].items():
        aggregate = view_result["uninspected_families"]["aggregate"]
        permutation = view_result["uninspected_label_permutation"]
        stratified = view_result["uninspected_layer_stratified_permutation"]
        print(
            f"{view_name}: family-balanced mAP={aggregate['mean_average_precision']:.4f}, "
            f"global p={permutation['upper_tail_p_value']:.4g}, "
            f"layer-stratified p={stratified['upper_tail_p_value']:.4g}"
        )


if __name__ == "__main__":
    main()
