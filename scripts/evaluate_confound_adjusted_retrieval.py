"""Test functional retrieval after removing layer and spectral predictor directions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from head_atlas.distances import weighted_product_distances
from head_atlas.retrieval import evaluate_family_retrieval, permutation_retrieval_test
from head_atlas.structure import residualize_euclidean_distances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qk-input", type=Path, required=True)
    parser.add_argument("--ov-input", type=Path, required=True)
    parser.add_argument("--qk-statistics", type=Path, required=True)
    parser.add_argument("--ov-statistics", type=Path, required=True)
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
        kind = str(np.asarray(bundle["kinds"])[0])
    return distances, layers, heads, kind


def load_spectral_predictors(
    path: Path,
    layers: np.ndarray,
    heads: np.ndarray,
    expected_kind: str,
) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    records = {(int(row["layer"]), int(row["head"])): row for row in rows}
    if len(records) != len(rows):
        raise ValueError(f"duplicate head statistics in {path}")
    ordered = []
    for layer, head in zip(layers, heads, strict=True):
        row = records[(int(layer), int(head))]
        if row["kind"] != expected_kind:
            raise ValueError(f"unexpected operator kind in {path}: {row['kind']}")
        ordered.append([float(row["effective_rank"]), float(row["top_1_energy"])])
    return np.asarray(ordered, dtype=np.float64)


def benchmark_families(benchmark: dict[str, object]) -> dict[str, list[tuple[int, int]]]:
    return {
        record["family_id"]: [tuple(location) for location in record["primary_heads"]]
        for record in benchmark["families"]
        if record["use_for_primary_retrieval"]
        and record["inspection_status"] == "uninspected at benchmark freeze"
    }


def evaluate(
    distances: np.ndarray,
    layers: np.ndarray,
    heads: np.ndarray,
    families: dict[str, list[tuple[int, int]]],
    permutations: int,
    seed: int,
) -> dict[str, object]:
    return {
        "retrieval": evaluate_family_retrieval(distances, layers, heads, families),
        "global_permutation": permutation_retrieval_test(
            distances,
            layers,
            heads,
            families,
            permutations=permutations,
            seed=seed,
        ),
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
    qk_distances, layers, heads, qk_kind = load_distance_bundle(args.qk_input)
    ov_distances, ov_layers, ov_heads, ov_kind = load_distance_bundle(args.ov_input)
    if not np.array_equal(layers, ov_layers) or not np.array_equal(heads, ov_heads):
        raise ValueError("QK and OV bundles must contain heads in the same order")
    if qk_kind != "QK" or ov_kind != "OV":
        raise ValueError("expected one QK bundle and one OV bundle")

    qk_spectral = load_spectral_predictors(
        args.qk_statistics, layers, heads, expected_kind="QK"
    )
    ov_spectral = load_spectral_predictors(
        args.ov_statistics, layers, heads, expected_kind="OV"
    )
    layer_design = np.eye(int(np.max(layers)) + 1)[layers]
    joint_spectral = np.hstack([qk_spectral, ov_spectral])
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    families = benchmark_families(benchmark)

    raw_views = {
        "QK": (qk_distances, qk_spectral),
        "OV": (ov_distances, ov_spectral),
        "JOINT": (
            weighted_product_distances([qk_distances, ov_distances]),
            joint_spectral,
        ),
    }
    result_views = {}
    for view_name, (raw_distances, spectral_design) in raw_views.items():
        geometries = {
            "raw": raw_distances,
            "layer_residual": residualize_euclidean_distances(
                raw_distances, layer_design
            ),
            "spectral_residual": residualize_euclidean_distances(
                raw_distances, spectral_design
            ),
            "layer_and_spectral_residual": residualize_euclidean_distances(
                raw_distances, np.hstack([layer_design, spectral_design])
            ),
        }
        result_views[view_name] = {
            adjustment: evaluate(
                distances,
                layers,
                heads,
                families,
                args.permutations,
                args.seed,
            )
            for adjustment, distances in geometries.items()
        }

    result = {
        "benchmark_id": benchmark["benchmark_id"],
        "metric": "normalized_frobenius",
        "method": "feature-space linear residualization via centered Gram matrix",
        "permutations": args.permutations,
        "seed": args.seed,
        "views": result_views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved confound-adjusted retrieval audit to {args.output}")
    for view_name, adjustments in result_views.items():
        summaries = []
        for adjustment, record in adjustments.items():
            aggregate = record["retrieval"]["aggregate"]
            permutation = record["layer_stratified_permutation"]
            summaries.append(
                f"{adjustment}={aggregate['mean_average_precision']:.3f} "
                f"(p={permutation['upper_tail_p_value']:.4g})"
            )
        print(f"{view_name}: " + "; ".join(summaries))


if __name__ == "__main__":
    main()
