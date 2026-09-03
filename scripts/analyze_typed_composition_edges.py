"""Build a static atlas of OV-writer to Q/K/V-reader composition edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from head_atlas.factor_io import load_factor_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument("--late-checkpoints", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/typed_composition_edges_v1.json"),
    )
    return parser.parse_args()


def coupling_strength(writer: np.ndarray, reader: np.ndarray) -> float:
    """Return scale-free OV-writer to reader overlap in residual coordinates."""

    coupling = writer.T @ reader
    denominator = np.linalg.norm(writer) * np.linalg.norm(reader)
    return float(np.linalg.norm(coupling) / max(denominator, 1e-12))


def edge_tensor(ov_path: Path, qk_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ov, _ = load_factor_bundle(ov_path)
    qk, _ = load_factor_bundle(qk_path)
    if [(item.layer, item.head) for item in ov] != [(item.layer, item.head) for item in qk]:
        raise ValueError("OV and QK bundles do not share head locations")
    layers = np.asarray([item.layer for item in ov], dtype=np.int64)
    heads = np.asarray([item.head for item in ov], dtype=np.int64)
    layer_count = int(layers.max()) + 1
    head_count = int(heads.max()) + 1
    result = {
        "Q": np.full((layer_count, layer_count, head_count, head_count), np.nan),
        "K": np.full((layer_count, layer_count, head_count, head_count), np.nan),
        "V": np.full((layer_count, layer_count, head_count, head_count), np.nan),
    }
    location = {(item.layer, item.head): index for index, item in enumerate(ov)}
    for source_layer in range(layer_count):
        for target_layer in range(source_layer + 1, layer_count):
            for source_head in range(head_count):
                writer = ov[location[(source_layer, source_head)]].right
                for target_head in range(head_count):
                    qk_target = qk[location[(target_layer, target_head)]]
                    ov_target = ov[location[(target_layer, target_head)]]
                    result["Q"][source_layer, target_layer, source_head, target_head] = (
                        coupling_strength(writer, qk_target.left)
                    )
                    result["K"][source_layer, target_layer, source_head, target_head] = (
                        coupling_strength(writer, qk_target.right)
                    )
                    result["V"][source_layer, target_layer, source_head, target_head] = (
                        coupling_strength(writer, ov_target.left)
                    )
    return result, layers, heads


def temporal_identity_test(
    values: list[np.ndarray], *, resamples: int, rng: np.random.Generator
) -> dict[str, float]:
    """Does edge identity persist beyond each layer-pair's strength distribution?"""

    first, final = values[0], values[-1]
    layer_count, _, head_count, _ = first.shape
    correlations = []
    pairs = []
    for source_layer in range(layer_count):
        for target_layer in range(source_layer + 1, layer_count):
            early = first[source_layer, target_layer].ravel()
            late = final[source_layer, target_layer].ravel()
            correlations.append(float(spearmanr(early, late).statistic))
            pairs.append((early, late))
    observed = float(np.mean(correlations))
    null_values = []
    for _ in range(resamples):
        shuffled = []
        for early, late in pairs:
            matrix = early.reshape(head_count, head_count)
            source_order = rng.permutation(head_count)
            target_order = rng.permutation(head_count)
            shuffled.append(float(spearmanr(matrix[source_order][:, target_order].ravel(), late).statistic))
        null_values.append(float(np.mean(shuffled)))
    null = np.asarray(null_values)
    return {
        "mean_layer_pair_spearman": observed,
        "head_identity_shuffle_mean": float(np.mean(null)),
        "head_identity_shuffle_standard_deviation": float(np.std(null)),
        "upper_tail_p_value": float((1 + np.sum(null >= observed)) / (1 + len(null))),
    }


def final_top_edges(values: np.ndarray, count: int = 15) -> list[dict[str, float | int]]:
    records = []
    layer_count = values.shape[0]
    for source_layer in range(layer_count):
        for target_layer in range(source_layer + 1, layer_count):
            for source_head in range(values.shape[2]):
                for target_head in range(values.shape[3]):
                    records.append(
                        {
                            "source_layer": source_layer,
                            "source_head": source_head,
                            "target_layer": target_layer,
                            "target_head": target_head,
                            "normalized_overlap": float(
                                values[source_layer, target_layer, source_head, target_head]
                            ),
                        }
                    )
    return sorted(records, key=lambda item: -float(item["normalized_overlap"]))[:count]


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"][-args.late_checkpoints :]
    tensors: dict[str, list[np.ndarray]] = {kind: [] for kind in ("Q", "K", "V")}
    revisions = []
    for record in records:
        current, _, _ = edge_tensor(
            Path(record["factors"]["OV"]["path"]), Path(record["factors"]["QK"]["path"])
        )
        for kind, values in current.items():
            tensors[kind].append(values)
        revisions.append(record["revision"])
        print(f"loaded {record['revision']}", flush=True)
    report = {
        "status": "static scale-free OV-writer to Q/K/V-reader composition atlas",
        "revisions": revisions,
        "edge_definition": "||W_O(source) @ W_reader(target)||_F / (||W_O(source)||_F ||W_reader(target)||_F)",
        "temporal_null": "independent source- and target-head permutations within each ordered layer pair",
        "types": {
            kind: {
                "temporal_identity": temporal_identity_test(
                    values, resamples=args.resamples, rng=np.random.default_rng(args.seed + index)
                ),
                "final_top_edges": final_top_edges(values[-1]),
                "final_overlap_summary": {
                    "mean": float(np.nanmean(values[-1])),
                    "standard_deviation": float(np.nanstd(values[-1])),
                },
            }
            for index, (kind, values) in enumerate(tensors.items())
        },
        "seed": args.seed,
        "resamples": args.resamples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
