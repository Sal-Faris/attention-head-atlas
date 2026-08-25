"""Test recurrence of RoPE-aware rank-2/4 bilinear QK channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head, rope

from head_atlas.bilinear import fit_bilinear_margin_model
from head_atlas.factor_io import load_factor_bundle
from head_atlas.qk_events import qk_logits, relative_offset_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/pythia-70m-deduped/qk_bilinear_margin_confirmation_v1.npz"))
    parser.add_argument("--tuning", type=Path, default=Path("results/pythia-70m-deduped/qk_bilinear_margin_tuning_v1.json"))
    parser.add_argument("--family-audit", type=Path, default=Path("results/pythia-70m-deduped/subspace_family_audit.json"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pythia-70m-deduped-pilot.json"))
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--output", type=Path, default=Path("results/pythia-70m-deduped/qk_bilinear_channel_recurrence_v1.json"))
    return parser.parse_args()


def rotations(length: int = 64) -> np.ndarray:
    identity = np.eye(64)
    return np.stack([rope(identity, np.full(64, position)) for position in range(length)])


def normalized_distances(matrices: list[np.ndarray]) -> np.ndarray:
    vectors = np.stack([matrix.reshape(-1) for matrix in matrices])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))


def exact_layer_test(
    distances: np.ndarray,
    layers: np.ndarray,
    edges: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    excluded: set[tuple[int, int]],
) -> dict[str, float]:
    observed = float(np.mean(distances[edges[:, 0], edges[:, 1]]))
    null = []
    for _ in range(repetitions):
        values = []
        for first, second in edges:
            candidates = np.asarray(
                [
                    (source, target)
                    for source in np.flatnonzero(layers == layers[first])
                    for target in np.flatnonzero(layers == layers[second])
                    if (int(source), int(target)) not in excluded
                ]
            )
            source, target = candidates[rng.integers(len(candidates))]
            values.append(distances[source, target])
        null.append(np.mean(values))
    null_array = np.asarray(null)
    return {"observed": observed, "null_mean": float(null_array.mean()), "ratio": float(observed / null_array.mean()), "lower_tail_p": float((1 + np.count_nonzero(null_array <= observed)) / (repetitions + 1))}


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["heads"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == "step143000")
    operators, _ = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    rotation = rotations()
    # The previous cross-checkpoint family audit contains rank-4 side families.
    # Restrict this confirmatory comparison to the common rank rather than
    # pretending that its rank-4 families provide a rank-2 reference set.
    kernels = {4: []}
    for index, operator in enumerate(operators):
        layer, head = operator.layer, operator.head
        qpre, kpre = data["discovery_query_pre_rope"][:, layer, head], data["discovery_key_pre_rope"][:, layer, head]
        qpost, kpost = data["discovery_query_post_rope"][:, layer, head], data["discovery_key_post_rope"][:, layer, head]
        mean, scale = relative_offset_statistics(qk_logits(qpost, kpost))
        event = events_for_head(qpre, kpre, qpost, kpost, mean, scale)
        difference = event.k_positive_rotated - event.k_negative_rotated
        for rank, rank_kernels in kernels.items():
            ridge = float(tuning[f"L{layer}H{head}"][str(rank)]["selected_ridge"])
            model = fit_bilinear_margin_model(event.q_rotated, difference, rank=rank, ridge=ridge, iterations=args.iterations)
            channel = model.left @ model.right.T
            effective = np.mean([rotation[i] @ channel @ rotation[j].T for i, j in zip(event.destinations, event.positive_sources, strict=True)], axis=0)
            rank_kernels.append(operator.left @ effective @ operator.right.T)
        print(f"mapped L{layer}H{head}", flush=True)
    source = json.loads(args.family_audit.read_text(encoding="utf-8"))["views"]["QK"]["rank_results"]
    layers = np.asarray([operator.layer for operator in operators])
    rank_four = source["4"]["sides"]
    all_family_edges = {
        (8 * record["first_layer"] + record["first_head"], 8 * record["second_layer"] + record["second_head"])
        for side in ("left", "right")
        for record in rank_four[side]["recurrent_cross_layer_edges"]
    }
    report = {"n_excluded_family_edges": len(all_family_edges)}
    for rank, matrices in kernels.items():
        distances = normalized_distances(matrices)
        for side in ("left", "right"):
            records = source[str(rank)]["sides"][side]["recurrent_cross_layer_edges"]
            edges = np.asarray([[8 * record["first_layer"] + record["first_head"], 8 * record["second_layer"] + record["second_head"]] for record in records])
            seed = rank + (0 if side == "left" else 100)
            report[f"rank_{rank}_{side}"] = {
                "all_exact_layer_pairs": exact_layer_test(
                    distances,
                    layers,
                    edges,
                    args.permutations,
                    np.random.default_rng(seed),
                    excluded=set(),
                ),
                "excluding_all_known_family_edges": exact_layer_test(
                    distances,
                    layers,
                    edges,
                    args.permutations,
                    np.random.default_rng(seed + 1000),
                    excluded=all_family_edges,
                ),
                "n_recurrent_edges": len(edges),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
