"""Tune joint low-rank QK margin compressors without using confirmation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head, pca_basis

from head_atlas.bilinear import (
    bilinear_scores,
    fit_bilinear_margin_model,
    projected_identity_scores,
    qk_margins,
    r_squared,
)
from head_atlas.qk_events import qk_logits, relative_offset_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/qk_conditional_events_v1.npz"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_bilinear_margin_tuning_v1.json"),
    )
    return parser.parse_args()


def event_split(data: dict[str, np.ndarray], split: str, layer: int, head: int, means: np.ndarray, scales: np.ndarray):
    return events_for_head(
        data[f"{split}_query_pre_rope"][:, layer, head],
        data[f"{split}_key_pre_rope"][:, layer, head],
        data[f"{split}_query_post_rope"][:, layer, head],
        data[f"{split}_key_post_rope"][:, layer, head],
        means,
        scales,
    )


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    with np.load(args.input, allow_pickle=False) as artifact:
        data = {name: np.asarray(artifact[name]) for name in artifact.files if name != "metadata"}
    ranks = (1, 2, 4, 8, 16)
    ridges = (1e-5, 1e-4, 1e-3, 1e-2)
    records = {}
    for layer in range(6):
        for head in range(8):
            discovery_q = data["discovery_query_post_rope"][:, layer, head]
            discovery_k = data["discovery_key_post_rope"][:, layer, head]
            means, scales = relative_offset_statistics(qk_logits(discovery_q, discovery_k))
            discovery = event_split(data, "discovery", layer, head, means, scales)
            tuning = event_split(data, "tuning", layer, head, means, scales)
            q_discovery = discovery.q_rotated
            d_discovery = discovery.k_positive_rotated - discovery.k_negative_rotated
            q_tuning = tuning.q_rotated
            d_tuning = tuning.k_positive_rotated - tuning.k_negative_rotated
            target_tuning = qk_margins(q_tuning, d_tuning)
            by_rank = {}
            for rank in ranks:
                candidates = [
                    fit_bilinear_margin_model(
                        q_discovery,
                        d_discovery,
                        rank=rank,
                        ridge=ridge,
                        iterations=args.iterations,
                        seed=args.seed,
                    )
                    for ridge in ridges
                ]
                tuning_scores = [
                    r_squared(bilinear_scores(q_tuning, d_tuning, model), target_tuning)
                    for model in candidates
                ]
                best = int(np.argmax(tuning_scores))
                pca = pca_basis(q_discovery, rank)
                rng = np.random.default_rng(args.seed + 100 * layer + 10 * head + rank)
                haar, _ = np.linalg.qr(rng.normal(size=(64, rank)), mode="reduced")
                by_rank[str(rank)] = {
                    "selected_ridge": ridges[best],
                    "bilinear_tuning_r2": tuning_scores[best],
                    "query_pca_tuning_r2": r_squared(projected_identity_scores(q_tuning, d_tuning, pca), target_tuning),
                    "haar_tuning_r2": r_squared(projected_identity_scores(q_tuning, d_tuning, haar), target_tuning),
                    "discovery_events": len(q_discovery),
                    "tuning_events": len(q_tuning),
                }
            records[f"L{layer}H{head}"] = by_rank
            print(f"tuned L{layer}H{head}", flush=True)
    report = {"status": "discovery/tuning only; no new confirmation data used", "ranks": ranks, "ridges": ridges, "iterations": args.iterations, "heads": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
