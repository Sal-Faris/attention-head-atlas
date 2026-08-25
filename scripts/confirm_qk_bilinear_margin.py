"""Confirm frozen joint QK bilinear compressors on a new document split."""

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
        "--input", type=Path,
        default=Path("artifacts/pythia-70m-deduped/qk_bilinear_margin_confirmation_v1.npz"),
    )
    parser.add_argument(
        "--tuning", type=Path,
        default=Path("results/pythia-70m-deduped/qk_bilinear_margin_tuning_v1.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/pythia-70m-deduped/qk_bilinear_margin_confirmation_v1.json"),
    )
    return parser.parse_args()


def events(data: dict[str, np.ndarray], split: str, layer: int, head: int, mean: np.ndarray, scale: np.ndarray):
    return events_for_head(
        data[f"{split}_query_pre_rope"][:, layer, head], data[f"{split}_key_pre_rope"][:, layer, head],
        data[f"{split}_query_post_rope"][:, layer, head], data[f"{split}_key_post_rope"][:, layer, head],
        mean, scale,
    )


def bootstrap_difference(prediction: np.ndarray, baseline: np.ndarray, target: np.ndarray, documents: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    unique = np.unique(documents)
    differences = []
    for _ in range(1000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(documents == document) for document in sampled])
        differences.append(r_squared(prediction[indices], target[indices]) - r_squared(baseline[indices], target[indices]))
    return {"mean_difference": float(r_squared(prediction, target) - r_squared(baseline, target)), "lower_95": float(np.quantile(differences, .025)), "upper_95": float(np.quantile(differences, .975))}


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["heads"]
    ranks = (1, 2, 4, 8, 16)
    report = {}
    for layer in range(6):
        for head in range(8):
            discovery_q = data["discovery_query_post_rope"][:, layer, head]
            discovery_k = data["discovery_key_post_rope"][:, layer, head]
            mean, scale = relative_offset_statistics(qk_logits(discovery_q, discovery_k))
            discovery = events(data, "discovery", layer, head, mean, scale)
            confirmation = events(data, "confirmation", layer, head, mean, scale)
            qd, dd = discovery.q_rotated, discovery.k_positive_rotated - discovery.k_negative_rotated
            qc, dc = confirmation.q_rotated, confirmation.k_positive_rotated - confirmation.k_negative_rotated
            target = qk_margins(qc, dc)
            by_rank = {}
            for rank in ranks:
                ridge = float(tuning[f"L{layer}H{head}"][str(rank)]["selected_ridge"])
                model = fit_bilinear_margin_model(qd, dd, rank=rank, ridge=ridge, iterations=args.iterations, seed=args.seed)
                prediction = bilinear_scores(qc, dc, model)
                query_pca = projected_identity_scores(qc, dc, pca_basis(qd, rank))
                key_pca = projected_identity_scores(qc, dc, pca_basis(dd, rank))
                rng = np.random.default_rng(args.seed + layer * 100 + head * 10 + rank)
                haar, _ = np.linalg.qr(rng.normal(size=(64, rank)), mode="reduced")
                haar_prediction = projected_identity_scores(qc, dc, haar)
                shuffled = fit_bilinear_margin_model(qd, dd[rng.permutation(len(dd))], rank=rank, ridge=ridge, iterations=args.iterations, seed=args.seed)
                shuffled_prediction = bilinear_scores(qc, dc, shuffled)
                baseline = query_pca if r_squared(query_pca, target) >= r_squared(key_pca, target) else key_pca
                by_rank[str(rank)] = {"ridge": ridge, "bilinear_r2": r_squared(prediction, target), "query_pca_r2": r_squared(query_pca, target), "key_pca_r2": r_squared(key_pca, target), "haar_r2": r_squared(haar_prediction, target), "shuffled_r2": r_squared(shuffled_prediction, target), "vs_best_pca_bootstrap": bootstrap_difference(prediction, baseline, target, confirmation.documents, rng)}
            report[f"L{layer}H{head}"] = by_rank
            print(f"confirmed L{layer}H{head}", flush=True)
    result = {"status": "fresh confirmation", "ranks": ranks, "iterations": args.iterations, "heads": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
