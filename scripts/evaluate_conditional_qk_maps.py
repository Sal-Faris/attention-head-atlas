"""Evaluate discovery-fitted class-conditional QK maps on held-out margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head
from test_qk_channel_input_classes import assign, event_residuals, fit_input_classes, head_locations

from head_atlas.bilinear import bilinear_scores, fit_bilinear_margin_model, qk_margins, r_squared
from head_atlas.qk_events import qk_logits, relative_offset_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/qk_bilinear_margin_confirmation_v1.npz"),
    )
    parser.add_argument(
        "--tuning",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_bilinear_margin_tuning_v1.json"),
    )
    parser.add_argument(
        "--family-audit",
        type=Path,
        default=Path("results/pythia-70m-deduped/subspace_family_audit.json"),
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--minimum-events", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/conditional_qk_map_evaluation_v1.json"),
    )
    return parser.parse_args()


def events(
    data: dict[str, np.ndarray], split: str, layer: int, head: int, mean: np.ndarray, scale: np.ndarray
):
    return events_for_head(
        data[f"{split}_query_pre_rope"][:, layer, head],
        data[f"{split}_key_pre_rope"][:, layer, head],
        data[f"{split}_query_post_rope"][:, layer, head],
        data[f"{split}_key_post_rope"][:, layer, head],
        mean,
        scale,
    )


def closest_tuned_rank(rank: int) -> str:
    return str(min((1, 2, 4, 8, 16), key=lambda candidate: abs(candidate - rank)))


def group_ids(query_classes: np.ndarray, key_classes: np.ndarray, key_count: int) -> np.ndarray:
    return query_classes * key_count + key_classes


def predict_by_group(
    queries: np.ndarray,
    key_differences: np.ndarray,
    groups: np.ndarray,
    models: dict[int, object],
    fallback: object,
) -> np.ndarray:
    prediction = bilinear_scores(queries, key_differences, fallback)
    for group, model in models.items():
        indices = np.flatnonzero(groups == group)
        if len(indices) == 1:
            prediction[indices] = np.sum(
                (queries[indices] @ model.left) * (key_differences[indices] @ model.right), axis=1
            ) / np.sqrt(queries.shape[1])
        elif len(indices) > 1:
            prediction[indices] = bilinear_scores(queries[indices], key_differences[indices], model)
    return prediction


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.minimum_events < 2:
        raise ValueError("iterations must be positive and minimum-events must be at least two")
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["heads"]
    reports = {}
    for layer, head in head_locations(args.family_audit):
        location = f"L{layer}H{head}"
        discovery_q = data["discovery_query_post_rope"][:, layer, head]
        discovery_k = data["discovery_key_post_rope"][:, layer, head]
        mean, scale = relative_offset_statistics(qk_logits(discovery_q, discovery_k))
        discovery = events(data, "discovery", layer, head, mean, scale)
        confirmation = events(data, "confirmation", layer, head, mean, scale)
        residual = data["discovery_normalized_residual"][:, layer]
        confirmation_residual = data["confirmation_normalized_residual"][:, layer]
        query_classes = fit_input_classes(
            event_residuals(residual, discovery, source=False), 4, 32, args.seed + layer * 10 + head
        )
        key_classes = fit_input_classes(
            event_residuals(residual, discovery, source=True), 4, 32, args.seed + 1000 + layer * 10 + head
        )
        discovery_groups = group_ids(
            assign(query_classes, event_residuals(residual, discovery, source=False)),
            assign(key_classes, event_residuals(residual, discovery, source=True)),
            key_classes.selected_components,
        )
        confirmation_groups = group_ids(
            assign(query_classes, event_residuals(confirmation_residual, confirmation, source=False)),
            assign(key_classes, event_residuals(confirmation_residual, confirmation, source=True)),
            key_classes.selected_components,
        )
        qd = discovery.q_rotated
        dd = discovery.k_positive_rotated - discovery.k_negative_rotated
        qc = confirmation.q_rotated
        dc = confirmation.k_positive_rotated - confirmation.k_negative_rotated
        fallback = fit_bilinear_margin_model(
            qd,
            dd,
            rank=1,
            ridge=float(tuning[location]["1"]["selected_ridge"]),
            iterations=args.iterations,
        )
        group_models = {
            int(group): fit_bilinear_margin_model(
                qd[indices],
                dd[indices],
                rank=1,
                ridge=float(tuning[location]["1"]["selected_ridge"]),
                iterations=args.iterations,
            )
            for group in np.unique(discovery_groups)
            if len(indices := np.flatnonzero(discovery_groups == group)) >= args.minimum_events
        }
        rank_budget = 1 + len(group_models)
        global_budget = fit_bilinear_margin_model(
            qd,
            dd,
            rank=rank_budget,
            ridge=float(tuning[location][closest_tuned_rank(rank_budget)]["selected_ridge"]),
            iterations=args.iterations,
        )
        global_rank_four = fit_bilinear_margin_model(
            qd,
            dd,
            rank=4,
            ridge=float(tuning[location]["4"]["selected_ridge"]),
            iterations=args.iterations,
        )
        target = qk_margins(qc, dc)
        conditional = predict_by_group(qc, dc, confirmation_groups, group_models, fallback)
        reports[location] = {
            "query_classes": query_classes.selected_components,
            "key_classes": key_classes.selected_components,
            "conditional_rank_one_groups": len(group_models),
            "total_rank_budget": rank_budget,
            "confirmation_r2": {
                "conditional_maps": r_squared(conditional, target),
                "global_equal_budget": r_squared(bilinear_scores(qc, dc, global_budget), target),
                "global_rank_four": r_squared(bilinear_scores(qc, dc, global_rank_four), target),
            },
            "confirmation_group_event_counts": {
                str(group): int(np.count_nonzero(confirmation_groups == group))
                for group in np.unique(confirmation_groups)
            },
        }
        print(f"evaluated {location}", flush=True)
    conditional_scores = [item["confirmation_r2"]["conditional_maps"] for item in reports.values()]
    equal_budget_scores = [item["confirmation_r2"]["global_equal_budget"] for item in reports.values()]
    rank_four_scores = [item["confirmation_r2"]["global_rank_four"] for item in reports.values()]
    result = {
        "status": "held-out class-conditional QK map evaluation",
        "conditional_model": "rank-one map per populated discovery query-class/key-class pair plus a rank-one fallback",
        "equal_budget_control": "one global low-rank map with identical number of rank-one factors",
        "heads": reports,
        "summary": {
            "mean_conditional_r2": float(np.mean(conditional_scores)),
            "mean_global_equal_budget_r2": float(np.mean(equal_budget_scores)),
            "mean_global_rank_four_r2": float(np.mean(rank_four_scores)),
            "conditional_minus_equal_budget_mean": float(
                np.mean(np.asarray(conditional_scores) - np.asarray(equal_budget_scores))
            ),
            "heads_conditional_beats_equal_budget": int(
                np.count_nonzero(np.asarray(conditional_scores) > np.asarray(equal_budget_scores))
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
