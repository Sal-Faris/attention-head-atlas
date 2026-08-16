"""Validate recurrent subspace-neighbor pairs against held-out head behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.activation import layer_pair_matched_edge_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        type=Path,
        default=Path("results/pythia-70m-deduped/subspace_family_audit.json"),
    )
    parser.add_argument(
        "--behavior",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/activation_validation_pilot.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/recurrent_pair_behavior.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/recurrent_pair_behavior.png"),
    )
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, float(values[index] * len(values) / rank))
        adjusted[index] = running
    return adjusted.tolist()


def edge_indices(
    records: list[dict[str, object]], index_by_location: dict[tuple[int, int], int]
) -> np.ndarray:
    edges = []
    for record in records:
        first = (int(record["first_layer"]), int(record["first_head"]))
        second = (int(record["second_layer"]), int(record["second_head"]))
        edges.append([index_by_location[first], index_by_location[second]])
    return np.asarray(edges, dtype=np.int64)


def plot_report(report: dict[str, object], output: Path) -> None:
    panels = (
        ("QK", "attention_centered", "Held-out attention patterns"),
        ("OV", "ov_response_centered", "Held-out OV responses"),
        ("OV", "head_result_centered", "Held-out composed head outputs"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, (view, target, title) in zip(axes, panels, strict=True):
        records = report["tests"][view][target]
        labels = [f"{record['side'][0].upper()} r{record['rank']}" for record in records]
        ratios = [record["observed_to_null_mean_ratio"] for record in records]
        colors = ["tab:blue" if record["fdr_q_value"] <= 0.05 else "0.65" for record in records]
        positions = np.arange(len(records))
        axis.bar(positions, ratios, color=colors)
        axis.axhline(1.0, color="black", linestyle=":", linewidth=1)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_ylabel("observed / layer-pair-matched distance")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Do recurrent subspace neighbors behave similarly on held-out documents?",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.permutations < 1:
        raise ValueError("permutations must be positive")
    family_report = json.loads(args.families.read_text(encoding="utf-8"))
    with np.load(args.behavior, allow_pickle=False) as bundle:
        layers = np.asarray(bundle["layers"], dtype=np.int64)
        heads = np.asarray(bundle["heads"], dtype=np.int64)
        targets = {
            "attention_centered": np.asarray(
                bundle["test_attention_centered_distances"], dtype=np.float64
            ),
            "ov_response_centered": np.asarray(
                bundle["test_ov_response_centered_distances"], dtype=np.float64
            ),
            "head_result_centered": np.asarray(
                bundle["test_head_result_centered_distances"], dtype=np.float64
            ),
        }
    index_by_location = {
        (int(layer), int(head)): index
        for index, (layer, head) in enumerate(zip(layers, heads, strict=True))
    }
    target_by_view = {
        "QK": ("attention_centered",),
        "OV": ("ov_response_centered", "head_result_centered"),
    }
    tests = {view: {target: [] for target in names} for view, names in target_by_view.items()}
    flat_records = []
    for view, target_names in target_by_view.items():
        for rank in family_report["ranks"]:
            rank_record = family_report["views"][view]["rank_results"][str(rank)]
            for side in ("left", "right"):
                edges = edge_indices(
                    rank_record["sides"][side]["recurrent_cross_layer_edges"],
                    index_by_location,
                )
                for target_index, target_name in enumerate(target_names):
                    result = layer_pair_matched_edge_test(
                        targets[target_name],
                        layers,
                        edges,
                        repetitions=args.permutations,
                        rng=np.random.default_rng(
                            args.seed
                            + (0 if view == "QK" else 10000)
                            + 100 * int(rank)
                            + (0 if side == "left" else 10)
                            + target_index
                        ),
                    )
                    record = {
                        "rank": int(rank),
                        "side": side,
                        **result,
                    }
                    tests[view][target_name].append(record)
                    flat_records.append(record)

    q_values = benjamini_hochberg(
        [float(record["lower_tail_p_value"]) for record in flat_records]
    )
    for record, q_value in zip(flat_records, q_values, strict=True):
        record["fdr_q_value"] = q_value

    report = {
        "analysis_status": "held-out validation of weight-selected recurrent edges",
        "edge_selection": "top recurrent cross-layer three-nearest-neighbor edges from checkpoint 0006",
        "behavior_split": "held-out test documents only",
        "null": "independent random head pairs with each edge's exact ordered layer pair",
        "multiple_testing": "Benjamini-Hochberg across all reported view/rank/side/target tests",
        "permutations": args.permutations,
        "seed": args.seed,
        "tests": tests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved recurrent-pair behavior audit to {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
