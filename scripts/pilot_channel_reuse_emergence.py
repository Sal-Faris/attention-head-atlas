"""Training emergence of gauge-invariant writer-to-Q channel reuse.

All three rank-4 endpoints use identical balanced partner splits and are
aggregated equally over source layers.  Each null trajectory applies one
shared residual-coordinate permutation to every Q reader at a layer across
all checkpoints, preserving temporal Q geometry while breaking writer-Q
alignment.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.relational_invariants import orthonormal_span, permute_ambient_coordinates

CHECKPOINTS = (0, 64, 512, 1000, 4000, 16000, 64000, 143000)
RANK = 4


def balanced_splits(count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [(train, np.asarray([i for i in range(8) if i not in train])) for train in [np.sort(rng.choice(8, 4, replace=False)) for _ in range(count)]]


def load_spans(root: Path, checkpoint: int) -> tuple[dict, dict, dict]:
    directory = root / f"step{checkpoint}"
    ov, metadata = load_factor_bundle(directory / "ov_factors.npz")
    qk, _ = load_factor_bundle(directory / "qk_factors.npz")
    writers = {(x.layer, x.head): orthonormal_span(x.right.astype(np.float64)) for x in ov}
    readers = {(x.layer, x.head): orthonormal_span(x.left.astype(np.float64)) for x in qk}
    return writers, readers, metadata


def family_rank4(writer: np.ndarray, readers: dict, target_layer: int, heads: list[int], split: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    normalized, raw, overlaps = [], [], []
    for head in heads:
        cross = writer.T @ readers[target_layer, head]
        covariance = cross @ cross.T
        trace = float(np.trace(covariance))
        raw.append(covariance); normalized.append(covariance / max(trace, 1e-15))
        overlaps.append(trace / min(writer.shape[1], readers[target_layer, head].shape[1]))
    train, test = split
    _, vectors = np.linalg.eigh(np.mean([normalized[index] for index in train], axis=0))
    basis = vectors[:, ::-1][:, :RANK]
    equal = float(np.mean([np.trace(basis.T @ normalized[index] @ basis) for index in test]))
    _, weighted_vectors = np.linalg.eigh(np.sum([raw[index] for index in train], axis=0))
    weighted_basis = weighted_vectors[:, ::-1][:, :RANK]
    held_raw = np.sum([raw[index] for index in test], axis=0)
    weighted = float(np.trace(weighted_basis.T @ held_raw @ weighted_basis) / max(float(np.trace(held_raw)), 1e-15))
    return float(np.mean(overlaps)), equal, weighted


def trajectory(writers: dict, readers: dict, splits: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    layers = sorted({layer for layer, _ in writers})
    heads = sorted({head for _, head in writers})
    per_layer = {str(layer): [] for layer, _ in pairwise(layers)}
    for source_layer, target_layer in pairwise(layers):
        for split in splits:
            values = [family_rank4(writers[source_layer, head], readers, target_layer, heads, split) for head in heads]
            per_layer[str(source_layer)].append({"overlap": float(np.mean([x[0] for x in values])), "equal": float(np.mean([x[1] for x in values])), "weighted": float(np.mean([x[2] for x in values]))})
    return per_layer


def aggregate(per_layer: dict) -> dict[str, float]:
    return {key: float(np.mean([item[key] for values in per_layer.values() for item in values])) for key in ("overlap", "equal", "weighted")}


def shared_permute(readers_by_checkpoint: dict, rng: np.random.Generator) -> dict:
    first = next(iter(readers_by_checkpoint.values()))[1]
    ambient = next(iter(first.values())).shape[0]
    layers = sorted({layer for layer, _ in first})
    permutations = {layer: rng.permutation(ambient) for layer in layers}
    return {checkpoint: (value[0], {key: permute_ambient_coordinates(span, permutations[key[0]]) for key, span in value[1].items()}, value[2]) for checkpoint, value in readers_by_checkpoint.items()}


def empirical_p(nulls: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(nulls >= observed)) / (len(nulls) + 1))


def plot_report(report: dict, output: Path) -> None:
    """Plot the observed trajectory and its trajectory-null decomposition."""

    checkpoints = np.asarray(report["checkpoints"], dtype=float)
    plot_x = np.log10(checkpoints + 64.0)
    metrics = ("overlap", "equal", "weighted")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for metric in metrics:
        axes[0].plot(
            plot_x,
            [report["real"][str(int(checkpoint))]["aggregate"][metric] for checkpoint in checkpoints],
            marker="o",
            label=metric,
        )
    axes[0].axhline(64 / 512, color="#4C78A8", linestyle=":", alpha=0.7)
    axes[0].axhline(4 / 64, color="black", linestyle=":", alpha=0.55)
    axes[0].set(
        xlabel="training checkpoint",
        ylabel="rank-4 statistic",
        title="Learned writer→Q geometry",
    )
    axes[0].set_xticks(plot_x, [str(int(checkpoint)) for checkpoint in checkpoints], rotation=35)
    axes[0].legend()

    confirmatory = report["confirmatory"]["metrics"]
    positions = np.arange(len(metrics))
    width = 0.36
    axes[1].bar(
        positions - width / 2,
        [confirmatory[metric]["observed_final_minus_step0"] for metric in metrics],
        width,
        label="observed training change",
    )
    axes[1].bar(
        positions + width / 2,
        [confirmatory[metric]["null_mean"] for metric in metrics],
        width,
        label="trajectory-null change",
    )
    axes[1].set(
        xticks=positions,
        xticklabels=metrics,
        title="What training adds",
        ylabel="final minus initialization",
    )
    axes[1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/pythia-70m-deduped"))
    parser.add_argument("--null-repetitions", type=int, default=199)
    parser.add_argument("--split-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7717)
    parser.add_argument("--output", type=Path, default=Path("results/pythia-70m-deduped/channel_reuse_emergence_v1.json"))
    parser.add_argument("--figure", type=Path, default=Path("results/pythia-70m-deduped/channel_reuse_emergence_v1.png"))
    args = parser.parse_args()
    if args.null_repetitions < 1:
        raise ValueError("null repetitions must be positive")
    splits = balanced_splits(args.split_count, args.seed + 1)
    loaded = {checkpoint: load_spans(args.root, checkpoint) for checkpoint in CHECKPOINTS}
    real = {str(checkpoint): trajectory(value[0], value[1], splits) for checkpoint, value in loaded.items()}
    real_aggregate = {checkpoint: aggregate(real[str(checkpoint)]) for checkpoint in CHECKPOINTS}
    null_trajectories = {key: [] for key in ("overlap", "equal", "weighted")}
    null_layer_values = {str(checkpoint): [] for checkpoint in CHECKPOINTS}
    rng = np.random.default_rng(args.seed)
    for repetition in range(args.null_repetitions):
        permuted = shared_permute(loaded, rng)
        draw = {}
        for checkpoint in CHECKPOINTS:
            values = trajectory(permuted[checkpoint][0], permuted[checkpoint][1], splits)
            draw[str(checkpoint)] = values
            null_layer_values[str(checkpoint)].append(values)
        for metric, values in null_trajectories.items():
            values.append([aggregate(draw[str(checkpoint)])[metric] for checkpoint in CHECKPOINTS])
        if (repetition + 1) % 10 == 0:
            print(f"completed null trajectory {repetition + 1}/{args.null_repetitions}", flush=True)
    null_arrays = {metric: np.asarray(values) for metric, values in null_trajectories.items()}
    final_minus_step0 = {metric: real_aggregate[143000][metric] - real_aggregate[0][metric] for metric in null_trajectories}
    null_deltas = {metric: values[:, -1] - values[:, 0] for metric, values in null_arrays.items()}
    confirmatory = {metric: {"observed_final_minus_step0": final_minus_step0[metric], "null_mean": float(np.mean(null_deltas[metric])), "empirical_upper_tail_p_value": empirical_p(null_deltas[metric], final_minus_step0[metric])} for metric in null_trajectories}
    x = np.log10(np.asarray(CHECKPOINTS, dtype=float) + 64.0)
    slope = lambda y: float(np.dot(x - x.mean(), y - np.mean(y)) / np.dot(x - x.mean(), x - x.mean()))
    slope_records = {}
    for metric, values in null_arrays.items():
        observed_slope = slope(np.asarray([real_aggregate[c][metric] for c in CHECKPOINTS]))
        null_slopes = np.asarray([slope(row) for row in values])
        slope_records[metric] = {"observed": observed_slope, "null_mean": float(np.mean(null_slopes)), "empirical_upper_tail_p_value": empirical_p(null_slopes, observed_slope)}
    leaveout = {}
    for excluded in sorted(int(x) for x in real[str(CHECKPOINTS[0])]):
        observed_delta = {metric: float(aggregate({layer: real[str(143000)][layer] for layer in real[str(143000)] if int(layer) != excluded})[metric] - aggregate({layer: real[str(0)][layer] for layer in real[str(0)] if int(layer) != excluded})[metric]) for metric in null_trajectories}
        null_deltas_leave = {metric: [] for metric in null_trajectories}
        for draw_index in range(args.null_repetitions):
            final_draw = {layer: values for layer, values in null_layer_values[str(143000)][draw_index].items() if layer != str(excluded)}
            step0_draw = {layer: values for layer, values in null_layer_values[str(0)][draw_index].items() if layer != str(excluded)}
            final_stat = aggregate(final_draw); step0_stat = aggregate(step0_draw)
            for metric in null_trajectories:
                null_deltas_leave[metric].append(final_stat[metric] - step0_stat[metric])
        leaveout[str(excluded)] = {
            "observed_final_minus_step0": observed_delta,
            "empirical_p_values": {metric: empirical_p(np.asarray(null_deltas_leave[metric]), observed_delta[metric]) for metric in null_trajectories},
            "all_three_positive_required": all(value > 0 for value in observed_delta.values()),
        }
        leaveout[str(excluded)]["IUT_p_value"] = max(leaveout[str(excluded)]["empirical_p_values"].values())
    ratio = {metric: float(abs(real_aggregate[0][metric] - np.mean(values[:, 0])) / max(abs(real_aggregate[143000][metric] - np.mean(values[:, -1])), 1e-15)) for metric, values in null_arrays.items()}
    report = {"status": "rank-4 gauge-invariant writer-Q channel reuse emergence", "checkpoints": list(CHECKPOINTS), "rank": RANK, "split_count": args.split_count, "null_repetitions": args.null_repetitions, "null": "one shared residual-coordinate permutation per Q layer across every checkpoint and Q head", "aggregation": "equal source-layer weighting; per-layer/per-split values retained", "real": {str(x): {"aggregate": real_aggregate[x], "per_layer": real[str(x)]} for x in CHECKPOINTS}, "confirmatory": {"endpoint": "final minus step0 for overlap, equal-partner reuse, and overlap-weighted reuse", "all_three_positive_required": all(value > 0 for value in final_minus_step0.values()), "IUT_p_value": max(item["empirical_upper_tail_p_value"] for item in confirmatory.values()), "metrics": confirmatory}, "slope_secondary": {"metrics": slope_records, "IUT_p_value": max(item["empirical_upper_tail_p_value"] for item in slope_records.values()), "all_three_positive_required": all(item["observed"] > item["null_mean"] for item in slope_records.values())}, "null_step0_percentile": {metric: float(np.mean(values[:, 0] <= real_aggregate[0][metric])) for metric, values in null_arrays.items()}, "step0_excess_to_final_excess_ratio": ratio, "leave_one_source_layer_out": leaveout}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved result to {args.output}"); print(f"saved figure to {args.figure}")


if __name__ == "__main__":
    main()
