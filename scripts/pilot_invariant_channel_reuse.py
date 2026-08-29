"""Pilot gauge-invariant reuse of residual-stream writer/reader channels.

This experiment intentionally discards arbitrary coordinates inside each head.
It compares only residual-stream subspaces: OV writer spans, and Q/K/V reader
spans.  For each fixed component, alternating partner heads are used to learn a
shared channel subspace and the held-out partners test whether that channel is
reused.  The primary null applies one shared residual-coordinate permutation
to every partner in a layer, preserving partner spectra and all pairwise
subspace geometry while breaking alignment to the fixed component.
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.relational_invariants import (
    orthonormal_span,
    permute_ambient_coordinates,
)

RANKS = (1, 2, 4, 8, 16)
TYPES = ("Q", "K", "V")
DIRECTIONS = ("fan_out", "fan_in")
N_SPLITS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ov", type=Path,
        default=Path("artifacts/pythia-70m-deduped/step143000/ov_factors.npz"),
    )
    parser.add_argument(
        "--qk", type=Path,
        default=Path("artifacts/pythia-70m-deduped/step143000/qk_factors.npz"),
    )
    parser.add_argument("--null-repetitions", type=int, default=49)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/pythia-70m-deduped/invariant_channel_reuse_v1.json"),
    )
    parser.add_argument(
        "--figure", type=Path,
        default=Path("results/pythia-70m-deduped/invariant_channel_reuse_v1.png"),
    )
    return parser.parse_args()


def component_spans(ov_path: Path, qk_path: Path) -> tuple[dict, dict, dict]:
    """Load factors and return residual-coordinate orthonormal spans."""

    ov, metadata = load_factor_bundle(ov_path)
    qk, _ = load_factor_bundle(qk_path)
    ov_map = {(operator.layer, operator.head): operator for operator in ov}
    qk_map = {(operator.layer, operator.head): operator for operator in qk}
    writers = {
        key: orthonormal_span(operator.right.astype(np.float64))
        for key, operator in ov_map.items()
    }
    readers = {
        "Q": {key: orthonormal_span(operator.left.astype(np.float64)) for key, operator in qk_map.items()},
        "K": {key: orthonormal_span(operator.right.astype(np.float64)) for key, operator in qk_map.items()},
        "V": {key: orthonormal_span(operator.left.astype(np.float64)) for key, operator in ov_map.items()},
    }
    return writers, readers, metadata


def balanced_split_schedule(partner_count: int = 8, count: int = N_SPLITS, seed: int = 2718) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a deterministic, balanced schedule shared by real and null data."""

    if partner_count < 4 or partner_count % 2:
        raise ValueError("partner_count must be an even number at least four")
    rng = np.random.default_rng(seed)
    half = partner_count // 2
    schedules = []
    for _ in range(count):
        training = np.sort(rng.choice(partner_count, size=half, replace=False))
        held_out = np.asarray([index for index in range(partner_count) if index not in training])
        schedules.append((training, held_out))
    return schedules


def _measure_one(
    fixed: np.ndarray,
    partners: tuple[np.ndarray, ...],
    split_schedule: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    if len(partners) < 4 or len(partners) % 2:
        raise ValueError("this pilot expects an even number of at least four partner heads")
    if any(
        len(training) + len(held_out) != len(partners)
        or set(training).intersection(held_out)
        or set(training).union(held_out) != set(range(len(partners)))
        for training, held_out in split_schedule
    ):
        raise ValueError("each split must partition all partner heads")
    normalized = []
    raw = []
    overlaps = []
    for partner in partners:
        cross = fixed.T @ partner
        covariance = cross @ cross.T
        trace = float(np.trace(covariance))
        if trace <= 1e-15:
            raise ValueError("a partner has numerically zero overlap with the fixed subspace")
        raw.append(covariance)
        normalized.append(covariance / trace)
        overlaps.append(trace / min(fixed.shape[1], partner.shape[1]))
    normalized_captures = []
    weighted_captures = []
    for training_indices, held_out_indices in split_schedule:
        _, vectors = np.linalg.eigh(np.mean([normalized[i] for i in training_indices], axis=0))
        basis_order = vectors[:, ::-1]
        _, weighted_vectors = np.linalg.eigh(np.sum([raw[i] for i in training_indices], axis=0))
        weighted_order = weighted_vectors[:, ::-1]
        normalized_captures.append([
            float(np.mean([np.trace(basis_order[:, :rank].T @ normalized[i] @ basis_order[:, :rank]) for i in held_out_indices]))
            for rank in RANKS
        ])
        held_out_raw = np.sum([raw[i] for i in held_out_indices], axis=0)
        total_trace = float(np.trace(held_out_raw))
        weighted_captures.append([
            float(np.trace(weighted_order[:, :rank].T @ held_out_raw @ weighted_order[:, :rank]) / total_trace)
            for rank in RANKS
        ])
    return {
        "split_captures": normalized_captures,
        "weighted_split_captures": weighted_captures,
        "mean_pair_overlap": float(np.mean(overlaps)),
    }


def measure_population(writers: dict, readers: dict, split_schedule: list[tuple[np.ndarray, np.ndarray]], *, primary_only: bool = False) -> dict:
    """Measure every adjacent-layer family, retaining reciprocal splits."""

    layers = sorted({layer for layer, _ in writers})
    heads = sorted({head for _, head in writers})
    result = {direction: {reader_type: [] for reader_type in TYPES} for direction in DIRECTIONS}
    for source_layer, target_layer in pairwise(layers):
        for reader_type in TYPES:
            if primary_only and reader_type != "Q":
                continue
            for source_head in heads:
                partners = tuple(readers[reader_type][target_layer, head] for head in heads)
                result["fan_out"][reader_type].append({
                    "label": f"L{source_layer}H{source_head}->L{target_layer}{reader_type}",
                    "layer": source_layer,
                    "score": _measure_one(writers[source_layer, source_head], partners, split_schedule),
                })
            if primary_only:
                continue
            for target_head in heads:
                partners = tuple(writers[source_layer, head] for head in heads)
                result["fan_in"][reader_type].append({
                    "label": f"L{source_layer}->L{target_layer}H{target_head}{reader_type}",
                    "layer": target_layer,
                    "score": _measure_one(readers[reader_type][target_layer, target_head], partners, split_schedule),
                })
    # Source-mismatch control: compare each source writer with a same-sized
    # Q-reader family from a different target layer.  This asks whether the
    # apparent reuse is specific to the intended target layer, rather than a
    # generic Q-reader bus shared by every layer.  It is descriptive because
    # layer identity itself may be functionally meaningful.
    target_layers = layers[1:]
    for source_index, source_layer in enumerate(layers[:-1]):
        actual_target = layers[source_index + 1]
        alternatives = [layer for layer in target_layers if layer != actual_target]
        if not alternatives:
            continue
        mismatch_target = min(alternatives, key=lambda layer: (abs(layer - actual_target), layer))
        for source_head in heads:
            partners = tuple(readers["Q"][mismatch_target, head] for head in heads)
            record_index = next(
                index for index, record in enumerate(result["fan_out"]["Q"])
                if record["layer"] == source_layer and record["label"].startswith(f"L{source_layer}H{source_head}->")
            )
            result["fan_out"]["Q"][record_index]["mismatch_score"] = _measure_one(
                writers[source_layer, source_head], partners, split_schedule
            )
            result["fan_out"]["Q"][record_index]["mismatch_target_layer"] = mismatch_target
    return result


def jointly_permute_readers(readers: dict, rng: np.random.Generator) -> dict:
    """Use one permutation per layer, shared across Q/K/V and all heads."""

    layers = {layer for values in readers.values() for layer, _ in values}
    first_reader = next(iter(readers.values()))
    first_key = next(iter(first_reader))
    ambient = first_reader[first_key].shape[0]
    permutations = {layer: rng.permutation(ambient) for layer in layers}
    return {
        reader_type: {
            key: permute_ambient_coordinates(value, permutations[key[0]])
            for key, value in values.items()
        }
        for reader_type, values in readers.items()
    }


def jointly_permute_writers(writers: dict, rng: np.random.Generator) -> dict:
    """Use one permutation per source layer for all source writers."""

    layers = {layer for layer, _ in writers}
    ambient = next(iter(writers.values())).shape[0]
    permutations = {layer: rng.permutation(ambient) for layer in layers}
    return {
        key: permute_ambient_coordinates(value, permutations[key[0]])
        for key, value in writers.items()
    }


def _arrays(population: dict, direction: str, reader_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    records = population[direction][reader_type]
    captures = np.asarray([record["score"]["split_captures"] for record in records])
    weighted = np.asarray([record["score"]["weighted_split_captures"] for record in records])
    overlaps = np.asarray([record["score"]["mean_pair_overlap"] for record in records])
    layers = [int(record["layer"]) for record in records]
    return captures, weighted, overlaps, layers


def _empirical_p(null_values: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1))


def _primary_statistics(capture: np.ndarray, weighted: np.ndarray, overlap: np.ndarray) -> dict[str, float]:
    """Population statistic at one rank, averaged over families and splits."""

    return {
        "mean_pair_overlap": float(np.mean(overlap)),
        "equal_partner_capture": float(np.mean(capture)),
        "overlap_weighted_capture": float(np.mean(weighted)),
    }


def summarize(real: dict, nulls: list[dict]) -> dict:
    report = {}
    for direction in DIRECTIONS:
        report[direction] = {}
        for reader_type in TYPES:
            real_capture, real_weighted, real_overlap, layers = _arrays(real, direction, reader_type)
            has_null = all(item[direction].get(reader_type) for item in nulls)
            if not has_null:
                ranks = {}
                for index, rank in enumerate(RANKS):
                    ranks[str(rank)] = {
                        "real_mean_pair_overlap": float(np.mean(real_overlap)),
                        "real_equal_partner_crossfit_capture": float(np.mean(real_capture[:, :, index])),
                        "real_overlap_weighted_crossfit_capture": float(np.mean(real_weighted[:, :, index])),
                        "descriptive_only": True,
                    }
                report[direction][reader_type] = {"family_count": len(real_capture), "ranks": ranks, "rank4_by_layer": {}, "descriptive_only": True}
                continue
            null_capture = np.stack([_arrays(item, direction, reader_type)[0] for item in nulls])
            null_weighted = np.stack([_arrays(item, direction, reader_type)[1] for item in nulls])
            null_overlap = np.stack([_arrays(item, direction, reader_type)[2] for item in nulls])
            ranks = {}
            for index, rank in enumerate(RANKS):
                observed_split = np.mean(real_capture[:, :, index], axis=0)
                null_split = np.mean(null_capture[:, :, :, index], axis=1)
                observed_weighted_split = np.mean(real_weighted[:, :, index], axis=0)
                null_weighted_split = np.mean(null_weighted[:, :, :, index], axis=1)
                observed = float(np.mean(observed_split))
                null_population = np.mean(null_split, axis=1)
                weighted_observed = float(np.mean(observed_weighted_split))
                weighted_population = np.mean(null_weighted_split, axis=1)
                observed_overlap = float(np.mean(real_overlap))
                overlap_population = np.mean(null_overlap, axis=1)
                ranks[str(rank)] = {
                    "real_mean_pair_overlap": observed_overlap,
                    "null_mean_pair_overlap": float(np.mean(overlap_population)),
                    "overlap_advantage": observed_overlap - float(np.mean(overlap_population)),
                    "overlap_empirical_upper_tail_p_value": _empirical_p(overlap_population, observed_overlap),
                    "real_equal_partner_crossfit_capture": observed,
                    "null_equal_partner_crossfit_capture_mean": float(np.mean(null_population)),
                    "equal_partner_capture_advantage": observed - float(np.mean(null_population)),
                    "equal_partner_empirical_upper_tail_p_value": _empirical_p(null_population, observed),
                    "real_overlap_weighted_crossfit_capture": weighted_observed,
                    "null_overlap_weighted_crossfit_capture_mean": float(np.mean(weighted_population)),
                    "weighted_capture_advantage": weighted_observed - float(np.mean(weighted_population)),
                    "weighted_empirical_upper_tail_p_value": _empirical_p(weighted_population, weighted_observed),
                    "repeated_split_p_values": [
                        _empirical_p(null_split[:, split], observed_split[split])
                        for split in range(null_split.shape[1])
                    ],
                }
            layer_summary = {}
            for layer in sorted(set(layers)):
                indices = np.asarray([i for i, value in enumerate(layers) if value == layer])
                observed_split = np.mean(real_capture[indices, :, RANKS.index(4)], axis=0)
                null_split = np.mean(null_capture[:, indices, :, RANKS.index(4)], axis=1)
                observed = float(np.mean(observed_split))
                null_population = np.mean(null_split, axis=1)
                layer_summary[str(layer)] = {
                    "real_rank4_capture": observed,
                    "null_rank4_capture_mean": float(np.mean(null_population)),
                    "rank4_advantage": observed - float(np.mean(null_population)),
                    "rank4_empirical_upper_tail_p_value": _empirical_p(null_population, observed),
                    "repeated_split_p_values": [
                        _empirical_p(null_split[:, split], observed_split[split])
                        for split in range(null_split.shape[1])
                    ],
                }
            report[direction][reader_type] = {
                "family_count": len(real_capture),
                "ranks": ranks,
                "rank4_by_layer": layer_summary,
            }
    # The source-mismatch control is only defined for the primary Q fan-out
    # endpoint.  It uses the same writer and split schedule, replacing the
    # intended next-layer Q family with a nearest different target layer.
    real_records = real["fan_out"]["Q"]
    mismatch_records = [record for record in real_records if "mismatch_score" in record]
    if mismatch_records:
        real_mismatch = np.asarray([record["mismatch_score"]["split_captures"] for record in mismatch_records])
        real_mismatch_weighted = np.asarray([record["mismatch_score"]["weighted_split_captures"] for record in mismatch_records])
        real_mismatch_overlap = np.asarray([record["mismatch_score"]["mean_pair_overlap"] for record in mismatch_records])
        null_mismatch = np.stack([
            np.asarray([record["mismatch_score"]["split_captures"] for record in item["fan_out"]["Q"] if "mismatch_score" in record])
            for item in nulls
        ])
        null_mismatch_weighted = np.stack([
            np.asarray([record["mismatch_score"]["weighted_split_captures"] for record in item["fan_out"]["Q"] if "mismatch_score" in record])
            for item in nulls
        ])
        null_mismatch_overlap = np.stack([
            np.asarray([record["mismatch_score"]["mean_pair_overlap"] for record in item["fan_out"]["Q"] if "mismatch_score" in record])
            for item in nulls
        ])
        rank_index = RANKS.index(4)
        actual_capture, actual_weighted, actual_overlap, _ = _arrays(real, "fan_out", "Q")
        null_capture = np.stack([_arrays(item, "fan_out", "Q")[0] for item in nulls])
        null_weighted = np.stack([_arrays(item, "fan_out", "Q")[1] for item in nulls])
        null_overlap = np.stack([_arrays(item, "fan_out", "Q")[2] for item in nulls])
        actual = _primary_statistics(actual_capture[:, :, rank_index], actual_weighted[:, :, rank_index], actual_overlap)
        mismatch = _primary_statistics(real_mismatch[:, :, rank_index], real_mismatch_weighted[:, :, rank_index], real_mismatch_overlap)
        null_actual = [
            _primary_statistics(item_capture[:, :, rank_index], item_weighted[:, :, rank_index], item_overlap)
            for item_capture, item_weighted, item_overlap in zip(null_capture, null_weighted, null_overlap, strict=True)
        ]
        null_mismatch_stats = [
            _primary_statistics(item_capture[:, :, rank_index], item_weighted[:, :, rank_index], item_overlap)
            for item_capture, item_weighted, item_overlap in zip(null_mismatch, null_mismatch_weighted, null_mismatch_overlap, strict=True)
        ]
        report["fan_out"]["Q"]["source_mismatch_control"] = {
            "description": "same source writers, but Q-reader partners from the nearest different target layer",
            "actual_target_layer_statistics": actual,
            "mismatched_target_layer_statistics": mismatch,
            "actual_minus_mismatch": {key: actual[key] - mismatch[key] for key in actual},
            "empirical_p_values_for_actual_minus_mismatch": {
                key: _empirical_p(
                    np.asarray([a[key] - m[key] for a, m in zip(null_actual, null_mismatch_stats, strict=True)]),
                    actual[key] - mismatch[key],
                )
                for key in actual
            },
        }
        primary = report["fan_out"]["Q"]["ranks"]["4"]
        primary["confirmatory_endpoint"] = True
        primary["intersection_union_p_value"] = max(
            primary["overlap_empirical_upper_tail_p_value"],
            primary["equal_partner_empirical_upper_tail_p_value"],
            primary["weighted_empirical_upper_tail_p_value"],
        )
        primary["all_three_positive_excess"] = all(
            primary[key] > 0 for key in ("overlap_advantage", "equal_partner_capture_advantage", "weighted_capture_advantage")
        )
        # Leave-one-source-layer-out summaries for the same preregistered endpoint.
        leave_out = {}
        layers = sorted(set(_arrays(real, "fan_out", "Q")[3]))
        for layer in layers:
            keep = np.asarray([value != layer for value in _arrays(real, "fan_out", "Q")[3]])
            obs = _primary_statistics(actual_capture[keep, :, rank_index], actual_weighted[keep, :, rank_index], actual_overlap[keep])
            null_stats = []
            for item in nulls:
                cap, wei, ov, item_layers = _arrays(item, "fan_out", "Q")
                item_keep = np.asarray([value != layer for value in item_layers])
                null_stats.append(_primary_statistics(cap[item_keep, :, rank_index], wei[item_keep, :, rank_index], ov[item_keep]))
            leave_out[str(layer)] = {
                "observed": obs,
                "null_means": {key: float(np.mean([item[key] for item in null_stats])) for key in obs},
                "empirical_p_values": {key: _empirical_p(np.asarray([item[key] for item in null_stats]), obs[key]) for key in obs},
            }
        primary["leave_one_source_layer_out"] = leave_out
    return report


def plot_report(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"Q": "#4C78A8", "K": "#F58518", "V": "#54A24B"}
    for column, direction in enumerate(DIRECTIONS):
        advantage_plotted = False
        for reader_type in TYPES:
            records = report[direction][reader_type]["ranks"]
            real = [records[str(rank)]["real_equal_partner_crossfit_capture"] for rank in RANKS]
            axes[0, column].plot(RANKS, real, marker="o", color=colors[reader_type], label=reader_type)
            if not report[direction][reader_type].get("descriptive_only", False):
                null = [records[str(rank)]["null_equal_partner_crossfit_capture_mean"] for rank in RANKS]
                axes[0, column].plot(RANKS, null, linestyle="--", color=colors[reader_type], alpha=0.7)
                axes[1, column].plot(
                    RANKS,
                    [records[str(rank)]["equal_partner_capture_advantage"] for rank in RANKS],
                    marker="o", color=colors[reader_type], label=reader_type,
                )
                advantage_plotted = True
        axes[0, column].set_title("Fan-out" if direction == "fan_out" else "Fan-in")
        axes[0, column].set_ylabel("Held-out subspace capture")
        axes[0, column].legend(title="reader", loc="best")
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].set_xlabel("rank")
        axes[1, column].set_ylabel("real minus permutation null")
        if not advantage_plotted:
            axes[1, column].text(
                0.5, 0.5, "descriptive only\n(no confirmatory null)",
                ha="center", va="center", transform=axes[1, column].transAxes,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.null_repetitions < 1:
        raise ValueError("null repetitions must be positive")
    rng = np.random.default_rng(args.seed)
    writers, readers, metadata = component_spans(args.ov, args.qk)
    head_count = len({head for _, head in writers})
    split_schedule = balanced_split_schedule(partner_count=head_count)
    real = measure_population(writers, readers, split_schedule)
    nulls = []
    for repetition in range(args.null_repetitions):
        # Fan-out breaks fixed-writer/partner alignment using shared layer axes.
        permuted_readers = jointly_permute_readers(readers, rng)
        nulls.append(measure_population(writers, permuted_readers, split_schedule, primary_only=True))
        print(f"completed null {repetition + 1}/{args.null_repetitions}", flush=True)
    report = {
        "status": "gauge-invariant typed writer-to-reader channel reuse pilot",
        "model": metadata.get("model"),
        "revision": metadata.get("revision"),
        "scope": "all adjacent-layer OV-writer to Q/K/V-reader edges",
        "discovery": "weights only; no prompts, activations, labels, or semantic operator types",
        "representation": {
            "writer": "orthonormal span of columns of OV.right (= W_O.T)",
            "Q_reader": "orthonormal span of columns of QK.left (= W_Q)",
            "K_reader": "orthonormal span of columns of QK.right (= W_K)",
            "V_reader": "orthonormal span of columns of OV.left (= W_V)",
            "gauge_invariance": "invariant to invertible changes of internal head coordinates",
        },
        "crossfit": "learn the mean normalized overlap covariance on each balanced training half and test the held-out half across 16 deterministic splits",
        "balanced_split_schedule": {"partner_count": head_count, "split_count": N_SPLITS, "seed": 2718, "same_for_real_and_all_nulls": True},
        "ranks": list(RANKS),
        "primary_descriptive_rank": 4,
        "null": "for the confirmatory endpoint, one shared random permutation of ambient residual coordinates per Q-reader partner layer, shared across all partner heads; null evaluation is restricted to fan_out/Q plus its source-mismatch control for economy",
        "null_preserves": "each partner spectrum, each individual subspace dimension, and all pairwise subspace geometry within a jointly permuted layer",
        "null_repetitions": args.null_repetitions,
        "seed": args.seed,
        "p_values": "empirical permutation p-values only; no Gaussian tail approximation",
        "confirmatory_endpoint": "fan_out/Q/rank4; intersection-union p is the maximum of the one-sided empirical p-values for total overlap, equal-partner normalized reuse, and overlap-weighted reuse; all other endpoints are descriptive",
        "results": summarize(real, nulls),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report["results"], args.figure)
    print(f"saved result to {args.output}")
    print(f"saved figure to {args.figure}")


if __name__ == "__main__":
    main()
