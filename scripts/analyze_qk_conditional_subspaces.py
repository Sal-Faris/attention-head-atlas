"""Fit and validate unsupervised conditional QK subspaces on frozen splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from head_atlas.conditional_qk import (
    ConditionalQKSubspace,
    benjamini_hochberg,
    fit_conditional_qk_subspace,
    mapped_residual_projector,
    normalized_chordal_distance,
    orthogonal_projector,
)
from head_atlas.factor_io import load_factor_bundle
from head_atlas.qk_events import (
    DEFAULT_OFFSET_BINS,
    matched_source_events,
    qk_logits,
    relative_offset_statistics,
    residualize_by_offset,
)


@dataclass(frozen=True)
class Events:
    q_pre: np.ndarray
    k_positive_pre: np.ndarray
    k_negative_pre: np.ndarray
    q_rotated: np.ndarray
    k_positive_rotated: np.ndarray
    k_negative_rotated: np.ndarray
    destinations: np.ndarray
    positive_sources: np.ndarray
    negative_sources: np.ndarray
    bins: np.ndarray
    documents: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/qk_conditional_events_v1.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--family-audit",
        type=Path,
        default=Path("results/pythia-70m-deduped/subspace_family_audit.json"),
    )
    parser.add_argument("--model-revision", default="step143000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_conditional_subspaces_v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_conditional_subspaces_v1.png"),
    )
    return parser.parse_args()


def events_for_head(
    q_pre: np.ndarray,
    k_pre: np.ndarray,
    q_post: np.ndarray,
    k_post: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> Events:
    """Create one unsupervised matched-event table from a head's QK tensors."""

    residualized = residualize_by_offset(qk_logits(q_post, k_post), means, standard_deviations)
    payload: list[tuple[np.ndarray, ...]] = []
    for document, matrix in enumerate(residualized):
        selected = matched_source_events(matrix)
        if len(selected) == 0:
            continue
        destinations, positives, negatives, bins = selected.T
        payload.append(
            (
                q_pre[document, destinations],
                k_pre[document, positives],
                k_pre[document, negatives],
                q_post[document, destinations],
                k_post[document, positives],
                k_post[document, negatives],
                destinations,
                positives,
                negatives,
                bins,
                np.full(len(selected), document, dtype=np.int64),
            )
        )
    if not payload:
        raise RuntimeError("head has no eligible matched routing events")
    fields = tuple(np.concatenate([item[index] for item in payload], axis=0) for index in range(11))
    return Events(*fields)


def q_feature_margins(events: Events, basis: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact full and query-feature QK margins in actual rotary coordinates."""

    projector = orthogonal_projector(basis)
    query_feature = (events.q_pre - mean) @ projector
    # RoPE is position-dependent, but full and feature coordinates are already
    # aligned for these tensors only after rotating the projected component.
    # Reconstruct the position-wise rotation from q_pre -> q_rotated by using
    # the model's full rotary matrix supplied in the artifact's coordinates.
    # The explicit GPT-NeoX formula is used rather than an elementwise ratio.
    rotated_feature = rope(query_feature, events.destinations)
    key_difference = events.k_positive_rotated - events.k_negative_rotated
    full = np.sum(events.q_rotated * key_difference, axis=1) / np.sqrt(events.q_pre.shape[1])
    feature = np.sum(rotated_feature * key_difference, axis=1) / np.sqrt(events.q_pre.shape[1])
    return full, feature


def rope(
    values: np.ndarray,
    positions: np.ndarray,
    *,
    base: float = 10000.0,
    rotary_fraction: float = 0.25,
) -> np.ndarray:
    """Apply the full GPT-NeoX RoPE used by Pythia in row-vector coordinates."""

    width = values.shape[1]
    rotary_width = int(width * rotary_fraction)
    if rotary_width < 1 or rotary_width % 2:
        raise ValueError("rotary width must be a positive even number")
    frequencies = 1.0 / base ** (np.arange(0, rotary_width, 2) / rotary_width)
    angles = np.asarray(positions)[:, None] * frequencies
    cosine = np.concatenate((np.cos(angles), np.cos(angles)), axis=1)
    sine = np.concatenate((np.sin(angles), np.sin(angles)), axis=1)
    half = rotary_width // 2
    rotated_half = np.concatenate(
        (-values[:, half:rotary_width], values[:, :half]), axis=1
    )
    rotated = values[:, :rotary_width] * cosine + rotated_half * sine
    return np.concatenate((rotated, values[:, rotary_width:]), axis=1)


def score_basis(events: Events, basis: np.ndarray, mean: np.ndarray) -> float:
    full, feature = q_feature_margins(events, basis, mean)
    correlation = np.corrcoef(full, feature)[0, 1]
    return float(correlation**2) if np.isfinite(correlation) else -np.inf


def pca_basis(values: np.ndarray, rank: int) -> np.ndarray:
    centered = values - values.mean(axis=0)
    _, _, right_transpose = np.linalg.svd(centered, full_matrices=False)
    return right_transpose[:rank].T


def auc_positive_vs_negative(full_margin: np.ndarray) -> float:
    """AUC implied by one paired score difference per event."""

    ranks = rankdata(np.concatenate((full_margin, np.zeros_like(full_margin))))
    count = len(full_margin)
    return float((ranks[:count].sum() - count * (count + 1) / 2) / (count * count))


def distance_matrix(projectors: list[np.ndarray]) -> np.ndarray:
    count = len(projectors)
    result = np.zeros((count, count), dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            result[first, second] = result[second, first] = normalized_chordal_distance(
                projectors[first], projectors[second]
            )
    return result


def family_edges(path: Path) -> dict[str, np.ndarray]:
    source = json.loads(path.read_text(encoding="utf-8"))["views"]["QK"]["rank_results"]
    result = {}
    for rank in (4, 8, 16):
        for side in ("left", "right"):
            edges = source[str(rank)]["sides"][side]["recurrent_cross_layer_edges"]
            result[f"rank_{rank}_{side}"] = np.asarray(
                [[8 * item["first_layer"] + item["first_head"], 8 * item["second_layer"] + item["second_head"]] for item in edges],
                dtype=np.int64,
            )
    return result


def layer_matched_test(
    distances: np.ndarray,
    layers: np.ndarray,
    edges: np.ndarray,
    *,
    repetitions: int,
    rng: np.random.Generator,
    excluded: set[tuple[int, int]],
) -> dict[str, float]:
    observed = float(np.mean(distances[edges[:, 0], edges[:, 1]]))
    pools: dict[tuple[int, int], np.ndarray] = {}
    for first, second in edges:
        key = int(layers[first]), int(layers[second])
        if key not in pools:
            pools[key] = np.asarray(
                [
                    (left, right)
                    for left in np.flatnonzero(layers == key[0])
                    for right in np.flatnonzero(layers == key[1])
                    if (int(left), int(right)) not in excluded
                ],
                dtype=np.int64,
            )
        if len(pools[key]) == 0:
            raise RuntimeError("exclusion leaves no exact-layer-pair controls")
    null = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = []
        for first, second in edges:
            key = int(layers[first]), int(layers[second])
            sampled_edge = pools[key][rng.integers(len(pools[key]))]
            sampled.append(distances[int(sampled_edge[0]), int(sampled_edge[1])])
        null[repetition] = np.mean(sampled)
    return {
        "observed_mean_distance": observed,
        "null_mean": float(null.mean()),
        "ratio": float(observed / null.mean()),
        "lower_tail_p_value": float((1 + np.count_nonzero(null <= observed)) / (repetitions + 1)),
    }


def main() -> None:
    args = parse_args()
    if args.permutations < 1:
        raise ValueError("permutations must be positive")
    with np.load(args.input, allow_pickle=False) as artifact:
        data = {name: np.asarray(artifact[name]) for name in artifact.files if name != "metadata"}
        metadata = json.loads(str(artifact["metadata"]))
    probe_pre = data["discovery_query_pre_rope"][:, 0, 0]
    probe_post = data["discovery_query_post_rope"][:, 0, 0]
    probe_positions = np.tile(np.arange(probe_pre.shape[1]), probe_pre.shape[0])
    reconstructed_probe = rope(probe_pre.reshape(-1, 64), probe_positions).reshape(probe_pre.shape)
    rotary_reconstruction_error = float(np.max(np.abs(reconstructed_probe - probe_post)))
    if rotary_reconstruction_error > 2e-5:
        raise RuntimeError(
            f"analysis-side RoPE reconstruction failed: {rotary_reconstruction_error:.3e}"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == args.model_revision)
    operators, _ = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    layers = np.asarray([operator.layer for operator in operators], dtype=np.int64)
    edges_by_family = family_edges(args.family_audit)
    all_recurrent = {tuple(edge) for edges in edges_by_family.values() for edge in edges}
    choices = (1e-4, 1e-3, 1e-2, 1e-1)
    ranks = (1, 2, 4, 8, 16)
    fitted: dict[tuple[int, int], ConditionalQKSubspace] = {}
    summaries: dict[str, object] = {}
    projectors: dict[tuple[int, str], list[np.ndarray]] = {}
    for rank in ranks:
        projectors[(rank, "left")] = []
        projectors[(rank, "right")] = []

    for index, operator in enumerate(operators):
        layer, head = operator.layer, operator.head
        q_pre = data["discovery_query_pre_rope"][:, layer, head]
        k_pre = data["discovery_key_pre_rope"][:, layer, head]
        q_post = data["discovery_query_post_rope"][:, layer, head]
        k_post = data["discovery_key_post_rope"][:, layer, head]
        means, standard_deviations = relative_offset_statistics(qk_logits(q_post, k_post))
        discovery = events_for_head(q_pre, k_pre, q_post, k_post, means, standard_deviations)
        tuning = events_for_head(
            data["tuning_query_pre_rope"][:, layer, head], data["tuning_key_pre_rope"][:, layer, head],
            data["tuning_query_post_rope"][:, layer, head], data["tuning_key_post_rope"][:, layer, head], means, standard_deviations,
        )
        confirmation = events_for_head(
            data["confirmation_query_pre_rope"][:, layer, head], data["confirmation_key_pre_rope"][:, layer, head],
            data["confirmation_query_post_rope"][:, layer, head], data["confirmation_key_post_rope"][:, layer, head], means, standard_deviations,
        )
        if len(discovery.q_pre) < 1000:
            raise RuntimeError(f"L{layer}H{head} has fewer than 1,000 discovery events")
        by_rank = {}
        for rank in ranks:
            candidates = [fit_conditional_qk_subspace(discovery.q_pre, discovery.k_positive_pre, discovery.k_negative_pre, rank=rank, shrinkage=value) for value in choices]
            scores = [score_basis(tuning, candidate.query_basis, candidate.query_mean) for candidate in candidates]
            fit = candidates[int(np.argmax(scores))]
            fitted[index, rank] = fit
            full, feature = q_feature_margins(confirmation, fit.query_basis, fit.query_mean)
            pca = pca_basis(discovery.q_pre, rank)
            rng = np.random.default_rng(args.seed + 1000 * index + rank)
            haar, _ = np.linalg.qr(rng.normal(size=(64, rank)), mode="reduced")
            shuffled = fit_conditional_qk_subspace(discovery.q_pre, discovery.k_positive_pre[rng.permutation(len(discovery.q_pre))], discovery.k_negative_pre, rank=rank, shrinkage=fit.shrinkage)
            by_rank[str(rank)] = {
                "selected_shrinkage": fit.shrinkage,
                "discovery_events": len(discovery.q_pre), "tuning_events": len(tuning.q_pre), "confirmation_events": len(confirmation.q_pre),
                "conditional_query_r2": score_basis(confirmation, fit.query_basis, fit.query_mean),
                "pca_query_r2": score_basis(confirmation, pca, discovery.q_pre.mean(axis=0)),
                "haar_query_r2": score_basis(confirmation, haar, discovery.q_pre.mean(axis=0)),
                "shuffled_triplet_query_r2": score_basis(confirmation, shuffled.query_basis, shuffled.query_mean),
                "mean_full_margin": float(full.mean()), "mean_feature_margin": float(feature.mean()),
                "auc_positive_vs_negative": auc_positive_vs_negative(full),
                "singular_values": fit.singular_values[:rank].tolist(),
            }
            projectors[(rank, "left")].append(mapped_residual_projector(operator.left, fit.query_basis))
            projectors[(rank, "right")].append(mapped_residual_projector(operator.right, fit.key_basis))
        summaries[f"L{layer}H{head}"] = by_rank
        print(f"fit L{layer}H{head}", flush=True)

    family_results = {}
    for family, edges in edges_by_family.items():
        rank = int(family.split("_")[1])
        side = family.rsplit("_", 1)[1]
        distances = distance_matrix(projectors[(rank, side)])
        family_results[family] = {
            "including_recurrent_controls": layer_matched_test(distances, layers, edges, repetitions=args.permutations, rng=np.random.default_rng(args.seed + rank), excluded=set()),
            "excluding_all_26_recurrent_edges": layer_matched_test(distances, layers, edges, repetitions=args.permutations, rng=np.random.default_rng(args.seed + 100 + rank), excluded=all_recurrent),
        }
    for control_name in (
        "including_recurrent_controls",
        "excluding_all_26_recurrent_edges",
    ):
        names = list(family_results)
        adjusted = benjamini_hochberg(
            np.asarray(
                [
                    family_results[name][control_name]["lower_tail_p_value"]
                    for name in names
                ]
            )
        )
        for name, value in zip(names, adjusted, strict=True):
            family_results[name][control_name]["bh_adjusted_p_value"] = float(value)

    report = {"analysis_status": "conditional QK subspaces; confirmation is frozen", "metadata": metadata, "rotary_reconstruction_maximum_absolute_error": rotary_reconstruction_error, "ranks": ranks, "shrinkage_candidates": choices, "offset_bins": DEFAULT_OFFSET_BINS, "heads": summaries, "family_projector_tests": family_results, "permutations": args.permutations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    rank4 = [head["4"] for head in summaries.values()]
    axes[0].boxplot([[item[key] for item in rank4] for key in ("conditional_query_r2", "pca_query_r2", "haar_query_r2", "shuffled_triplet_query_r2")], tick_labels=["conditional", "PCA", "Haar", "shuffled"])
    axes[0].set_ylabel("confirmation query-margin $R^2$"); axes[0].set_title("Conditional QK feature prediction")
    names = list(family_results); ratios = [family_results[name]["excluding_all_26_recurrent_edges"]["ratio"] for name in names]
    axes[1].bar(np.arange(len(names)), ratios, color=["tab:blue" if value < 1 else "gray" for value in ratios]); axes[1].axhline(1, color="black", linewidth=1); axes[1].set_xticks(np.arange(len(names)), names, rotation=30, ha="right"); axes[1].set_ylabel("edge distance / exact-layer-pair null"); axes[1].set_title("Mapped conditional subspace recurrence")
    figure.savefig(args.figure, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
