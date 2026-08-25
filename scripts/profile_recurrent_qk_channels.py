"""Attach held-out, token-level examples to recurrent joint QK channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head

from head_atlas.bilinear import fit_bilinear_margin_model, qk_margins
from head_atlas.qk_events import DEFAULT_OFFSET_BINS, qk_logits, relative_offset_statistics


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
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/recurrent_qk_channel_profiles_v1.json"),
    )
    return parser.parse_args()


def event_table(
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


def canonical_components(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired SVD components, with a deterministic simultaneous sign."""

    left, singular_values, right_transpose = np.linalg.svd(channel, full_matrices=False)
    for component in range(len(singular_values)):
        pivot = int(np.argmax(np.abs(left[:, component])))
        if left[pivot, component] < 0.0:
            left[:, component] *= -1.0
            right_transpose[component] *= -1.0
    return left, singular_values, right_transpose.T


def context(tokenizer: object, tokens: np.ndarray, position: int) -> dict[str, object]:
    start = max(0, position - 3)
    stop = min(len(tokens), position + 4)
    return {
        "token": tokenizer.decode([int(tokens[position])]),
        "context": tokenizer.decode(tokens[start:stop].tolist()),
        "position": int(position),
    }


def example_records(
    tokenizer: object,
    tokens: np.ndarray,
    events: object,
    contribution: np.ndarray,
    count: int,
) -> dict[str, list[dict[str, object]]]:
    def make(indices: np.ndarray) -> list[dict[str, object]]:
        records = []
        for index in indices:
            document = int(events.documents[index])
            row = tokens[document]
            records.append(
                {
                    "contribution": float(contribution[index]),
                    "document": document,
                    "offset_bin": int(events.bins[index]),
                    "destination": context(tokenizer, row, int(events.destinations[index])),
                    "positive_source": context(tokenizer, row, int(events.positive_sources[index])),
                    "neutral_source": context(tokenizer, row, int(events.negative_sources[index])),
                }
            )
        return records

    return {
        "positive": make(np.argsort(contribution)[-count:][::-1]),
        "negative": make(np.argsort(contribution)[:count]),
    }


def offset_enrichment(weighted: np.ndarray, counts: np.ndarray) -> list[float | None]:
    """Return magnitude enrichment, leaving absent offset bins unspecified."""

    total_weight = float(weighted.sum())
    total_count = int(counts.sum())
    if total_weight == 0.0 or total_count == 0:
        return [None] * len(counts)
    return [
        None if count == 0 else float(weight / total_weight / (count / total_count))
        for weight, count in zip(weighted, counts, strict=True)
    ]


def finite_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    correlation = float(np.corrcoef(first, second)[0, 1])
    return correlation if np.isfinite(correlation) else None


def token_match_profile(
    tokens: np.ndarray, events: object, magnitude: np.ndarray
) -> dict[str, float]:
    """Measure whether a component concentrates on literal token repetition."""

    destination_ids = tokens[events.documents, events.destinations]
    positive_ids = tokens[events.documents, events.positive_sources]
    top = magnitude >= np.quantile(magnitude, 0.9)
    baseline = float(np.mean(destination_ids == positive_ids))
    top_rate = float(np.mean(destination_ids[top] == positive_ids[top]))
    return {
        "all_events_destination_equals_positive_source": baseline,
        "top_decile_destination_equals_positive_source": top_rate,
        "top_decile_match_enrichment": float(top_rate / baseline) if baseline else 0.0,
    }


def profile_head(
    data: dict[str, np.ndarray],
    tuning: dict[str, object],
    tokenizer: object,
    layer: int,
    head: int,
    iterations: int,
    examples: int,
) -> dict[str, object]:
    discovery_q = data["discovery_query_post_rope"][:, layer, head]
    discovery_k = data["discovery_key_post_rope"][:, layer, head]
    mean, scale = relative_offset_statistics(qk_logits(discovery_q, discovery_k))
    discovery = event_table(data, "discovery", layer, head, mean, scale)
    confirmation = event_table(data, "confirmation", layer, head, mean, scale)
    ridge = float(tuning[f"L{layer}H{head}"]["4"]["selected_ridge"])
    model = fit_bilinear_margin_model(
        discovery.q_rotated,
        discovery.k_positive_rotated - discovery.k_negative_rotated,
        rank=4,
        ridge=ridge,
        iterations=iterations,
    )
    query_vectors, singular_values, key_vectors = canonical_components(model.left @ model.right.T)
    query_scores = confirmation.q_rotated @ query_vectors
    key_scores = (confirmation.k_positive_rotated - confirmation.k_negative_rotated) @ key_vectors
    contributions = query_scores * key_scores * singular_values / np.sqrt(query_scores.shape[1])
    dominant = np.argmax(np.abs(contributions), axis=1)
    target = qk_margins(
        confirmation.q_rotated,
        confirmation.k_positive_rotated - confirmation.k_negative_rotated,
    )
    overall_bins = np.bincount(confirmation.bins, minlength=len(DEFAULT_OFFSET_BINS))
    components = []
    for component in range(4):
        magnitude = np.abs(contributions[:, component])
        weighted_bins = np.bincount(
            confirmation.bins, weights=magnitude, minlength=len(DEFAULT_OFFSET_BINS)
        )
        components.append(
            {
                "component": component,
                "singular_value": float(singular_values[component]),
                "dominant_event_fraction": float(np.mean(dominant == component)),
                "contribution_target_correlation": finite_correlation(
                    contributions[:, component], target
                ),
                "offset_event_counts": overall_bins.tolist(),
                "magnitude_offset_enrichment": offset_enrichment(weighted_bins, overall_bins),
                "literal_token_match": token_match_profile(
                    data["confirmation_tokens"], confirmation, magnitude
                ),
                "examples": example_records(
                    tokenizer, data["confirmation_tokens"], confirmation, contributions[:, component], examples
                ),
            }
        )
    return {
        "ridge": ridge,
        "discovery_events": len(discovery.q_rotated),
        "confirmation_events": len(confirmation.q_rotated),
        "components": components,
    }


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.examples < 1:
        raise ValueError("iterations and examples must be positive")
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["heads"]
    families = json.loads(args.family_audit.read_text(encoding="utf-8"))["views"]["QK"]["rank_results"]["4"]["sides"]["right"]["recurrent_cross_layer_edges"]
    head_locations = {
        (record[f"{prefix}_layer"], record[f"{prefix}_head"])
        for record in families
        for prefix in ("first", "second")
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    snapshot = next(record["snapshot"] for record in manifest["records"] if record["revision"] == "step143000")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    profiles = {}
    for layer, head in sorted(head_locations):
        profiles[f"L{layer}H{head}"] = profile_head(
            data, tuning, tokenizer, layer, head, args.iterations, args.examples
        )
        print(f"profiled L{layer}H{head}", flush=True)
    result = {
        "status": "held-out component examples for rank-4 recurrent key-side QK channels",
        "selection": "heads incident to prior recurrent rank-4 key-side families",
        "offset_bins": DEFAULT_OFFSET_BINS,
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
