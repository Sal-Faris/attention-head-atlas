"""Test whether recurrent QK key sides transfer after an orthogonal gauge alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head, rope
from test_qk_channel_transfer import raw_qk_affine_factors

from head_atlas.bilinear import qk_margins, r_squared
from head_atlas.qk_events import qk_logits, relative_offset_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/qk_bilinear_margin_confirmation_v1.npz"),
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
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--resamples", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_key_side_transfer_v1.json"),
    )
    return parser.parse_args()


def head_events(
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


def calibrated_prediction(discovery_score: np.ndarray, discovery_target: np.ndarray, confirmation_score: np.ndarray) -> np.ndarray:
    design = np.column_stack((discovery_score, np.ones(len(discovery_score))))
    slope, intercept = np.linalg.lstsq(design, discovery_target, rcond=None)[0]
    return slope * confirmation_score + intercept


def rotated_keys(residual: np.ndarray, factor: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    documents, sequence, _ = residual.shape
    positions = np.tile(np.arange(sequence), documents)
    values = residual.reshape(-1, residual.shape[-1]) @ factor[0] + factor[1]
    return rope(values, positions).reshape(documents, sequence, -1)


def scores(events: object, donor_keys: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    difference = donor_keys[events.documents, events.positive_sources] - donor_keys[
        events.documents, events.negative_sources
    ]
    return np.sum(events.q_rotated * (difference @ alignment), axis=1) / np.sqrt(
        events.q_rotated.shape[1]
    )


def alignment(donor_keys: np.ndarray, target_keys: np.ndarray, rank: int) -> np.ndarray:
    cross_covariance = donor_keys.reshape(-1, donor_keys.shape[-1]).T @ target_keys.reshape(
        -1, target_keys.shape[-1]
    )
    left, _, right_transpose = np.linalg.svd(cross_covariance, full_matrices=False)
    return left[:, :rank] @ right_transpose[:rank]


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    family = json.loads(args.family_audit.read_text(encoding="utf-8"))["views"]["QK"][
        "rank_results"
    ]["4"]["sides"]["right"]["recurrent_cross_layer_edges"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == "step143000")
    factors = raw_qk_affine_factors(Path(record["snapshot"]))
    head_count = data["discovery_query_pre_rope"].shape[2]
    events_by_recipient = {}
    for edge in family:
        location = (edge["second_layer"], edge["second_head"])
        if location in events_by_recipient:
            continue
        layer, head = location
        query = data["discovery_query_post_rope"][:, layer, head]
        key = data["discovery_key_post_rope"][:, layer, head]
        mean, scale = relative_offset_statistics(qk_logits(query, key))
        events_by_recipient[location] = (
            head_events(data, "discovery", layer, head, mean, scale),
            head_events(data, "confirmation", layer, head, mean, scale),
        )

    reports_by_rank: dict[str, dict[str, object]] = {}
    for rank in args.ranks:
        reports = []
        control_scores_by_edge = []
        for edge in family:
            donor_location = (edge["first_layer"], edge["first_head"])
            recipient_location = (edge["second_layer"], edge["second_head"])
            recipient_layer, recipient_head = recipient_location
            discovery, confirmation = events_by_recipient[recipient_location]
            target_discovery = qk_margins(
                discovery.q_rotated, discovery.k_positive_rotated - discovery.k_negative_rotated
            )
            target_confirmation = qk_margins(
                confirmation.q_rotated, confirmation.k_positive_rotated - confirmation.k_negative_rotated
            )
            target_keys = data["discovery_key_post_rope"][:, recipient_layer, recipient_head]

            candidate_scores = []
            for candidate_head in range(head_count):
                candidate_location = (donor_location[0], candidate_head)
                donor_discovery = rotated_keys(
                    data["discovery_normalized_residual"][:, recipient_layer],
                    factors[candidate_location][1],
                )
                donor_confirmation = rotated_keys(
                    data["confirmation_normalized_residual"][:, recipient_layer],
                    factors[candidate_location][1],
                )
                transform = alignment(donor_discovery, target_keys, rank)
                discovery_score = scores(discovery, donor_discovery, transform)
                confirmation_score = scores(confirmation, donor_confirmation, transform)
                candidate_scores.append(
                    r_squared(
                        calibrated_prediction(discovery_score, target_discovery, confirmation_score),
                        target_confirmation,
                    )
                )
            observed = candidate_scores[donor_location[1]]
            controls = [
                value for head, value in enumerate(candidate_scores) if head != donor_location[1]
            ]
            reports.append(
                {
                    "donor": f"L{donor_location[0]}H{donor_location[1]}",
                    "recipient": f"L{recipient_location[0]}H{recipient_location[1]}",
                    "key_side_transfer_r2": observed,
                    "alternative_donor_mean_r2": float(np.mean(controls)),
                    "alternative_donor_rank": int(1 + np.sum(np.asarray(controls) >= observed)),
                    "alternative_donor_count": len(controls),
                }
            )
            control_scores_by_edge.append(np.asarray(controls))
        observed = np.asarray([item["key_side_transfer_r2"] for item in reports])
        rng = np.random.default_rng(args.seed)
        null_means = np.asarray(
            [
                np.mean([values[rng.integers(len(values))] for values in control_scores_by_edge])
                for _ in range(args.resamples)
            ]
        )
        reports_by_rank[str(rank)] = {
            "edges": reports,
            "mean_key_side_transfer_r2": float(np.mean(observed)),
            "mean_alternative_donor_r2": float(
                np.mean([item["alternative_donor_mean_r2"] for item in reports])
            ),
            "source_layer_control_null_mean": float(np.mean(null_means)),
            "source_layer_control_null_standard_deviation": float(np.std(null_means)),
            "source_layer_control_upper_tail_p_value": float(
                (1 + np.sum(null_means >= np.mean(observed))) / (1 + len(null_means))
            ),
        }
        print(f"completed key-side rank {rank}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "recipient-query, donor-key-side transfer with orthogonal gauge alignment",
                "selection": "rank-four recurrent QK key-side edges from the static family audit",
                "control": "other donor heads from the same donor layer",
                "ranks": reports_by_rank,
                "seed": args.seed,
                "resamples": args.resamples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
