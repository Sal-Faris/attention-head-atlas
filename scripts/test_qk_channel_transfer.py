"""Test whether recurrent heads transfer learned QK channels in residual coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head, rope

from head_atlas.bilinear import fit_bilinear_margin_model, qk_margins, r_squared
from head_atlas.factor_io import load_factor_bundle
from head_atlas.qk_events import qk_logits, relative_offset_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/pythia-70m-deduped/qk_bilinear_margin_confirmation_v1.npz"))
    parser.add_argument("--tuning", type=Path, default=Path("results/pythia-70m-deduped/qk_bilinear_margin_tuning_step143000_iter200_v1.json"))
    parser.add_argument("--family-audit", type=Path, default=Path("results/pythia-70m-deduped/subspace_family_audit.json"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pythia-70m-deduped-pilot.json"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--resamples", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/pythia-70m-deduped/qk_channel_transfer_v1.json"))
    return parser.parse_args()


def events(data: dict[str, np.ndarray], split: str, layer: int, head: int, mean: np.ndarray, scale: np.ndarray):
    return events_for_head(data[f"{split}_query_pre_rope"][:, layer, head], data[f"{split}_key_pre_rope"][:, layer, head], data[f"{split}_query_post_rope"][:, layer, head], data[f"{split}_key_post_rope"][:, layer, head], mean, scale)


def calibrated_prediction(discovery_score: np.ndarray, discovery_target: np.ndarray, confirmation_score: np.ndarray) -> np.ndarray:
    design = np.column_stack((discovery_score, np.ones(len(discovery_score))))
    slope, intercept = np.linalg.lstsq(design, discovery_target, rcond=None)[0]
    return slope * confirmation_score + intercept


def transferred_scores(residual: np.ndarray, recipient: object, donor_operator: object, channel: np.ndarray) -> np.ndarray:
    query_residual = residual[recipient.documents, recipient.destinations]
    positive_residual = residual[recipient.documents, recipient.positive_sources]
    negative_residual = residual[recipient.documents, recipient.negative_sources]
    query = rope(query_residual @ donor_operator.left, recipient.destinations)
    positive = rope(positive_residual @ donor_operator.right, recipient.positive_sources)
    negative = rope(negative_residual @ donor_operator.right, recipient.negative_sources)
    return np.sum((query @ channel) * (positive - negative), axis=1) / np.sqrt(query.shape[1])


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["heads"]
    family = json.loads(args.family_audit.read_text(encoding="utf-8"))["views"]["QK"]["rank_results"]["4"]["sides"]["right"]["recurrent_cross_layer_edges"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == "step143000")
    operators, _ = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    operator_by_head = {(operator.layer, operator.head): operator for operator in operators}
    heads_by_layer = {
        layer: sorted(head for current_layer, head in operator_by_head if current_layer == layer)
        for layer in {operator.layer for operator in operators}
    }
    channel_by_head = {}
    event_by_head = {}
    for operator in operators:
        layer, head = operator.layer, operator.head
        q = data["discovery_query_post_rope"][:, layer, head]
        k = data["discovery_key_post_rope"][:, layer, head]
        mean, scale = relative_offset_statistics(qk_logits(q, k))
        discovery = events(data, "discovery", layer, head, mean, scale)
        confirmation = events(data, "confirmation", layer, head, mean, scale)
        model = fit_bilinear_margin_model(discovery.q_rotated, discovery.k_positive_rotated - discovery.k_negative_rotated, rank=4, ridge=float(tuning[f"L{layer}H{head}"]["4"]["selected_ridge"]), iterations=args.iterations)
        channel_by_head[(layer, head)] = model.left @ model.right.T
        event_by_head[(layer, head)] = (discovery, confirmation)
    reports = []
    alternative_scores_by_edge = []
    for edge in family:
        donor_location = (edge["first_layer"], edge["first_head"])
        recipient_location = (edge["second_layer"], edge["second_head"])
        discovery, confirmation = event_by_head[recipient_location]
        donor = operator_by_head[donor_location]
        discovery_score = transferred_scores(data["discovery_normalized_residual"][:, recipient_location[0]], discovery, donor, channel_by_head[donor_location])
        confirmation_score = transferred_scores(data["confirmation_normalized_residual"][:, recipient_location[0]], confirmation, donor, channel_by_head[donor_location])
        target = qk_margins(confirmation.q_rotated, confirmation.k_positive_rotated - confirmation.k_negative_rotated)
        observed_r2 = r_squared(
            calibrated_prediction(
                discovery_score,
                qk_margins(
                    discovery.q_rotated,
                    discovery.k_positive_rotated - discovery.k_negative_rotated,
                ),
                confirmation_score,
            ),
            target,
        )
        controls = []
        for alternative_head in heads_by_layer[donor_location[0]]:
            alternative_location = (donor_location[0], alternative_head)
            if alternative_location == donor_location:
                continue
            alternative = operator_by_head[alternative_location]
            alternative_discovery = transferred_scores(
                data["discovery_normalized_residual"][:, recipient_location[0]],
                discovery,
                alternative,
                channel_by_head[alternative_location],
            )
            alternative_confirmation = transferred_scores(
                data["confirmation_normalized_residual"][:, recipient_location[0]],
                confirmation,
                alternative,
                channel_by_head[alternative_location],
            )
            controls.append(
                r_squared(
                    calibrated_prediction(
                        alternative_discovery,
                        qk_margins(
                            discovery.q_rotated,
                            discovery.k_positive_rotated - discovery.k_negative_rotated,
                        ),
                        alternative_confirmation,
                    ),
                    target,
                )
            )
        reports.append(
            {
                "donor": f"L{donor_location[0]}H{donor_location[1]}",
                "recipient": f"L{recipient_location[0]}H{recipient_location[1]}",
                "transfer_r2": observed_r2,
                "alternative_donor_mean_r2": float(np.mean(controls)),
                "alternative_donor_rank": int(1 + np.sum(np.asarray(controls) >= observed_r2)),
                "alternative_donor_count": len(controls),
            }
        )
        alternative_scores_by_edge.append(np.asarray(controls))
        print(f"transferred {reports[-1]['donor']} -> {reports[-1]['recipient']}", flush=True)
    observed = np.asarray([report["transfer_r2"] for report in reports])
    alternatives = np.asarray([report["alternative_donor_mean_r2"] for report in reports])
    rng = np.random.default_rng(args.seed)
    null_means = np.empty(args.resamples)
    for index in range(args.resamples):
        null_means[index] = np.mean(
            [scores[rng.integers(len(scores))] for scores in alternative_scores_by_edge]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    observed_mean = float(np.mean(observed))
    args.output.write_text(
        json.dumps(
            {
                "status": "calibrated residual-space QK channel transfer with source-layer controls",
                "control": "Alternative donor heads from the same donor layer; recipient and calibration procedure fixed.",
                "edges": reports,
                "mean_transfer_r2": observed_mean,
                "mean_alternative_donor_r2": float(np.mean(alternatives)),
                "source_layer_control_null_mean": float(np.mean(null_means)),
                "source_layer_control_null_standard_deviation": float(np.std(null_means)),
                "source_layer_control_upper_tail_p_value": float(
                    (1 + np.sum(null_means >= observed_mean)) / (1 + len(null_means))
                ),
                "seed": args.seed,
                "resamples": args.resamples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
