"""Show representative held-out inputs for the strongest stable QK class effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head
from test_qk_channel_input_classes import event_residuals, fit_input_classes

from head_atlas.bilinear import fit_bilinear_margin_model
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
        "--full-test",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_channel_input_classes_v1.json"),
    )
    parser.add_argument(
        "--coarse-test",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_channel_input_classes_k4_sensitivity.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument("--top-sides", type=int, default=6)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_channel_input_class_profiles_v1.json"),
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


def candidate_sides(full: dict[str, object], coarse: dict[str, object], count: int) -> list[tuple[str, str]]:
    candidates = []
    for location, report in full["heads"].items():
        for side in ("query", "key"):
            full_test = report[f"{side}_test"]
            coarse_test = coarse["heads"][location][f"{side}_test"]
            if full_test["bh_q"] < 0.05 and coarse_test["bh_q"] < 0.05:
                candidates.append(
                    (
                        min(full_test["excess_mutual_information"], coarse_test["excess_mutual_information"]),
                        location,
                        side,
                    )
                )
    return [(location, side) for _, location, side in sorted(candidates, reverse=True)[:count]]


def context(tokenizer: object, tokens: np.ndarray, position: int) -> dict[str, object]:
    start, stop = max(0, position - 4), min(len(tokens), position + 5)
    return {
        "token": tokenizer.decode([int(tokens[position])]),
        "context": tokenizer.decode(tokens[start:stop].tolist()),
        "position": int(position),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    full = json.loads(args.full_test.read_text(encoding="utf-8"))
    coarse = json.loads(args.coarse_test.read_text(encoding="utf-8"))
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["heads"]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    snapshot = next(record["snapshot"] for record in manifest["records"] if record["revision"] == "step143000")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    profiles = {}
    for location, side in candidate_sides(full, coarse, args.top_sides):
        layer, head = int(location[1]), int(location[3])
        discovery_q = data["discovery_query_post_rope"][:, layer, head]
        discovery_k = data["discovery_key_post_rope"][:, layer, head]
        mean, scale = relative_offset_statistics(qk_logits(discovery_q, discovery_k))
        discovery = events(data, "discovery", layer, head, mean, scale)
        confirmation = events(data, "confirmation", layer, head, mean, scale)
        source_side = side == "key"
        class_model = fit_input_classes(
            event_residuals(data["discovery_normalized_residual"][:, layer], discovery, source=source_side),
            max_classes=4,
            pca_dimensions=32,
            seed=(1000 if source_side else 0) + layer * 10 + head,
        )
        values = event_residuals(
            data["confirmation_normalized_residual"][:, layer], confirmation, source=source_side
        )
        projected = class_model.pca.transform(values)
        labels = class_model.mixture.predict(projected)
        ridge = float(tuning[location]["4"]["selected_ridge"])
        channel = fit_bilinear_margin_model(
            discovery.q_rotated,
            discovery.k_positive_rotated - discovery.k_negative_rotated,
            rank=4,
            ridge=ridge,
            iterations=args.iterations,
        )
        contributions = (confirmation.q_rotated @ channel.left) * (
            (confirmation.k_positive_rotated - confirmation.k_negative_rotated) @ channel.right
        )
        dominant = np.argmax(np.abs(contributions), axis=1)
        classes = []
        for label in range(class_model.selected_components):
            indices = np.flatnonzero(labels == label)
            if len(indices) == 0:
                classes.append(
                    {
                        "class": label,
                        "event_fraction": 0.0,
                        "dominant_component_distribution": [None] * 4,
                        "examples": [],
                    }
                )
                continue
            distances = np.sum(
                (projected[indices] - class_model.mixture.means_[label]) ** 2
                / class_model.mixture.covariances_[label],
                axis=1,
            )
            representatives = []
            seen_documents = set()
            for index in indices[np.argsort(distances)]:
                document = int(confirmation.documents[index])
                if document not in seen_documents:
                    representatives.append(index)
                    seen_documents.add(document)
                if len(representatives) == args.examples:
                    break
            positions = confirmation.positive_sources if source_side else confirmation.destinations
            classes.append(
                {
                    "class": label,
                    "event_fraction": float(len(indices) / len(labels)),
                    "dominant_component_distribution": (
                        np.bincount(dominant[indices], minlength=4) / len(indices)
                    ).tolist(),
                    "examples": [
                        context(
                            tokenizer,
                            data["confirmation_tokens"][confirmation.documents[index]],
                            int(positions[index]),
                        )
                        for index in representatives
                    ],
                }
            )
        profiles[f"{location}_{side}"] = {
            "full_test": full["heads"][location][f"{side}_test"],
            "coarse_test": coarse["heads"][location][f"{side}_test"],
            "coarse_class_count": class_model.selected_components,
            "classes": classes,
        }
        print(f"profiled {location} {side}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"profiles": profiles}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
