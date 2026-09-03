"""Test joint query/key input-class gating of QK channel components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head
from sklearn.metrics import mutual_info_score
from test_qk_channel_input_classes import event_residuals, fit_input_classes, head_locations

from head_atlas.bilinear import fit_bilinear_margin_model
from head_atlas.conditional_qk import benjamini_hochberg
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
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_joint_input_rules_v1.json"),
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


def conditional_mi(component: np.ndarray, variable: np.ndarray, condition: np.ndarray) -> float:
    total = len(component)
    return float(
        sum(
            len(indices) / total * mutual_info_score(component[indices], variable[indices])
            for value in np.unique(condition)
            if len(indices := np.flatnonzero(condition == value)) > 1
        )
    )


def conditional_rule_test(
    component: np.ndarray,
    variable: np.ndarray,
    condition: np.ndarray,
    documents: np.ndarray,
    offsets: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Test CMI after preserving document, offset, and the conditioned side."""

    observed = conditional_mi(component, variable, condition)
    strata = (documents * 6 + offsets) * (int(condition.max()) + 1) + condition
    null = np.empty(permutations)
    for iteration in range(permutations):
        shuffled = component.copy()
        for stratum in np.unique(strata):
            indices = np.flatnonzero(strata == stratum)
            shuffled[indices] = shuffled[indices][rng.permutation(len(indices))]
        null[iteration] = conditional_mi(shuffled, variable, condition)
    return {
        "conditional_mutual_information": observed,
        "null_mean": float(null.mean()),
        "excess_conditional_mutual_information": float(observed - null.mean()),
        "upper_tail_p": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
    }


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["heads"]
    report = {}
    keys, p_values = [], []
    for layer, head in head_locations(args.family_audit):
        discovery_q = data["discovery_query_post_rope"][:, layer, head]
        discovery_k = data["discovery_key_post_rope"][:, layer, head]
        mean, scale = relative_offset_statistics(qk_logits(discovery_q, discovery_k))
        discovery = events(data, "discovery", layer, head, mean, scale)
        confirmation = events(data, "confirmation", layer, head, mean, scale)
        residual = data["discovery_normalized_residual"][:, layer]
        confirmation_residual = data["confirmation_normalized_residual"][:, layer]
        query_model = fit_input_classes(
            event_residuals(residual, discovery, source=False), 4, 32, args.seed + layer * 10 + head
        )
        key_model = fit_input_classes(
            event_residuals(residual, discovery, source=True), 4, 32, args.seed + 1000 + layer * 10 + head
        )
        query_classes = query_model.mixture.predict(
            query_model.pca.transform(event_residuals(confirmation_residual, confirmation, source=False))
        )
        key_classes = key_model.mixture.predict(
            key_model.pca.transform(event_residuals(confirmation_residual, confirmation, source=True))
        )
        ridge = float(tuning[f"L{layer}H{head}"]["4"]["selected_ridge"])
        channel = fit_bilinear_margin_model(
            discovery.q_rotated,
            discovery.k_positive_rotated - discovery.k_negative_rotated,
            rank=4,
            ridge=ridge,
            iterations=args.iterations,
        )
        component = np.argmax(
            np.abs(
                (confirmation.q_rotated @ channel.left)
                * ((confirmation.k_positive_rotated - confirmation.k_negative_rotated) @ channel.right)
            ),
            axis=1,
        )
        location = f"L{layer}H{head}"
        query_given_key = conditional_rule_test(
            component,
            query_classes,
            key_classes,
            confirmation.documents,
            confirmation.bins,
            args.permutations,
            np.random.default_rng(args.seed + 2000 + layer * 10 + head),
        )
        key_given_query = conditional_rule_test(
            component,
            key_classes,
            query_classes,
            confirmation.documents,
            confirmation.bins,
            args.permutations,
            np.random.default_rng(args.seed + 3000 + layer * 10 + head),
        )
        report[location] = {
            "query_classes": query_model.selected_components,
            "key_classes": key_model.selected_components,
            "query_given_key": query_given_key,
            "key_given_query": key_given_query,
        }
        for name, result in (("query_given_key", query_given_key), ("key_given_query", key_given_query)):
            keys.append((location, name))
            p_values.append(result["upper_tail_p"])
        print(f"tested {location}", flush=True)
    for (location, name), value in zip(keys, benjamini_hochberg(np.asarray(p_values)), strict=True):
        report[location][name]["bh_q"] = float(value)
    result = {
        "status": "held-out joint QK input-rule test",
        "input_classes": "discovery-only PCA-whitened diagonal-GMM, BIC over 1..4 classes",
        "null": "permute dominant component within document, offset, and conditioned-side class",
        "permutations": args.permutations,
        "heads": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
