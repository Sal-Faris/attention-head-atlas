"""Test whether unsupervised contextual input classes gate QK channels."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from analyze_qk_conditional_subspaces import events_for_head
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from sklearn.mixture import GaussianMixture

from head_atlas.bilinear import fit_bilinear_margin_model
from head_atlas.conditional_qk import benjamini_hochberg
from head_atlas.qk_events import qk_logits, relative_offset_statistics


@dataclass(frozen=True)
class InputClassModel:
    pca: PCA
    mixture: GaussianMixture
    selected_components: int


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
    parser.add_argument("--max-classes", type=int, default=8)
    parser.add_argument("--pca-dimensions", type=int, default=32)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_channel_input_classes_v1.json"),
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


def event_residuals(residual: np.ndarray, events: object, *, source: bool) -> np.ndarray:
    positions = events.positive_sources if source else events.destinations
    return residual[events.documents, positions]


def fit_input_classes(
    values: np.ndarray, max_classes: int, pca_dimensions: int, seed: int
) -> InputClassModel:
    dimensions = min(pca_dimensions, values.shape[0] - 1, values.shape[1])
    pca = PCA(n_components=dimensions, whiten=True, random_state=seed)
    projected = pca.fit_transform(values)
    candidates = [
        GaussianMixture(
            n_components=count,
            covariance_type="diag",
            random_state=seed,
            reg_covar=1e-4,
            max_iter=200,
        ).fit(projected)
        for count in range(1, max_classes + 1)
    ]
    scores = np.asarray([candidate.bic(projected) for candidate in candidates])
    selected = int(np.argmin(scores))
    return InputClassModel(pca, candidates[selected], selected + 1)


def assign(model: InputClassModel, values: np.ndarray) -> np.ndarray:
    return model.mixture.predict(model.pca.transform(values))


def conditional_mi_test(
    input_classes: np.ndarray,
    components: np.ndarray,
    offset_bins: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Test class/component association while preserving offset effects."""

    observed = float(mutual_info_score(input_classes, components))
    null = np.empty(permutations)
    for iteration in range(permutations):
        shuffled = components.copy()
        for offset in np.unique(offset_bins):
            indices = np.flatnonzero(offset_bins == offset)
            shuffled[indices] = shuffled[indices][rng.permutation(len(indices))]
        null[iteration] = mutual_info_score(input_classes, shuffled)
    return {
        "mutual_information": observed,
        "null_mean": float(null.mean()),
        "excess_mutual_information": float(observed - null.mean()),
        "upper_tail_p": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
    }


def head_locations(family_audit: Path) -> list[tuple[int, int]]:
    payload = json.loads(family_audit.read_text(encoding="utf-8"))
    families = payload["views"]["QK"]["rank_results"]["4"]["sides"]["right"][
        "recurrent_cross_layer_edges"
    ]
    return sorted(
        {
            (record[f"{prefix}_layer"], record[f"{prefix}_head"])
            for record in families
            for prefix in ("first", "second")
        }
    )


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.max_classes < 1 or args.pca_dimensions < 1:
        raise ValueError("iterations, max-classes, and pca-dimensions must be positive")
    with np.load(args.input, allow_pickle=False) as source:
        data = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))["heads"]
    reports = {}
    raw_p_values = []
    raw_keys = []
    for layer, head in head_locations(args.family_audit):
        discovery_q = data["discovery_query_post_rope"][:, layer, head]
        discovery_k = data["discovery_key_post_rope"][:, layer, head]
        mean, scale = relative_offset_statistics(qk_logits(discovery_q, discovery_k))
        discovery = event_table(data, "discovery", layer, head, mean, scale)
        confirmation = event_table(data, "confirmation", layer, head, mean, scale)
        ridge = float(tuning[f"L{layer}H{head}"]["4"]["selected_ridge"])
        channel = fit_bilinear_margin_model(
            discovery.q_rotated,
            discovery.k_positive_rotated - discovery.k_negative_rotated,
            rank=4,
            ridge=ridge,
            iterations=args.iterations,
        )
        contributions = (
            (confirmation.q_rotated @ channel.left)
            * ((confirmation.k_positive_rotated - confirmation.k_negative_rotated) @ channel.right)
        )
        dominant = np.argmax(np.abs(contributions), axis=1)
        residual = data["discovery_normalized_residual"][:, layer]
        confirmation_residual = data["confirmation_normalized_residual"][:, layer]
        query_model = fit_input_classes(
            event_residuals(residual, discovery, source=False),
            args.max_classes,
            args.pca_dimensions,
            args.seed + layer * 10 + head,
        )
        key_model = fit_input_classes(
            event_residuals(residual, discovery, source=True),
            args.max_classes,
            args.pca_dimensions,
            args.seed + 1000 + layer * 10 + head,
        )
        query_test = conditional_mi_test(
            assign(query_model, event_residuals(confirmation_residual, confirmation, source=False)),
            dominant,
            confirmation.bins,
            args.permutations,
            np.random.default_rng(args.seed + 2000 + layer * 10 + head),
        )
        key_test = conditional_mi_test(
            assign(key_model, event_residuals(confirmation_residual, confirmation, source=True)),
            dominant,
            confirmation.bins,
            args.permutations,
            np.random.default_rng(args.seed + 3000 + layer * 10 + head),
        )
        location = f"L{layer}H{head}"
        reports[location] = {
            "discovery_events": len(discovery.q_rotated),
            "confirmation_events": len(confirmation.q_rotated),
            "query_input_classes": query_model.selected_components,
            "key_input_classes": key_model.selected_components,
            "query_test": query_test,
            "key_test": key_test,
        }
        for side, test in (("query", query_test), ("key", key_test)):
            raw_keys.append((location, side))
            raw_p_values.append(test["upper_tail_p"])
        print(f"tested L{layer}H{head}", flush=True)
    adjusted = benjamini_hochberg(np.asarray(raw_p_values))
    for (location, side), value in zip(raw_keys, adjusted, strict=True):
        reports[location][f"{side}_test"]["bh_q"] = float(value)
    result = {
        "status": "held-out input-class gating test for rank-4 recurrent key-side QK channels",
        "class_discovery": "per-head PCA-whitened diagonal-GMM BIC selection on discovery residual inputs",
        "null": "permute component labels within exact relative-offset bins on confirmation events",
        "max_classes": args.max_classes,
        "pca_dimensions": args.pca_dimensions,
        "permutations": args.permutations,
        "heads": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
