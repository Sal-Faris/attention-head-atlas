"""Test whether static head geometry predicts held-out activation behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from head_atlas.activation import (
    distance_spearman,
    normalized_distances_from_gram,
    stratified_distance_permutation_test,
    stratified_predictor_difference_test,
)
from head_atlas.distances import chordal_subspace_distances, weighted_product_distances
from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import (
    factorized_singular_components,
    normalized_factorized_frobenius_distances,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("D:/Laptop/AI/model-cache/huggingface"),
    )
    parser.add_argument("--model-revision", default="step143000")
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument(
        "--dataset-revision",
        default="127bfedcd5047750df5ccf3a12979a47bfa0bafa",
    )
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--sequences-per-split", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/activation_validation_pilot.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/activation_validation_pilot.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/activation_validation_pilot.png"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_token_splits(
    dataset_snapshot: Path,
    tokenizer: object,
    *,
    sequence_length: int,
    sequences_per_split: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parquet_files = sorted((dataset_snapshot / "data").glob("*.parquet"))
    if len(parquet_files) != 1:
        raise ValueError("expected exactly one cached dataset parquet file")
    texts = pd.read_parquet(parquet_files[0], columns=["text"])["text"]
    rng = np.random.default_rng(seed)
    required = 2 * sequences_per_split
    selected_tokens = []
    selected_rows = []
    prefix_token = tokenizer.bos_token_id
    if prefix_token is None:
        prefix_token = tokenizer.eos_token_id
    if prefix_token is None:
        raise ValueError("tokenizer has neither a BOS nor EOS token")

    for row_index in rng.permutation(len(texts)):
        text = str(texts.iloc[int(row_index)])[:8192]
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=sequence_length - 1,
        )["input_ids"]
        if len(token_ids) != sequence_length - 1:
            continue
        selected_tokens.append([int(prefix_token), *token_ids])
        selected_rows.append(int(row_index))
        if len(selected_tokens) == required:
            break
    if len(selected_tokens) != required:
        raise RuntimeError("cached dataset does not contain enough usable documents")

    tokens = np.asarray(selected_tokens, dtype=np.int64)
    rows = np.asarray(selected_rows, dtype=np.int64)
    return (
        tokens[:sequences_per_split],
        tokens[sequences_per_split:],
        rows[:sequences_per_split],
        rows[sequences_per_split:],
    )


def factor_tensors(
    operators: list[object],
    *,
    layers: int,
    heads: int,
    torch: object,
) -> tuple[list[object], list[object]]:
    left = []
    right = []
    for layer in range(layers):
        layer_operators = [operator for operator in operators if operator.layer == layer]
        if [operator.head for operator in layer_operators] != list(range(heads)):
            raise ValueError("factor operators are not in canonical head order")
        left.append(
            torch.as_tensor(np.stack([operator.left for operator in layer_operators]))
        )
        right.append(
            torch.as_tensor(np.stack([operator.right for operator in layer_operators]))
        )
    return left, right


def update_gram(gram: np.ndarray, values: object, *, centered: bool) -> None:
    matrix = values.movedim(1, 0).reshape(values.shape[1], -1).float()
    if centered:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    gram += (matrix @ matrix.T).double().cpu().numpy()


def collect_behavior_grams(
    model: object,
    tokens: np.ndarray,
    ov_operators: list[object],
    *,
    batch_size: int,
    torch: object,
) -> dict[str, np.ndarray]:
    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    population = layer_count * head_count
    left_factors, right_factors = factor_tensors(
        ov_operators,
        layers=layer_count,
        heads=head_count,
        torch=torch,
    )
    grams = {
        "attention_raw": np.zeros((population, population), dtype=np.float64),
        "attention_centered": np.zeros((population, population), dtype=np.float64),
        "ov_response_raw": np.zeros((population, population), dtype=np.float64),
        "ov_response_centered": np.zeros((population, population), dtype=np.float64),
        "head_result_raw": np.zeros((population, population), dtype=np.float64),
        "head_result_centered": np.zeros((population, population), dtype=np.float64),
    }
    epsilon = float(model.config.layer_norm_eps)
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            input_ids = torch.as_tensor(tokens[start : start + batch_size])
            outputs = model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            attention = torch.cat(outputs.attentions, dim=1)
            update_gram(grams["attention_raw"], attention, centered=False)
            update_gram(grams["attention_centered"], attention, centered=True)

            responses = []
            head_results = []
            for layer in range(layer_count):
                residual = outputs.hidden_states[layer].float()
                centered_residual = residual - residual.mean(dim=-1, keepdim=True)
                normalized = centered_residual * torch.rsqrt(
                    centered_residual.square().mean(dim=-1, keepdim=True) + epsilon
                )
                values = torch.einsum(
                    "bpd,hdk->bhpk", normalized, left_factors[layer]
                )
                layer_response = torch.einsum(
                    "bhpk,hdk->bhpd", values, right_factors[layer]
                )
                responses.append(layer_response)
                head_results.append(
                    torch.einsum(
                        "bhqs,bhsd->bhqd",
                        outputs.attentions[layer],
                        layer_response,
                    )
                )
            response = torch.cat(responses, dim=1)
            update_gram(grams["ov_response_raw"], response, centered=False)
            update_gram(grams["ov_response_centered"], response, centered=True)
            head_result = torch.cat(head_results, dim=1)
            update_gram(grams["head_result_raw"], head_result, centered=False)
            update_gram(grams["head_result_centered"], head_result, centered=True)
            print(
                f"processed {min(start + batch_size, len(tokens))}/{len(tokens)} sequences",
                flush=True,
            )
    return grams


def weight_predictors(
    operators: list[object], ranks: list[int]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    layers = np.asarray([operator.layer for operator in operators], dtype=np.int64)
    heads = np.asarray([operator.head for operator in operators], dtype=np.int64)
    left_bases = []
    right_bases = []
    for operator in operators:
        left, _, right = factorized_singular_components(operator)
        left_bases.append(left)
        right_bases.append(right)
    left_array = np.stack(left_bases)
    right_array = np.stack(right_bases)
    predictors = {"raw_operator": normalized_factorized_frobenius_distances(operators)}
    for rank in ranks:
        left_distances = chordal_subspace_distances(left_array[:, :, :rank])
        right_distances = chordal_subspace_distances(right_array[:, :, :rank])
        predictors[f"left_rank_{rank}"] = left_distances
        predictors[f"right_rank_{rank}"] = right_distances
        predictors[f"joint_rank_{rank}"] = weighted_product_distances(
            [left_distances, right_distances]
        )
    maximum_layer = max(int(np.max(layers)), 1)
    predictors["layer_depth_baseline"] = (
        np.abs(layers[:, None] - layers[None, :]) / maximum_layer
    )
    return predictors, layers, heads


def evaluate_predictors(
    predictors: dict[str, np.ndarray],
    train_target: np.ndarray,
    test_target: np.ndarray,
    layers: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    records = {}
    selectable = [name for name in predictors if not name.endswith("baseline")]
    for index, (name, distances) in enumerate(predictors.items()):
        records[name] = {
            "train_spearman": distance_spearman(distances, train_target),
            "held_out_test": stratified_distance_permutation_test(
                distances,
                test_target,
                layers,
                repetitions=permutations,
                rng=np.random.default_rng(seed + 1000 + index),
            ),
        }
    selected = max(selectable, key=lambda name: records[name]["train_spearman"])
    subspace_selectable = [name for name in selectable if "rank_" in name]
    if not subspace_selectable:
        raise ValueError("at least one subspace predictor is required")
    selected_subspace = max(
        subspace_selectable, key=lambda name: records[name]["train_spearman"]
    )
    return {
        "selection_rule": "maximum train-split Spearman among weight predictors",
        "selected_predictor": selected,
        "selected_subspace_predictor": selected_subspace,
        "predictors": records,
        "selected_vs_raw_on_held_out_test": stratified_predictor_difference_test(
            predictors[selected],
            predictors["raw_operator"],
            test_target,
            layers,
            repetitions=permutations,
            rng=np.random.default_rng(seed + 5000),
        ),
        "selected_subspace_vs_raw_on_held_out_test": stratified_predictor_difference_test(
            predictors[selected_subspace],
            predictors["raw_operator"],
            test_target,
            layers,
            repetitions=permutations,
            rng=np.random.default_rng(seed + 6000),
        ),
    }


def raw_behavior_sensitivity(
    predictors: dict[str, np.ndarray],
    selected: str,
    train_target: np.ndarray,
    test_target: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "train_spearman": distance_spearman(predictors[name], train_target),
            "test_spearman": distance_spearman(predictors[name], test_target),
        }
        for name in dict.fromkeys((selected, "raw_operator", "layer_depth_baseline"))
    }


def plot_report(report: dict[str, object], output: Path) -> None:
    view_order = ("QK", "OV", "HEAD_RESULT")
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    titles = {
        "QK": "QK: centered attention-pattern behavior",
        "OV": "OV: centered activation-conditioned response",
        "HEAD_RESULT": "Composed head: centered attention-weighted output",
    }
    for row, view in enumerate(view_order):
        result = report["views"][view]["centered_behavior_evaluation"]
        predictor_results = result["predictors"]
        names = list(predictor_results)
        train = [predictor_results[name]["train_spearman"] for name in names]
        test = [
            predictor_results[name]["held_out_test"]["observed_spearman"]
            for name in names
        ]
        positions = np.arange(len(names))
        axes[row, 0].barh(positions + 0.18, train, height=0.34, label="train")
        axes[row, 0].barh(positions - 0.18, test, height=0.34, label="held-out test")
        axes[row, 0].set_yticks(positions, names, fontsize=8)
        axes[row, 0].axvline(0.0, color="black", linewidth=0.8)
        axes[row, 0].set_xlabel("weight/behavior distance Spearman")
        axes[row, 0].set_ylabel(view)
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(axis="x", alpha=0.25)

        selected = result["selected_predictor"]
        selected_subspace = result["selected_subspace_predictor"]
        comparison_names = list(
            dict.fromkeys(
                (selected, selected_subspace, "raw_operator", "layer_depth_baseline")
            )
        )
        observed = [
            predictor_results[name]["held_out_test"]["observed_spearman"]
            for name in comparison_names
        ]
        null_mean = [
            predictor_results[name]["held_out_test"]["null_mean"]
            for name in comparison_names
        ]
        null_deviation = [
            predictor_results[name]["held_out_test"]["null_standard_deviation"]
            for name in comparison_names
        ]
        x = np.arange(len(comparison_names))
        axes[row, 1].bar(x, observed, width=0.55, label="held-out observed")
        axes[row, 1].errorbar(
            x,
            null_mean,
            yerr=null_deviation,
            fmt="o",
            color="black",
            capsize=4,
            label="within-layer shuffle",
        )
        axes[row, 1].set_xticks(x, comparison_names, rotation=15, ha="right")
        axes[row, 1].set_ylabel("held-out Spearman")
        axes[row, 1].grid(axis="y", alpha=0.25)
        axes[row, 1].legend(fontsize=8)
        axes[row, 0].set_title(titles[view])
        axes[row, 1].set_title(f"{view} selected metric versus controls")
    figure.suptitle(
        "Pythia-70M static geometry predicting held-out activation behavior",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ranks = sorted(set(args.ranks))
    if args.sequence_length < 8 or args.sequences_per_split < 2 or args.batch_size < 1:
        raise ValueError("invalid corpus or batch dimensions")
    if not ranks or ranks[0] < 1 or args.permutations < 1:
        raise ValueError("ranks and permutations must be positive")

    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit('Install model dependencies with: pip install -e ".[models]"') from error

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        record for record in manifest["records"] if record["revision"] == args.model_revision
    ]
    if len(records) != 1:
        raise ValueError("model revision is absent or duplicated in the manifest")
    record = records[0]
    model_snapshot = Path(record["snapshot"])
    dataset_snapshot = Path(
        snapshot_download(
            args.dataset,
            repo_type="dataset",
            revision=args.dataset_revision,
            cache_dir=args.cache_root,
            local_files_only=True,
            allow_patterns=["README.md", "data/*.parquet"],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
    train_tokens, test_tokens, train_rows, test_rows = load_token_splits(
        dataset_snapshot,
        tokenizer,
        sequence_length=args.sequence_length,
        sequences_per_split=args.sequences_per_split,
        seed=args.seed,
    )
    print("prepared deterministic train/test token splits", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    qk_operators, _ = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    ov_operators, _ = load_factor_bundle(Path(record["factors"]["OV"]["path"]))
    if max(ranks) > qk_operators[0].d_head:
        raise ValueError("requested rank exceeds the head width")

    print("collecting train activation behavior", flush=True)
    train_grams = collect_behavior_grams(
        model,
        train_tokens,
        ov_operators,
        batch_size=args.batch_size,
        torch=torch,
    )
    print("collecting held-out activation behavior", flush=True)
    test_grams = collect_behavior_grams(
        model,
        test_tokens,
        ov_operators,
        batch_size=args.batch_size,
        torch=torch,
    )
    train_behavior = {
        name: normalized_distances_from_gram(gram) for name, gram in train_grams.items()
    }
    test_behavior = {
        name: normalized_distances_from_gram(gram) for name, gram in test_grams.items()
    }

    qk_predictors, qk_layers, qk_heads = weight_predictors(qk_operators, ranks)
    ov_predictors, ov_layers, ov_heads = weight_predictors(ov_operators, ranks)
    if not np.array_equal(qk_layers, ov_layers) or not np.array_equal(qk_heads, ov_heads):
        raise ValueError("QK and OV populations are not aligned")

    head_result_predictors = {
        "raw_operator": weighted_product_distances(
            [qk_predictors["raw_operator"], ov_predictors["raw_operator"]]
        ),
        "qk_raw_only": qk_predictors["raw_operator"],
        "ov_raw_only": ov_predictors["raw_operator"],
        "layer_depth_baseline": qk_predictors["layer_depth_baseline"],
    }
    for rank in ranks:
        head_result_predictors[f"qk_subspace_rank_{rank}"] = qk_predictors[
            f"joint_rank_{rank}"
        ]
        head_result_predictors[f"ov_subspace_rank_{rank}"] = ov_predictors[
            f"joint_rank_{rank}"
        ]
        head_result_predictors[f"qk_ov_subspace_rank_{rank}"] = weighted_product_distances(
            [
                qk_predictors[f"joint_rank_{rank}"],
                ov_predictors[f"joint_rank_{rank}"],
            ]
        )

    view_inputs = {
        "QK": (
            qk_predictors,
            "attention_centered",
            "attention_raw",
            0,
        ),
        "OV": (
            ov_predictors,
            "ov_response_centered",
            "ov_response_raw",
            10000,
        ),
        "HEAD_RESULT": (
            head_result_predictors,
            "head_result_centered",
            "head_result_raw",
            20000,
        ),
    }
    views = {}
    for view, (predictors, centered_name, raw_name, seed_offset) in view_inputs.items():
        evaluation = evaluate_predictors(
            predictors,
            train_behavior[centered_name],
            test_behavior[centered_name],
            qk_layers,
            permutations=args.permutations,
            seed=args.seed + seed_offset,
        )
        views[view] = {
            "behavior_target": centered_name,
            "split_reliability": stratified_distance_permutation_test(
                train_behavior[centered_name],
                test_behavior[centered_name],
                qk_layers,
                repetitions=args.permutations,
                rng=np.random.default_rng(args.seed + seed_offset + 9000),
            ),
            "centered_behavior_evaluation": evaluation,
            "raw_behavior_sensitivity": raw_behavior_sensitivity(
                predictors,
                evaluation["selected_predictor"],
                train_behavior[raw_name],
                test_behavior[raw_name],
            ),
        }

    artifact_payload = {
        "train_tokens": train_tokens,
        "test_tokens": test_tokens,
        "train_dataset_rows": train_rows,
        "test_dataset_rows": test_rows,
        "layers": qk_layers,
        "heads": qk_heads,
    }
    for split, behavior in (("train", train_behavior), ("test", test_behavior)):
        for name, distances in behavior.items():
            artifact_payload[f"{split}_{name}_distances"] = distances
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact, **artifact_payload)

    report = {
        "analysis_status": "held-out activation-behavior validation pilot",
        "model": manifest["model"],
        "model_revision": args.model_revision,
        "model_snapshot_commit": record["snapshot_commit"],
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "dataset_source": "first 10K examples of The Pile; deterministic document sample",
        "sequence_length": args.sequence_length,
        "sequences_per_split": args.sequences_per_split,
        "train_dataset_rows": train_rows.tolist(),
        "test_dataset_rows": test_rows.tolist(),
        "ranks": ranks,
        "permutations": args.permutations,
        "seed": args.seed,
        "primary_targets": {
            "QK": "attention probabilities centered across heads at each corpus feature",
            "OV": "bias-free processed OV response to actual pre-layer-normalized residuals, centered across heads",
            "HEAD_RESULT": "bias-free OV responses mixed by actual attention probabilities, centered across heads",
        },
        "test_null": "permute target head identities within each layer",
        "selection": "weight predictor selected on train documents; inference on disjoint test documents",
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved activation validation to {args.output} and {args.figure}")


if __name__ == "__main__":
    main()
