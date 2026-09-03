"""Validate static OV-to-reader edges using held-out activation-weighted action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_typed_composition_edges import edge_tensor
from scipy.stats import spearmanr

from head_atlas.factor_io import load_factor_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/pythia-70m-deduped-pilot.json")
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/qk_bilinear_margin_confirmation_v1.npz"),
    )
    parser.add_argument("--model-revision", default="step143000")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/typed_composition_action_validation_v1.json"),
    )
    return parser.parse_args()


def factors_by_layer(operators: list[object], *, layers: int, heads: int, torch: object) -> list[object]:
    result = []
    for layer in range(layers):
        current = [item for item in operators if item.layer == layer]
        if [item.head for item in current] != list(range(heads)):
            raise ValueError("factor operators are not in canonical head order")
        result.append(torch.as_tensor(np.stack([item.left for item in current])))
    return result


def action_tensor(
    model: object,
    tokens: np.ndarray,
    ov: list[object],
    qk: list[object],
    *,
    batch_size: int,
    torch: object,
) -> dict[str, np.ndarray]:
    """RMS action of actual source-head outputs through target Q/K/V readers."""

    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_attention_heads)
    epsilon = float(model.config.layer_norm_eps)
    value_factors = factors_by_layer(ov, layers=layers, heads=heads, torch=torch)
    writer_factors = []
    query_factors = factors_by_layer(qk, layers=layers, heads=heads, torch=torch)
    key_factors = []
    for layer in range(layers):
        current_ov = [item for item in ov if item.layer == layer]
        current_qk = [item for item in qk if item.layer == layer]
        writer_factors.append(torch.as_tensor(np.stack([item.right for item in current_ov])))
        key_factors.append(torch.as_tensor(np.stack([item.right for item in current_qk])))
    readers = {"Q": query_factors, "K": key_factors, "V": value_factors}
    squared = {kind: np.zeros((layers, layers, heads, heads), dtype=np.float64) for kind in readers}
    count = 0
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
            source_results = []
            for layer in range(layers):
                residual = outputs.hidden_states[layer].float()
                centered = residual - residual.mean(dim=-1, keepdim=True)
                normalized = centered * torch.rsqrt(
                    centered.square().mean(dim=-1, keepdim=True) + epsilon
                )
                values = torch.einsum("bpd,hdk->bhpk", normalized, value_factors[layer])
                response = torch.einsum("bhpk,hdk->bhpd", values, writer_factors[layer])
                source_results.append(
                    torch.einsum("bhqs,bhsd->bhqd", outputs.attentions[layer], response)
                )
            for source_layer in range(layers):
                for target_layer in range(source_layer + 1, layers):
                    for kind, factor_layers in readers.items():
                        action = torch.einsum(
                            "bspd,tdk->bstpk", source_results[source_layer], factor_layers[target_layer]
                        )
                        squared[kind][source_layer, target_layer] += (
                            action.square().sum(dim=(0, 3, 4)).double().cpu().numpy()
                        )
            count += len(input_ids) * input_ids.shape[1] * qk[0].d_head
            print(f"processed {min(start + batch_size, len(tokens))}/{len(tokens)} sequences", flush=True)
    return {kind: np.sqrt(values / count) for kind, values in squared.items()}


def within_pair_prediction(static: np.ndarray, action: np.ndarray, *, resamples: int, rng: np.random.Generator) -> dict[str, float]:
    layer_count, _, head_count, _ = static.shape
    pairs = []
    for source_layer in range(layer_count):
        for target_layer in range(source_layer + 1, layer_count):
            pairs.append((static[source_layer, target_layer], action[source_layer, target_layer]))
    observed = float(np.mean([spearmanr(first.ravel(), second.ravel()).statistic for first, second in pairs]))
    null = []
    for _ in range(resamples):
        scores = []
        for first, second in pairs:
            shuffled = first[rng.permutation(head_count)][:, rng.permutation(head_count)]
            scores.append(float(spearmanr(shuffled.ravel(), second.ravel()).statistic))
        null.append(float(np.mean(scores)))
    null_values = np.asarray(null)
    return {
        "mean_within_layer_pair_spearman": observed,
        "head_identity_shuffle_mean": float(np.mean(null_values)),
        "head_identity_shuffle_standard_deviation": float(np.std(null_values)),
        "upper_tail_p_value": float((1 + np.sum(null_values >= observed)) / (1 + len(null_values))),
    }


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == args.model_revision)
    ov, _ = load_factor_bundle(Path(record["factors"]["OV"]["path"]))
    qk, _ = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    static, _, _ = edge_tensor(Path(record["factors"]["OV"]["path"]), Path(record["factors"]["QK"]["path"]))
    with np.load(args.tokens, allow_pickle=False) as source:
        tokens = np.asarray(source["confirmation_tokens"], dtype=np.int64)
    midpoint = len(tokens) // 2
    model = AutoModelForCausalLM.from_pretrained(
        Path(record["snapshot"]), local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    print("collecting discovery action", flush=True)
    discovery = action_tensor(model, tokens[:midpoint], ov, qk, batch_size=args.batch_size, torch=torch)
    print("collecting held-out action", flush=True)
    confirmation = action_tensor(model, tokens[midpoint:], ov, qk, batch_size=args.batch_size, torch=torch)
    report = {
        "status": "held-out activation-weighted validation of static typed composition edges",
        "action": "RMS actual attention-weighted source OV output after linear application of target Q/K/V reader",
        "static_edge": "scale-free OV-writer to Q/K/V-reader overlap",
        "selection": "all cross-layer head pairs; no activation-based edge selection",
        "types": {
            kind: {
                "discovery_prediction": within_pair_prediction(
                    static[kind], discovery[kind], resamples=args.resamples, rng=np.random.default_rng(args.seed + index)
                ),
                "confirmation_prediction": within_pair_prediction(
                    static[kind], confirmation[kind], resamples=args.resamples, rng=np.random.default_rng(args.seed + 100 + index)
                ),
                "split_reliability": within_pair_prediction(
                    discovery[kind], confirmation[kind], resamples=args.resamples, rng=np.random.default_rng(args.seed + 200 + index)
                ),
            }
            for index, kind in enumerate(("Q", "K", "V"))
        },
        "sequences_per_split": midpoint,
        "seed": args.seed,
        "resamples": args.resamples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
