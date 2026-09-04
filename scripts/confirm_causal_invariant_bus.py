"""Efficient causal validation of ambient writer-to-Q channels on Pythia."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.factor_io import load_factor_bundle
from head_atlas.relational_invariants import orthonormal_span


def ambient_channel(writer: np.ndarray, readers: tuple[np.ndarray, ...], rank: int = 4) -> np.ndarray:
    covariances = []
    for reader in readers:
        cross = writer.T @ reader
        cov = cross @ cross.T
        covariances.append(cov / max(float(np.trace(cov)), 1e-15))
    _, vectors = np.linalg.eigh(np.mean(covariances, axis=0))
    basis, _ = np.linalg.qr(writer @ vectors[:, ::-1][:, :rank], mode="reduced")
    return basis @ basis.T


def random_channel(writer: np.ndarray, rank: int, rng: np.random.Generator) -> np.ndarray:
    basis, _ = np.linalg.qr(writer @ rng.standard_normal((writer.shape[1], rank)), mode="reduced")
    return basis @ basis.T


def empirical_p(null_values: np.ndarray, observed: float) -> float:
    return float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1))


def aggregate_outcomes(records: list[dict], *, control_index: int = 0, excluded_layer: int | None = None) -> dict[str, float]:
    """Layer-equal JS and layer-equal ratio-of-sums over source records."""
    selected = [item for item in records if excluded_layer is None or item["source_layer"] != excluded_layer]
    by_layer: dict[int, list[dict]] = {}
    for item in selected:
        by_layer.setdefault(int(item["source_layer"]), []).append(item["controls"][control_index])
    layer_means = [float(np.mean([x["js_divergence"] for x in values])) for values in by_layer.values()]
    layer_ratios = [float(np.sum([x["js_divergence"] for x in values]) / max(np.sum([x["removed_energy_fraction"] for x in values]), 1e-15)) for values in by_layer.values()]
    return {"layer_equal_mean_js": float(np.mean(layer_means)), "layer_equal_ratio_of_sums_js_energy": float(np.mean(layer_ratios))}


def infer_population(real: dict[str, float], nulls: list[dict[str, float]]) -> dict:
    result = {}
    for metric, observed in real.items():
        values = np.asarray([item[metric] for item in nulls])
        result[metric] = {"observed": observed, "null_mean": float(np.mean(values)), "empirical_upper_tail_p_value": empirical_p(values, observed), "all_positive_excess": bool(observed > np.mean(values))}
    result["IUT_p_value"] = max(item["empirical_upper_tail_p_value"] for item in result.values())
    result["all_metrics_positive_excess"] = all(item["all_positive_excess"] for key, item in result.items() if key != "IUT_p_value")
    return result


def summarize_split(
    records: list[dict], null_repetitions: int, random_controls: int
) -> dict:
    """Infer against aligned coordinate-null draws and retain Haar diagnostics."""

    real = aggregate_outcomes(records, control_index=0)
    nulls = [
        aggregate_outcomes(records, control_index=1 + index)
        for index in range(null_repetitions)
    ]
    haar_start = 1 + null_repetitions
    haar = [
        aggregate_outcomes(records, control_index=haar_start + index)
        for index in range(random_controls)
    ]
    layers = sorted({int(item["source_layer"]) for item in records})
    leave_one_layer_out = {}
    for layer in layers:
        leave_real = aggregate_outcomes(records, control_index=0, excluded_layer=layer)
        leave_nulls = [
            aggregate_outcomes(
                records, control_index=1 + index, excluded_layer=layer
            )
            for index in range(null_repetitions)
        ]
        leave_one_layer_out[str(layer)] = infer_population(leave_real, leave_nulls)
    return {
        "real": real,
        "coordinate_null_mean": {
            metric: float(np.mean([item[metric] for item in nulls])) for metric in real
        },
        "haar_mean": {
            metric: float(np.mean([item[metric] for item in haar]))
            if haar
            else float("nan")
            for metric in real
        },
        "inference": infer_population(real, nulls),
        "leave_one_source_layer_out": leave_one_layer_out,
    }


def js_divergence(first: object, second: object, torch: object) -> object:
    midpoint = 0.5 * (first + second)
    first_term = (
        first
        * (torch.log(first.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12)))
    ).sum(-1)
    second_term = (
        second
        * (torch.log(second.clamp_min(1e-12)) - torch.log(midpoint.clamp_min(1e-12)))
    ).sum(-1)
    return 0.5 * (first_term + second_term)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pythia-70m-deduped-pilot.json"))
    parser.add_argument("--tokens", type=Path, default=Path("artifacts/pythia-70m-deduped/qk_bilinear_margin_confirmation_v1.npz"))
    parser.add_argument("--model-revision", default="step143000")
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--null-repetitions", type=int, default=199)
    parser.add_argument("--channel-batch-size", type=int, default=1)
    parser.add_argument("--splits", action="store_true", help="run both 4/4 target-head splits")
    parser.add_argument("--max-sources", type=int, default=40)
    parser.add_argument("--seed", type=int, default=9331)
    parser.add_argument("--output", type=Path, default=Path("results/pythia-70m-deduped/causal_invariant_bus_confirmed.json"))
    parser.add_argument("--figure", type=Path, default=Path("results/pythia-70m-deduped/causal_invariant_bus_confirmed.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == args.model_revision)
    ov, _ = load_factor_bundle(Path(record["factors"]["OV"]["path"]))
    qk, _ = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    with np.load(args.tokens, allow_pickle=False) as source:
        tokens = np.asarray(source["confirmation_tokens"][: args.sequences], dtype=np.int64)
    model = AutoModelForCausalLM.from_pretrained(Path(record["snapshot"]), local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    layers, heads = int(model.config.num_hidden_layers), int(model.config.num_attention_heads)
    width = int(model.config.hidden_size) // heads
    writers = {(x.layer, x.head): orthonormal_span(x.right.astype(np.float64)) for x in ov}
    readers = {(x.layer, x.head): orthonormal_span(x.left.astype(np.float64)) for x in qk}
    hidden_inputs = {}
    dense_inputs = {}
    handles = []
    for layer in range(layers):
        handles.append(model.gpt_neox.layers[layer].register_forward_pre_hook(lambda module, inputs, layer=layer: hidden_inputs.setdefault(layer, inputs[0].detach())))
        handles.append(model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(lambda module, inputs, layer=layer: dense_inputs.setdefault(layer, inputs[0].detach())))
    with torch.inference_mode():
        clean_full = model(torch.as_tensor(tokens), output_attentions=True, return_dict=True)
    for handle in handles:
        handle.remove()
    seq_len = tokens.shape[1]
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)[None, None]
    position_ids = torch.arange(seq_len, device=clean_full.logits.device).unsqueeze(0).expand(args.sequences, -1)
    if not model.config.use_parallel_residual:
        raise ValueError("efficient intervention requires a parallel-residual model")
    if args.max_sources < 1 or args.null_repetitions < 1 or args.random_controls < 0:
        raise ValueError("invalid source or control count")
    if args.channel_batch_size < 1:
        raise ValueError("channel batch size must be positive")

    permutation_rng = np.random.default_rng(args.seed)
    ambient = int(model.config.hidden_size)
    null_permutations = {
        target_layer: [
            permutation_rng.permutation(ambient) for _ in range(args.null_repetitions)
        ]
        for target_layer in range(1, layers)
    }
    split_definitions = [("forward", np.arange(4), np.arange(4, 8))]
    if args.splits:
        split_definitions.append(("reverse", np.arange(4, 8), np.arange(4)))

    local_replay_max_abs = {}
    clean_cache = {}
    for target_layer in range(1, layers):
        clean_hidden = hidden_inputs[target_layer]
        target_attention = model.gpt_neox.layers[target_layer].attention
        with torch.inference_mode():
            clean_norm = model.gpt_neox.layers[target_layer].input_layernorm(clean_hidden)
            clean_qkv = target_attention.query_key_value(clean_norm)
            replay = target_attention(
                clean_norm,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=False,
                output_attentions=True,
            )[2]
        local_replay_max_abs[str(target_layer)] = float(
            torch.max(torch.abs(replay - clean_full.attentions[target_layer])).cpu()
        )
        clean_cache[target_layer] = (clean_hidden, clean_qkv)

    split_records = {}
    for split_index, (split_name, train, test) in enumerate(split_definitions):
        records = []
        count = 0
        haar_rng = np.random.default_rng(args.seed + 10000 + split_index)
        for source_layer in range(layers - 1):
            target_layer = source_layer + 1
            clean_hidden, clean_qkv = clean_cache[target_layer]
            clean_qkv_view = clean_qkv.view(args.sequences, seq_len, heads, 3, width)
            clean_attention = clean_full.attentions[target_layer][:, test]
            dense_weight = model.gpt_neox.layers[source_layer].attention.dense.weight.detach()
            for source_head in range(heads):
                if count >= args.max_sources:
                    break
                writer = writers[source_layer, source_head]
                real_readers = tuple(readers[target_layer, int(head)] for head in train)
                projectors = [ambient_channel(writer, real_readers, args.rank)]
                for permutation in null_permutations[target_layer]:
                    permuted_readers = tuple(
                        readers[target_layer, int(head)][permutation] for head in train
                    )
                    projectors.append(
                        ambient_channel(writer, permuted_readers, args.rank)
                    )
                projectors.extend(
                    random_channel(writer, args.rank, haar_rng)
                    for _ in range(args.random_controls)
                )
                start = source_head * width
                stop = (source_head + 1) * width
                source_values = dense_inputs[source_layer][..., start:stop]
                source_output = source_values @ dense_weight[:, start:stop].T
                source_energy = max(float(source_output.square().mean().cpu()), 1e-12)
                outcomes = []
                for batch_start in range(0, len(projectors), args.channel_batch_size):
                    batch_projectors = torch.as_tensor(
                        np.stack(
                            projectors[
                                batch_start : batch_start + args.channel_batch_size
                            ]
                        ),
                        dtype=source_output.dtype,
                    )
                    projected = torch.einsum(
                        "bsm,cmn->cbsn", source_output, batch_projectors
                    )
                    condition_count = projected.shape[0]
                    modified_hidden = clean_hidden.unsqueeze(0) - projected
                    modified_norm = model.gpt_neox.layers[target_layer].input_layernorm(
                        modified_hidden.reshape(-1, seq_len, ambient)
                    )

                    def patch_qkv(
                        _module: object,
                        _inputs: tuple[object, ...],
                        output: object,
                        clean_qkv_view: object = clean_qkv_view,
                        condition_count: int = condition_count,
                    ) -> object:
                        patched = output.view(
                            condition_count,
                            args.sequences,
                            seq_len,
                            heads,
                            3,
                            width,
                        ).clone()
                        patched[..., 1:, :] = clean_qkv_view.unsqueeze(0)[..., 1:, :]
                        return patched.reshape(
                            condition_count * args.sequences, seq_len, -1
                        )

                    qkv_module = model.gpt_neox.layers[
                        target_layer
                    ].attention.query_key_value
                    handle = qkv_module.register_forward_hook(patch_qkv)
                    try:
                        with torch.inference_mode():
                            attention = model.gpt_neox.layers[target_layer].attention(
                                modified_norm,
                                attention_mask=causal_mask,
                                position_ids=position_ids.repeat(condition_count, 1),
                                use_cache=False,
                                output_attentions=True,
                            )[2]
                    finally:
                        handle.remove()
                    attention = attention.view(
                        condition_count, args.sequences, heads, seq_len, seq_len
                    )
                    js = js_divergence(
                        clean_attention[None, :, :, 1:],
                        attention[:, :, test, 1:],
                        torch,
                    )
                    js_means = js.mean(dim=(1, 2, 3)).cpu().numpy()
                    energies = (
                        projected.square().mean(dim=(1, 2, 3)).cpu().numpy()
                        / source_energy
                    )
                    outcomes.extend(
                        {
                            "js_divergence": float(js_value),
                            "removed_energy_fraction": float(energy),
                        }
                        for js_value, energy in zip(js_means, energies)
                    )
                records.append(
                    {
                        "source_layer": source_layer,
                        "source_head": source_head,
                        "target_layer": target_layer,
                        "held_out_target_heads": test.tolist(),
                        "controls": outcomes,
                    }
                )
                count += 1
                print(
                    f"{split_name} L{source_layer}H{source_head} "
                    f"({count}/{min(args.max_sources, (layers - 1) * heads)})",
                    flush=True,
                )
            if count >= args.max_sources:
                break
        split_records[split_name] = records

    summaries = {
        name: summarize_split(records, args.null_repetitions, args.random_controls)
        for name, records in split_records.items()
    }
    joint_iut = max(summary["inference"]["IUT_p_value"] for summary in summaries.values())
    joint_positive = all(
        summary["inference"]["all_metrics_positive_excess"]
        for summary in summaries.values()
    )
    report = {
        "status": "confirmatory causal ambient invariant writer-Q bus test",
        "model": manifest["model"],
        "model_revision": args.model_revision,
        "model_snapshot_commit": record["snapshot_commit"],
        "token_artifact": str(args.tokens),
        "sequences": args.sequences,
        "intervention": "subtract source-head residual contribution in a weight-discovered ambient channel; reevaluate only next-layer attention with clean K/V",
        "channel_order": "real, shared-coordinate null draws, Haar writer-span controls",
        "rank": args.rank,
        "random_controls": args.random_controls,
        "null_repetitions": args.null_repetitions,
        "source_count_per_split": len(next(iter(split_records.values()))),
        "splits": {name: {"summary": summaries[name], "records": records} for name, records in split_records.items()},
        "joint_split_IUT_p_value": joint_iut,
        "joint_split_all_metrics_positive": joint_positive,
        "local_replay_max_absolute_error": local_replay_max_abs,
        "aggregation": "equal mean over source layers; energy endpoint is per-layer ratio of summed JS to summed removed-energy fraction",
        "null": "one residual-coordinate permutation per target layer and draw, shared across discovery Q heads and all source writers",
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics = ("layer_equal_mean_js", "layer_equal_ratio_of_sums_js_energy")
    figure, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 4), squeeze=False)
    for axis, (name, summary) in zip(axes[0], summaries.items()):
        positions = np.arange(len(metrics))
        width_bar = 0.25
        axis.bar(positions - width_bar, [summary["real"][x] for x in metrics], width_bar, label="real bus")
        axis.bar(positions, [summary["coordinate_null_mean"][x] for x in metrics], width_bar, label="coordinate null")
        axis.bar(positions + width_bar, [summary["haar_mean"][x] for x in metrics], width_bar, label="Haar")
        axis.set_xticks(positions, ["attention JS", "JS / removed energy"])
        axis.set_title(f"{name} held-out Q heads")
        axis.legend()
    figure.suptitle("Causal effect of weight-discovered writer→Q bus channels")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
