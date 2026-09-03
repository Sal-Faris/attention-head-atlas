"""Causally test adjacent OV-writer to Q/K/V-reader edges by head ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_typed_composition_edges import edge_tensor
from scipy.stats import spearmanr


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
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/adjacent_composition_causal_v1.json"),
    )
    return parser.parse_args()


def target_qkv_change(
    model: object, tokens: np.ndarray, source_layer: int, *, torch: object
) -> np.ndarray:
    """Ablate each source head once and return next-layer Q/K/V RMS changes.

    Condition 0 is clean.  Conditions 1..H zero one head's post-attention,
    pre-output-projection vector, so the intervention propagates naturally
    through the source block and into the next layer.
    """

    heads = int(model.config.num_attention_heads)
    width = int(model.config.hidden_size) // heads
    target_layer = source_layer + 1
    conditions = heads + 1
    condition_ids = torch.arange(conditions).repeat_interleave(len(tokens))
    input_ids = torch.as_tensor(np.tile(tokens, (conditions, 1)))
    captured: list[object | None] = [None]

    def ablate_head(_module: object, inputs: tuple[object, ...]) -> tuple[object, ...]:
        values = inputs[0].clone()
        for head in range(heads):
            selected = condition_ids == head + 1
            values[selected, :, head * width : (head + 1) * width] = 0.0
        return (values, *inputs[1:])

    def capture_qkv(_module: object, _inputs: tuple[object, ...], output: object) -> None:
        captured[0] = output.detach().float()

    source_dense = model.gpt_neox.layers[source_layer].attention.dense
    target_qkv = model.gpt_neox.layers[target_layer].attention.query_key_value
    source_handle = source_dense.register_forward_pre_hook(ablate_head)
    target_handle = target_qkv.register_forward_hook(capture_qkv)
    try:
        with torch.inference_mode():
            model(input_ids, use_cache=False, return_dict=True)
    finally:
        source_handle.remove()
        target_handle.remove()
    if captured[0] is None:
        raise RuntimeError("target QKV hook did not fire")
    qkv = captured[0].view(conditions, len(tokens), tokens.shape[1], heads, 3, width)
    clean = qkv[0]
    changes = qkv[1:] - clean
    return np.sqrt(changes.square().mean(dim=(1, 2, 5)).cpu().numpy())


def prediction_test(static: np.ndarray, effect: np.ndarray, *, resamples: int, rng: np.random.Generator) -> dict[str, float]:
    layer_count, _, heads, _ = static.shape
    pairs = [(static[layer, layer + 1], effect[layer]) for layer in range(layer_count - 1)]
    observed = float(np.mean([spearmanr(first.ravel(), second.ravel()).statistic for first, second in pairs]))
    null = []
    for _ in range(resamples):
        null.append(
            float(
                np.mean(
                    [
                        spearmanr(
                            first[rng.permutation(heads)][:, rng.permutation(heads)].ravel(),
                            second.ravel(),
                        ).statistic
                        for first, second in pairs
                    ]
                )
            )
        )
    values = np.asarray(null)
    return {
        "mean_adjacent_layer_pair_spearman": observed,
        "head_identity_shuffle_mean": float(np.mean(values)),
        "head_identity_shuffle_standard_deviation": float(np.std(values)),
        "upper_tail_p_value": float((1 + np.sum(values >= observed)) / (1 + len(values))),
    }


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == args.model_revision)
    static, _, _ = edge_tensor(Path(record["factors"]["OV"]["path"]), Path(record["factors"]["QK"]["path"]))
    with np.load(args.tokens, allow_pickle=False) as source:
        tokens = np.asarray(source["confirmation_tokens"][: args.sequences], dtype=np.int64)
    model = AutoModelForCausalLM.from_pretrained(
        Path(record["snapshot"]), local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    effects = {kind: [] for kind in ("Q", "K", "V")}
    for source_layer in range(int(model.config.num_hidden_layers) - 1):
        change = target_qkv_change(model, tokens, source_layer, torch=torch)
        for index, kind in enumerate(("Q", "K", "V")):
            effects[kind].append(change[:, :, index])
        print(f"intervened on source layer {source_layer}", flush=True)
    effect_arrays = {kind: np.stack(values) for kind, values in effects.items()}
    report = {
        "status": "adjacent-layer causal head-ablation test of typed composition edges",
        "intervention": "zero one source head's attention output immediately before its output projection",
        "outcome": "RMS change in the next layer's raw Q/K/V preactivations",
        "selection": "all adjacent cross-layer head pairs; static edges selected without activations",
        "types": {
            kind: prediction_test(
                static[kind], effect_arrays[kind], resamples=args.resamples, rng=np.random.default_rng(args.seed + index)
            )
            for index, kind in enumerate(("Q", "K", "V"))
        },
        "sequences": len(tokens),
        "seed": args.seed,
        "resamples": args.resamples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
