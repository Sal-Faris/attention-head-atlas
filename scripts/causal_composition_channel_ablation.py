"""Ablate low-rank source-output channels selected by typed composition edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_typed_composition_edges import edge_tensor

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
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--edges-per-type", type=int, default=2)
    parser.add_argument("--random-controls", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/composition_channel_causal_v1.json"),
    )
    return parser.parse_args()


def selected_edges(static: dict[str, np.ndarray], count: int) -> list[dict[str, int | float | str]]:
    records = []
    for kind, values in static.items():
        candidates = []
        for layer in range(values.shape[0] - 1):
            for source_head in range(values.shape[2]):
                for target_head in range(values.shape[3]):
                    candidates.append(
                        {
                            "kind": kind,
                            "source_layer": layer,
                            "source_head": source_head,
                            "target_layer": layer + 1,
                            "target_head": target_head,
                            "static_overlap": float(values[layer, layer + 1, source_head, target_head]),
                        }
                    )
        records.extend(sorted(candidates, key=lambda item: -float(item["static_overlap"]))[:count])
    return records


def source_projection(writer: np.ndarray, reader: np.ndarray, rank: int) -> np.ndarray:
    coupling = writer.T @ reader
    left, _, _ = np.linalg.svd(coupling, full_matrices=False)
    basis = left[:, :rank]
    return basis @ basis.T


def reader_index(kind: str) -> int:
    return {"Q": 0, "K": 1, "V": 2}[kind]


def ablation_outcomes(
    model: object,
    tokens: np.ndarray,
    source_layer: int,
    source_head: int,
    projections: list[np.ndarray],
    *,
    torch: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return next-layer QKV changes and removed-energy fractions per projection."""

    heads = int(model.config.num_attention_heads)
    width = int(model.config.hidden_size) // heads
    conditions = len(projections) + 1
    input_ids = torch.as_tensor(np.tile(tokens, (conditions, 1)))
    captured: list[object | None] = [None]
    energies = np.zeros(len(projections), dtype=np.float64)

    def ablate(_module: object, inputs: tuple[object, ...]) -> tuple[object, ...]:
        values = inputs[0].clone()
        start, end = source_head * width, (source_head + 1) * width
        original = values[: len(tokens), :, start:end]
        total_energy = float(original.square().mean().cpu())
        for index, projection in enumerate(projections):
            selected = slice((index + 1) * len(tokens), (index + 2) * len(tokens))
            component = values[selected, :, start:end] @ torch.as_tensor(
                projection, dtype=values.dtype, device=values.device
            )
            energies[index] = float(component.square().mean().cpu()) / max(total_energy, 1e-12)
            values[selected, :, start:end] -= component
        return (values, *inputs[1:])

    def capture(_module: object, _inputs: tuple[object, ...], output: object) -> None:
        captured[0] = output.detach().float()

    dense = model.gpt_neox.layers[source_layer].attention.dense
    qkv = model.gpt_neox.layers[source_layer + 1].attention.query_key_value
    first = dense.register_forward_pre_hook(ablate)
    second = qkv.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(input_ids, use_cache=False, return_dict=True)
    finally:
        first.remove()
        second.remove()
    if captured[0] is None:
        raise RuntimeError("next-layer QKV hook did not fire")
    values = captured[0].view(conditions, len(tokens), tokens.shape[1], heads, 3, width)
    change = values[1:] - values[:1]
    return np.sqrt(change.square().mean(dim=(1, 2, 5)).cpu().numpy()), energies


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["revision"] == args.model_revision)
    ov, _ = load_factor_bundle(Path(record["factors"]["OV"]["path"]))
    qk, _ = load_factor_bundle(Path(record["factors"]["QK"]["path"]))
    static, _, _ = edge_tensor(Path(record["factors"]["OV"]["path"]), Path(record["factors"]["QK"]["path"]))
    location = {(item.layer, item.head): index for index, item in enumerate(ov)}
    with np.load(args.tokens, allow_pickle=False) as source:
        tokens = np.asarray(source["confirmation_tokens"][: args.sequences], dtype=np.int64)
    model = AutoModelForCausalLM.from_pretrained(
        Path(record["snapshot"]), local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    rng = np.random.default_rng(args.seed)
    reports = []
    for edge in selected_edges(static, args.edges_per_type):
        source = (int(edge["source_layer"]), int(edge["source_head"]))
        target = (int(edge["target_layer"]), int(edge["target_head"]))
        writer = ov[location[source]].right
        kind = str(edge["kind"])
        target_operator = qk[location[target]] if kind in {"Q", "K"} else ov[location[target]]
        reader = target_operator.left if kind in {"Q", "V"} else target_operator.right
        targeted = source_projection(writer, reader, args.rank)
        random_projections = []
        for _ in range(args.random_controls):
            frame, _ = np.linalg.qr(rng.standard_normal((targeted.shape[0], args.rank)))
            random_projections.append(frame @ frame.T)
        change, energy = ablation_outcomes(
            model, tokens, source[0], source[1], [targeted, *random_projections], torch=torch
        )
        index = reader_index(kind)
        targeted_effect = float(change[0, target[1], index])
        control_effects = change[1:, target[1], index]
        normalized = change[:, target[1], index] / np.sqrt(np.maximum(energy, 1e-12))
        reports.append(
            {
                **edge,
                "targeted_effect_rms": targeted_effect,
                "targeted_removed_energy_fraction": float(energy[0]),
                "targeted_effect_per_removed_rms": float(normalized[0]),
                "random_control_mean_effect_rms": float(np.mean(control_effects)),
                "random_control_targeted_rank": int(1 + np.sum(control_effects >= targeted_effect)),
                "random_control_mean_effect_per_removed_rms": float(np.mean(normalized[1:])),
                "random_control_normalized_targeted_rank": int(1 + np.sum(normalized[1:] >= normalized[0])),
                "off_target_mean_effect_rms": float(np.mean(np.delete(change[0], target[1], axis=0))),
            }
        )
        print(f"ablated {kind} channel L{source[0]}H{source[1]} -> L{target[0]}H{target[1]}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "rank-limited composition-channel ablation versus random source-coordinate controls",
                "intervention": "remove the source head-coordinate projection selected by the OV-writer to target-reader coupling SVD",
                "selection": "top static adjacent edges per type; activation and intervention outcomes held out",
                "edges": reports,
                "rank": args.rank,
                "random_controls_per_edge": args.random_controls,
                "sequences": len(tokens),
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
