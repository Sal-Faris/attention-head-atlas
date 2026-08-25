"""Collect exactly reconstructed QK inputs for conditional-subspace discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--pilot-artifact",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/activation_validation_pilot.npz"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("D:/Laptop/AI/model-cache/huggingface")
    )
    parser.add_argument("--model-revision", default="step143000")
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument(
        "--dataset-revision", default="127bfedcd5047750df5ccf3a12979a47bfa0bafa"
    )
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--confirmation-sequences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--confirmation-seed", type=int, default=1729)
    parser.add_argument(
        "--reuse-confirmation-from",
        type=Path,
        help="reuse frozen confirmation tokens and dataset rows from an existing QK artifact",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/qk_conditional_events_v1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/qk_conditional_events_v1.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_confirmation_tokens(
    dataset_snapshot: Path,
    tokenizer: object,
    *,
    used_rows: set[int],
    sequence_length: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select new token sequences without overlapping the checkpoint-0007 rows."""

    parquet_files = sorted((dataset_snapshot / "data").glob("*.parquet"))
    if len(parquet_files) != 1:
        raise ValueError("expected exactly one cached dataset parquet file")
    texts = pd.read_parquet(parquet_files[0], columns=["text"])["text"]
    prefix_token = getattr(tokenizer, "bos_token_id", None)
    if prefix_token is None:
        prefix_token = getattr(tokenizer, "eos_token_id", None)
    if prefix_token is None:
        raise ValueError("tokenizer has neither a BOS nor EOS token")

    selected_tokens: list[list[int]] = []
    selected_rows: list[int] = []
    rng = np.random.default_rng(seed)
    for row_index in rng.permutation(len(texts)):
        resolved_row = int(row_index)
        if resolved_row in used_rows:
            continue
        text = str(texts.iloc[resolved_row])[:8192]
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=sequence_length - 1,
        )["input_ids"]
        if len(token_ids) != sequence_length - 1:
            continue
        selected_tokens.append([int(prefix_token), *map(int, token_ids)])
        selected_rows.append(resolved_row)
        if len(selected_tokens) == count:
            break
    if len(selected_tokens) != count:
        raise RuntimeError("cached dataset does not contain enough new usable documents")
    return np.asarray(selected_tokens, dtype=np.int64), np.asarray(selected_rows, dtype=np.int64)


def _register_normalized_residual_hooks(model: Any) -> tuple[list[Any | None], list[Any]]:
    """Capture each layer's actual input-layernorm result for one forward pass."""

    captured: list[Any | None] = [None] * int(model.config.num_hidden_layers)
    handles = []
    for layer, block in enumerate(model.gpt_neox.layers):

        def save_output(_module: Any, _inputs: Any, output: Any, *, index: int = layer) -> None:
            captured[index] = output.detach()

        handles.append(block.input_layernorm.register_forward_hook(save_output))
    return captured, handles


def _head_qk_tensors(
    model: Any,
    normalized: Any,
    *,
    layer: int,
    position_ids: Any,
    apply_rotary_pos_emb: Any,
) -> tuple[Any, Any, Any, Any]:
    """Recreate a layer's pre- and post-RoPE query/key tensors exactly."""

    attention = model.gpt_neox.layers[layer].attention
    batch, sequence_length, _ = normalized.shape
    heads = int(model.config.num_attention_heads)
    width = int(attention.head_size)
    qkv = attention.query_key_value(normalized).view(batch, sequence_length, heads, 3 * width)
    qkv = qkv.transpose(1, 2)
    queries, keys, _ = qkv.chunk(3, dim=-1)
    cosine, sine = model.gpt_neox.rotary_emb(normalized, position_ids=position_ids)
    rotated_queries, rotated_keys = apply_rotary_pos_emb(queries, keys, cosine, sine)
    return queries, keys, rotated_queries, rotated_keys


def _causal_attention_from_qk(queries: Any, keys: Any, *, torch: Any) -> Any:
    logits = torch.matmul(queries, keys.transpose(-1, -2)) / np.sqrt(queries.shape[-1])
    sequence_length = logits.shape[-1]
    mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    return torch.softmax(masked_logits, dim=-1), logits


def collect_split(model: Any, tokens: np.ndarray, *, batch_size: int, torch: Any, apply_rotary_pos_emb: Any) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run a split and collect exact layer inputs plus QK reconstruction audits."""

    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    sequence_length = int(tokens.shape[1])
    normalized_batches: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
    query_pre_batches: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
    key_pre_batches: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
    query_rotated_batches: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
    key_rotated_batches: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
    maximum_float32_error = 0.0
    maximum_float16_error = 0.0

    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            input_ids = torch.as_tensor(tokens[start : start + batch_size])
            position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
            captured, handles = _register_normalized_residual_hooks(model)
            try:
                outputs = model(
                    input_ids,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                for handle in handles:
                    handle.remove()
            if outputs.attentions is None or len(outputs.attentions) != layer_count:
                raise RuntimeError("model did not return an attention tensor for every layer")

            for layer, normalized in enumerate(captured):
                if normalized is None:
                    raise RuntimeError(f"layer {layer} normalization hook did not fire")
                queries, keys, rotated_queries, rotated_keys = _head_qk_tensors(
                    model,
                    normalized,
                    layer=layer,
                    position_ids=position_ids,
                    apply_rotary_pos_emb=apply_rotary_pos_emb,
                )
                recovered_attention, _ = _causal_attention_from_qk(
                    rotated_queries, rotated_keys, torch=torch
                )
                expected_attention = outputs.attentions[layer]
                maximum_float32_error = max(
                    maximum_float32_error,
                    float((recovered_attention - expected_attention).abs().max().cpu()),
                )
                downcast_attention, _ = _causal_attention_from_qk(
                    rotated_queries.to(torch.float16).float(),
                    rotated_keys.to(torch.float16).float(),
                    torch=torch,
                )
                maximum_float16_error = max(
                    maximum_float16_error,
                    float((downcast_attention - expected_attention).abs().max().cpu()),
                )
                normalized_batches[layer].append(normalized.float().cpu().numpy())
                query_pre_batches[layer].append(queries.float().cpu().numpy())
                key_pre_batches[layer].append(keys.float().cpu().numpy())
                query_rotated_batches[layer].append(rotated_queries.float().cpu().numpy())
                key_rotated_batches[layer].append(rotated_keys.float().cpu().numpy())
            print(f"processed {min(start + batch_size, len(tokens))}/{len(tokens)} sequences", flush=True)

    def stack_by_layer(batches: list[list[np.ndarray]]) -> np.ndarray:
        arrays = [np.concatenate(layer_batches, axis=0) for layer_batches in batches]
        return np.stack(arrays, axis=1)

    result = {
        "normalized_residual": stack_by_layer(normalized_batches),
        "query_pre_rope": stack_by_layer(query_pre_batches),
        "key_pre_rope": stack_by_layer(key_pre_batches),
        "query_post_rope": stack_by_layer(query_rotated_batches),
        "key_post_rope": stack_by_layer(key_rotated_batches),
    }
    expected_qk_shape = (len(tokens), layer_count, head_count, sequence_length, 64)
    if result["query_pre_rope"].shape != expected_qk_shape:
        raise RuntimeError("collected query shape differs from the expected Pythia-70M geometry")
    return result, {
        "maximum_float32_attention_absolute_error": maximum_float32_error,
        "maximum_float16_attention_absolute_error": maximum_float16_error,
    }


def main() -> None:
    args = parse_args()
    if args.sequence_length != 64 or args.confirmation_sequences < 2 or args.batch_size < 1:
        raise ValueError("this frozen protocol requires length 64 and valid batch dimensions")
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
    except ImportError as error:
        raise SystemExit('Install model dependencies with: pip install -e ".[models]"') from error

    with np.load(args.pilot_artifact, allow_pickle=False) as pilot:
        train_tokens = np.asarray(pilot["train_tokens"], dtype=np.int64)
        tuning_tokens = np.asarray(pilot["test_tokens"], dtype=np.int64)
        train_rows = np.asarray(pilot["train_dataset_rows"], dtype=np.int64)
        tuning_rows = np.asarray(pilot["test_dataset_rows"], dtype=np.int64)
    if train_tokens.shape != tuning_tokens.shape or train_tokens.shape[1] != args.sequence_length:
        raise ValueError("pilot token split is incompatible with the frozen protocol")
    if set(train_rows).intersection(tuning_rows):
        raise ValueError("pilot discovery and tuning rows are not disjoint")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    matching_records = [
        record for record in manifest["records"] if record["revision"] == args.model_revision
    ]
    if len(matching_records) != 1:
        raise ValueError("model revision is absent or duplicated in the manifest")
    record = matching_records[0]
    model_snapshot = Path(record["snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot, local_files_only=True)
    if args.reuse_confirmation_from is None:
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
        confirmation_tokens, confirmation_rows = load_confirmation_tokens(
            dataset_snapshot,
            tokenizer,
            used_rows=set(train_rows).union(tuning_rows),
            sequence_length=args.sequence_length,
            count=args.confirmation_sequences,
            seed=args.confirmation_seed,
        )
    else:
        with np.load(args.reuse_confirmation_from, allow_pickle=False) as source:
            confirmation_tokens = np.asarray(source["confirmation_tokens"], dtype=np.int64)
            confirmation_rows = np.asarray(source["confirmation_dataset_rows"], dtype=np.int64)
        if confirmation_tokens.shape != (args.confirmation_sequences, args.sequence_length):
            raise ValueError("reused confirmation tokens have an incompatible shape")
        if confirmation_rows.shape != (args.confirmation_sequences,):
            raise ValueError("reused confirmation rows have an incompatible shape")
    if set(confirmation_rows).intersection(set(train_rows).union(tuning_rows)):
        raise RuntimeError("confirmation rows overlap pilot rows")

    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    split_tokens = {
        "discovery": train_tokens,
        "tuning": tuning_tokens,
        "confirmation": confirmation_tokens,
    }
    collected: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, float]] = {}
    for name, split in split_tokens.items():
        print(f"collecting {name} QK tensors", flush=True)
        collected[name], audits[name] = collect_split(
            model,
            split,
            batch_size=args.batch_size,
            torch=torch,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
        )

    float32_error = max(
        audit["maximum_float32_attention_absolute_error"] for audit in audits.values()
    )
    float16_error = max(
        audit["maximum_float16_attention_absolute_error"] for audit in audits.values()
    )
    if float32_error > 1e-5:
        raise RuntimeError(f"float32 QK reconstruction audit failed: {float32_error:.3e}")
    storage_dtype = np.float16 if float16_error <= 5e-4 else np.float32
    payload: dict[str, np.ndarray] = {
        "discovery_tokens": train_tokens,
        "tuning_tokens": tuning_tokens,
        "confirmation_tokens": confirmation_tokens,
        "discovery_dataset_rows": train_rows,
        "tuning_dataset_rows": tuning_rows,
        "confirmation_dataset_rows": confirmation_rows,
        "positions": np.arange(args.sequence_length, dtype=np.int64),
    }
    for split, values in collected.items():
        for name, value in values.items():
            payload[f"{split}_{name}"] = value.astype(storage_dtype)
    metadata = {
        "model": manifest["model"],
        "model_revision": args.model_revision,
        "model_snapshot_commit": record["snapshot_commit"],
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "sequence_length": args.sequence_length,
        "confirmation_seed": args.confirmation_seed,
        "storage_dtype": np.dtype(storage_dtype).name,
        "attention_reconstruction_audits": audits,
        "float16_accepted": bool(storage_dtype == np.float16),
        "attention_matrices_stored": False,
    }
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.artifact, **payload)
    report = {
        **metadata,
        "artifact": str(args.artifact),
        "artifact_sha256": sha256(args.artifact),
        "split_sizes": {name: len(tokens) for name, tokens in split_tokens.items()},
        "dataset_rows": {
            "discovery": train_rows.tolist(),
            "tuning": tuning_rows.tolist(),
            "confirmation": confirmation_rows.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved QK event tensors to {args.artifact}")


if __name__ == "__main__":
    main()
