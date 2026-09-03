"""TransformerLens extraction and serialization for compact operator factors."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .factors import (
    FactorizedHeadOperator,
    factorized_action,
    factorized_qk_scores,
)


def _tensor_as_float32(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _snapshot_tensor_files(snapshot: Path) -> dict[str, Path]:
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return {
            key: snapshot / filename for key, filename in index["weight_map"].items()
        }
    model_path = snapshot / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"no safetensors weights found in {snapshot}")
    from safetensors import safe_open

    with safe_open(model_path, framework="np") as weights:
        return dict.fromkeys(weights.keys(), model_path)


def _load_snapshot_tensor(tensor_files: dict[str, Path], key: str) -> np.ndarray:
    try:
        tensor_path = tensor_files[key]
    except KeyError as error:
        raise KeyError(f"model snapshot does not contain {key}") from error
    from safetensors import safe_open

    with safe_open(tensor_path, framework="np") as weights:
        return np.asarray(weights.get_tensor(key), dtype=np.float32)


def _fold_and_center_reader(weight: np.ndarray, layer_norm: np.ndarray) -> np.ndarray:
    folded = weight * layer_norm[:, None]
    return folded - folded.mean(axis=0, keepdims=True)


def _center_writer(weight: np.ndarray) -> np.ndarray:
    return weight - weight.mean(axis=-1, keepdims=True)


def extract_processed_factors_from_safetensors(
    snapshot: str | Path,
    kind: str,
) -> tuple[list[FactorizedHeadOperator], dict[str, Any]]:
    """Extract TransformerLens-equivalent processed factors without loading a model.

    GPT-2 and GPT-NeoX snapshots are read tensor-by-tensor. LayerNorm scales are
    folded into Q/K/V, residual-stream readers are centered, and O writers are
    centered, matching TransformerLens' default interpretability transforms.
    """

    snapshot_path = Path(snapshot)
    config = json.loads((snapshot_path / "config.json").read_text(encoding="utf-8"))
    normalized_kind = kind.upper()
    if normalized_kind not in {"OV", "QK"}:
        raise ValueError("kind must be 'OV' or 'QK'")
    model_type = str(config.get("model_type", ""))
    n_layers = int(config.get("n_layer", config.get("num_hidden_layers", 0)))
    n_heads = int(config.get("n_head", config.get("num_attention_heads", 0)))
    d_model = int(config.get("n_embd", config.get("hidden_size", 0)))
    if not n_layers or not n_heads or not d_model or d_model % n_heads:
        raise ValueError("snapshot has unsupported or inconsistent architecture dimensions")
    d_head = d_model // n_heads
    tensor_files = _snapshot_tensor_files(snapshot_path)
    operators: list[FactorizedHeadOperator] = []

    for layer in range(n_layers):
        if model_type == "gpt2":
            prefix = f"h.{layer}"
            qkv = _load_snapshot_tensor(tensor_files, f"{prefix}.attn.c_attn.weight")
            output = _load_snapshot_tensor(tensor_files, f"{prefix}.attn.c_proj.weight")
            layer_norm = _load_snapshot_tensor(tensor_files, f"{prefix}.ln_1.weight")
            qkv = qkv.reshape(d_model, 3, n_heads, d_head).transpose(1, 2, 0, 3)
            output = output.reshape(n_heads, d_head, d_model)
        elif model_type == "gpt_neox":
            prefix = f"gpt_neox.layers.{layer}"
            qkv = _load_snapshot_tensor(
                tensor_files, f"{prefix}.attention.query_key_value.weight"
            )
            output = _load_snapshot_tensor(tensor_files, f"{prefix}.attention.dense.weight")
            layer_norm = _load_snapshot_tensor(
                tensor_files, f"{prefix}.input_layernorm.weight"
            )
            qkv = qkv.reshape(n_heads, 3, d_head, d_model).transpose(1, 0, 3, 2)
            output = output.reshape(d_model, n_heads, d_head).transpose(1, 2, 0)
        else:
            raise ValueError(f"unsupported model_type: {model_type!r}")

        for head in range(n_heads):
            if normalized_kind == "QK":
                left = _fold_and_center_reader(qkv[0, head], layer_norm)
                right = _fold_and_center_reader(qkv[1, head], layer_norm)
            else:
                left = _fold_and_center_reader(qkv[2, head], layer_norm)
                right = _center_writer(output[head]).T.copy()
            operators.append(
                FactorizedHeadOperator(layer, head, normalized_kind, left, right)
            )

    metadata = {
        "source_format": "huggingface-safetensors",
        "model_type": model_type,
        "architectures": config.get("architectures"),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_model": d_model,
        "d_head": d_head,
        "rope_theta": config.get("rotary_emb_base"),
        "rotary_pct": config.get("rotary_pct"),
        "weight_processing": "fold_layer_norm_and_center_readers_and_writers",
    }
    return operators, metadata


def extract_factors_from_transformer_lens(
    model: Any,
    kind: str,
) -> list[FactorizedHeadOperator]:
    """Extract exact QK or OV skinny factors from an already loaded model."""

    normalized_kind = kind.upper()
    if normalized_kind not in {"OV", "QK"}:
        raise ValueError("kind must be 'OV' or 'QK'")
    operators = []
    for layer in range(int(model.cfg.n_layers)):
        for head in range(int(model.cfg.n_heads)):
            if normalized_kind == "QK":
                left = _tensor_as_float32(model.W_Q[layer, head])
                right = _tensor_as_float32(model.W_K[layer, head])
            else:
                left = _tensor_as_float32(model.W_V[layer, head])
                right = _tensor_as_float32(model.W_O[layer, head]).T.copy()
            operators.append(
                FactorizedHeadOperator(layer, head, normalized_kind, left, right)
            )
    return operators


def verify_factorized_actions(
    model: Any,
    operators: list[FactorizedHeadOperator],
    *,
    seed: int = 1729,
    n_states: int = 8,
) -> dict[str, float]:
    """Compare every factorized operator with direct model-weight computation."""

    if not operators:
        raise ValueError("cannot verify an empty operator list")
    rng = np.random.default_rng(seed)
    d_model = operators[0].d_model
    states = rng.standard_normal((n_states, d_model)).astype(np.float32)
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for operator in operators:
        layer, head = operator.layer, operator.head
        if operator.kind == "OV":
            direct = (states @ _tensor_as_float32(model.W_V[layer, head])) @ (
                _tensor_as_float32(model.W_O[layer, head])
            )
            recovered = factorized_action(operator, states)
        else:
            direct = (states @ _tensor_as_float32(model.W_Q[layer, head])) @ (
                states @ _tensor_as_float32(model.W_K[layer, head])
            ).T
            recovered = factorized_qk_scores(operator, states, states)
        difference = direct - recovered
        absolute_error = float(np.max(np.abs(difference)))
        relative_error = float(
            np.linalg.norm(difference) / max(np.linalg.norm(direct), 1e-12)
        )
        maximum_absolute_error = max(maximum_absolute_error, absolute_error)
        maximum_relative_error = max(maximum_relative_error, relative_error)
    return {
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


def save_factor_bundle(
    path: str | Path,
    operators: list[FactorizedHeadOperator],
    metadata: dict[str, Any],
) -> None:
    """Serialize a uniform population of compact factorized operators."""

    if not operators:
        raise ValueError("cannot save an empty factor bundle")
    expected_kind = operators[0].kind
    expected_shape = operators[0].left.shape
    if any(operator.kind != expected_kind for operator in operators):
        raise ValueError("factor bundle operators must have one kind")
    if any(operator.left.shape != expected_shape for operator in operators):
        raise ValueError("factor bundle operators must have one factor shape")
    payload_metadata = dict(metadata)
    payload_metadata.update(
        {
            "format": "factorized-head-operator-v1",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "operator_count": len(operators),
            "kind": expected_kind,
            "d_model": int(expected_shape[0]),
            "d_head": int(expected_shape[1]),
        }
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        left_factors=np.stack([operator.left for operator in operators]),
        right_factors=np.stack([operator.right for operator in operators]),
        layers=np.asarray([operator.layer for operator in operators], dtype=np.int64),
        heads=np.asarray([operator.head for operator in operators], dtype=np.int64),
        kinds=np.asarray([operator.kind for operator in operators]),
        metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True)),
    )


def load_factor_bundle(
    path: str | Path,
) -> tuple[list[FactorizedHeadOperator], dict[str, Any]]:
    """Load a factor bundle and validate every reconstructed record."""

    with np.load(Path(path), allow_pickle=False) as bundle:
        left_factors = bundle["left_factors"]
        right_factors = bundle["right_factors"]
        layers = bundle["layers"]
        heads = bundle["heads"]
        kinds = bundle["kinds"]
        metadata = json.loads(str(bundle["metadata_json"]))
    lengths = {
        len(left_factors), len(right_factors), len(layers), len(heads), len(kinds)
    }
    if len(lengths) != 1:
        raise ValueError("factor bundle arrays have inconsistent lengths")
    operators = [
        FactorizedHeadOperator(int(layer), int(head), str(kind), left, right)
        for layer, head, kind, left, right in zip(
            layers,
            heads,
            kinds,
            left_factors,
            right_factors,
            strict=True,
        )
    ]
    if metadata.get("format") != "factorized-head-operator-v1":
        raise ValueError("unsupported factor bundle format")
    return operators, metadata
