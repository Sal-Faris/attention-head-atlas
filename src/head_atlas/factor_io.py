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
