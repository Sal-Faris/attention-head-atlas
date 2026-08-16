"""Optional TransformerLens adapter and reproducible operator serialization."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .operators import HeadOperator, build_ov, build_qk


def extract_from_transformer_lens(model: Any, kind: str) -> list[HeadOperator]:
    """Extract operators from an already loaded TransformerLens model.

    Accepting the model as an argument keeps extraction testable without a
    network connection and avoids hiding model-download side effects.
    """

    kind = kind.upper()
    if kind not in {"OV", "QK"}:
        raise ValueError("kind must be 'OV' or 'QK'")

    operators: list[HeadOperator] = []
    for layer in range(int(model.cfg.n_layers)):
        for head in range(int(model.cfg.n_heads)):
            if kind == "OV":
                matrix = build_ov(
                    model.W_V[layer, head].detach().cpu().numpy(),
                    model.W_O[layer, head].detach().cpu().numpy(),
                )
            else:
                matrix = build_qk(
                    model.W_Q[layer, head].detach().cpu().numpy(),
                    model.W_K[layer, head].detach().cpu().numpy(),
                )
            operators.append(HeadOperator(layer, head, kind, matrix))
    return operators


def verify_extracted_actions(
    model: Any,
    operators: list[HeadOperator],
    *,
    seed: int = 1729,
    n_states: int = 8,
) -> dict[str, float]:
    """Measure extraction error against direct factored computation.

    The check covers every extracted head using deterministic synthetic residual
    states. It catches transpose and tensor-axis mistakes before artifacts are
    written to disk.
    """

    if not operators:
        raise ValueError("cannot verify an empty operator list")
    rng = np.random.default_rng(seed)
    d_model = operators[0].matrix.shape[0]
    states = rng.standard_normal((n_states, d_model)).astype(operators[0].matrix.dtype)
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for operator in operators:
        layer, head = operator.layer, operator.head
        if operator.kind == "OV":
            direct = (states @ model.W_V[layer, head].detach().cpu().numpy()) @ model.W_O[
                layer, head
            ].detach().cpu().numpy()
            recovered = states @ operator.matrix
        else:
            queries = states @ model.W_Q[layer, head].detach().cpu().numpy()
            keys = states @ model.W_K[layer, head].detach().cpu().numpy()
            direct = queries @ keys.T
            recovered = states @ operator.matrix @ states.T
        difference = direct - recovered
        absolute_error = float(np.max(np.abs(difference)))
        relative_error = float(np.linalg.norm(difference) / max(np.linalg.norm(direct), 1e-12))
        maximum_absolute_error = max(maximum_absolute_error, absolute_error)
        maximum_relative_error = max(maximum_relative_error, relative_error)
    return {
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
    }


def save_operator_bundle(
    path: str | Path,
    operators: list[HeadOperator],
    metadata: dict[str, Any],
) -> None:
    if not operators:
        raise ValueError("cannot save an empty operator bundle")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrices = np.stack([operator.matrix for operator in operators])
    layers = np.asarray([operator.layer for operator in operators], dtype=np.int64)
    heads = np.asarray([operator.head for operator in operators], dtype=np.int64)
    kinds = np.asarray([operator.kind for operator in operators])
    payload = dict(metadata)
    payload["python"] = platform.python_version()
    payload["numpy"] = np.__version__
    np.savez_compressed(
        path,
        matrices=matrices,
        layers=layers,
        heads=heads,
        kinds=kinds,
        metadata=np.asarray(json.dumps(payload, sort_keys=True)),
    )


def load_operator_bundle(path: str | Path) -> tuple[list[HeadOperator], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as bundle:
        matrices = bundle["matrices"]
        layers = bundle["layers"]
        heads = bundle["heads"]
        kinds = bundle["kinds"]
        metadata = json.loads(str(bundle["metadata"]))
    operators = [
        HeadOperator(int(layer), int(head), str(kind), matrix)
        for layer, head, kind, matrix in zip(layers, heads, kinds, matrices, strict=True)
    ]
    return operators, metadata
