"""Compact exact representations of low-rank attention-head operators."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class FactorizedHeadOperator:
    """An attention operator ``M = left @ right.T`` with model metadata."""

    layer: int
    head: int
    kind: str
    left: Array
    right: Array

    def __post_init__(self) -> None:
        left = np.asarray(self.left)
        right = np.asarray(self.right)
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            raise ValueError("left and right factors must have the same two-dimensional shape")
        if left.shape[0] < 1 or left.shape[1] < 1:
            raise ValueError("operator factors cannot be empty")
        if self.kind not in {"OV", "QK"}:
            raise ValueError("kind must be 'OV' or 'QK'")
        if self.layer < 0 or self.head < 0:
            raise ValueError("layer and head indices must be nonnegative")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("operator factors contain non-finite values")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @property
    def d_model(self) -> int:
        return int(self.left.shape[0])

    @property
    def d_head(self) -> int:
        return int(self.left.shape[1])

    def materialize(self, dtype: np.dtype | None = None) -> Array:
        """Construct the full residual-stream matrix on demand."""

        matrix = self.left @ self.right.T
        return matrix if dtype is None else matrix.astype(dtype, copy=False)


def factorized_inner_product(
    first: FactorizedHeadOperator,
    second: FactorizedHeadOperator,
) -> float:
    """Return the exact Frobenius inner product without materializing matrices."""

    if first.kind != second.kind:
        raise ValueError("operators must have the same kind")
    if first.left.shape != second.left.shape:
        raise ValueError("operators must have matching factor shapes")
    left_cross = np.asarray(first.left, dtype=np.float64).T @ np.asarray(
        second.left, dtype=np.float64
    )
    right_cross = np.asarray(first.right, dtype=np.float64).T @ np.asarray(
        second.right, dtype=np.float64
    )
    return float(np.sum(left_cross * right_cross))


def factorized_frobenius_norm(operator: FactorizedHeadOperator) -> float:
    """Return ``||left @ right.T||_F`` using only skinny factors."""

    squared_norm = factorized_inner_product(operator, operator)
    return float(np.sqrt(max(squared_norm, 0.0)))


def normalized_factorized_frobenius_distances(
    operators: Sequence[FactorizedHeadOperator],
    eps: float = 1e-12,
) -> Array:
    """Return exact scale-normalized Frobenius distances for factorized heads."""

    if not operators:
        raise ValueError("at least one operator is required")
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    expected_kind = operators[0].kind
    expected_shape = operators[0].left.shape
    if any(operator.kind != expected_kind for operator in operators):
        raise ValueError("operators must all have the same kind")
    if any(operator.left.shape != expected_shape for operator in operators):
        raise ValueError("operators must all have the same factor shape")

    norms = np.asarray([factorized_frobenius_norm(operator) for operator in operators])
    if np.any(norms <= eps):
        raise ValueError("cannot compare a near-zero operator")

    item_count = len(operators)
    distances = np.zeros((item_count, item_count), dtype=np.float64)
    for first_index in range(item_count):
        for second_index in range(first_index + 1, item_count):
            inner_product = factorized_inner_product(
                operators[first_index], operators[second_index]
            )
            similarity = inner_product / (norms[first_index] * norms[second_index])
            similarity = float(np.clip(similarity, -1.0, 1.0))
            distance = float(np.sqrt(max(2.0 - 2.0 * similarity, 0.0)))
            distances[first_index, second_index] = distance
            distances[second_index, first_index] = distance
    return distances


def blockwise_factorized_frobenius_distances(
    operators: Sequence[FactorizedHeadOperator],
    *,
    block_size: int = 8,
    scratch_directory: str | Path | None = None,
    eps: float = 1e-12,
) -> Array:
    """Compute normalized distances with bounded RAM and transient full matrices.

    Operators are materialized once into a temporary float32 memory map, then
    multiplied in float64 blocks. The scratch file is deleted on return and is
    never a research artifact.
    """

    if not operators:
        raise ValueError("at least one operator is required")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    expected_kind = operators[0].kind
    expected_shape = operators[0].left.shape
    if any(operator.kind != expected_kind for operator in operators):
        raise ValueError("operators must all have the same kind")
    if any(operator.left.shape != expected_shape for operator in operators):
        raise ValueError("operators must all have the same factor shape")

    item_count = len(operators)
    matrix_width = operators[0].d_model**2
    temporary_parent = None if scratch_directory is None else str(scratch_directory)
    with TemporaryDirectory(dir=temporary_parent) as temporary_directory:
        scratch_path = Path(temporary_directory) / "normalized_matrices.float32"
        flattened = np.memmap(
            scratch_path,
            mode="w+",
            dtype=np.float32,
            shape=(item_count, matrix_width),
        )
        for index, operator in enumerate(operators):
            flattened[index] = operator.materialize(dtype=np.float32).reshape(-1)
        flattened.flush()

        norms = np.empty(item_count, dtype=np.float64)
        for start in range(0, item_count, block_size):
            stop = min(start + block_size, item_count)
            block = np.asarray(flattened[start:stop], dtype=np.float64)
            norms[start:stop] = np.linalg.norm(block, axis=1)
        if np.any(norms <= eps):
            raise ValueError("cannot compare a near-zero operator")

        distances = np.zeros((item_count, item_count), dtype=np.float64)
        for first_start in range(0, item_count, block_size):
            first_stop = min(first_start + block_size, item_count)
            first = np.asarray(flattened[first_start:first_stop], dtype=np.float64)
            first /= norms[first_start:first_stop, None]
            for second_start in range(first_start, item_count, block_size):
                second_stop = min(second_start + block_size, item_count)
                second = np.asarray(flattened[second_start:second_stop], dtype=np.float64)
                second /= norms[second_start:second_stop, None]
                similarities = np.clip(first @ second.T, -1.0, 1.0)
                block_distances = np.sqrt(
                    np.maximum(2.0 - 2.0 * similarities, 0.0)
                )
                distances[first_start:first_stop, second_start:second_stop] = (
                    block_distances
                )
                distances[second_start:second_stop, first_start:first_stop] = (
                    block_distances.T
                )
        np.fill_diagonal(distances, 0.0)
        del flattened
    return distances


def factorized_action(operator: FactorizedHeadOperator, states: Array) -> Array:
    """Apply an OV-style row-vector operator without constructing its matrix."""

    states_array = np.asarray(states)
    if states_array.shape[-1] != operator.d_model:
        raise ValueError("state width does not match the operator")
    return (states_array @ operator.left) @ operator.right.T


def factorized_qk_scores(
    operator: FactorizedHeadOperator,
    queries: Array,
    keys: Array,
) -> Array:
    """Apply a factorized QK bilinear form to query and key state collections."""

    if operator.kind != "QK":
        raise ValueError("QK scores require a QK operator")
    query_array = np.asarray(queries)
    key_array = np.asarray(keys)
    if query_array.shape[-1] != operator.d_model or key_array.shape[-1] != operator.d_model:
        raise ValueError("query and key widths must match the operator")
    return (query_array @ operator.left) @ (key_array @ operator.right).T
