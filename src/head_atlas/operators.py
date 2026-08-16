"""Construction and validation of prompt-independent head operators."""

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class HeadOperator:
    """A matrix operator with its model location and kind."""

    layer: int
    head: int
    kind: str
    matrix: Array

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("head operator must be a square matrix")
        if self.kind not in {"OV", "QK"}:
            raise ValueError("kind must be 'OV' or 'QK'")
        if not np.isfinite(matrix).all():
            raise ValueError("head operator contains non-finite values")
        object.__setattr__(self, "matrix", matrix)


def build_ov(w_v: Array, w_o: Array) -> Array:
    """Return the row-vector OV operator ``W_V @ W_O``.

    Expected shapes are ``W_V[d_model, d_head]`` and
    ``W_O[d_head, d_model]``.
    """

    w_v = np.asarray(w_v)
    w_o = np.asarray(w_o)
    if w_v.ndim != 2 or w_o.ndim != 2 or w_v.shape[1] != w_o.shape[0]:
        raise ValueError(f"incompatible OV shapes: {w_v.shape} and {w_o.shape}")
    return w_v @ w_o


def build_qk(w_q: Array, w_k: Array) -> Array:
    """Return the row-vector QK bilinear operator ``W_Q @ W_K.T``."""

    w_q = np.asarray(w_q)
    w_k = np.asarray(w_k)
    if w_q.ndim != 2 or w_k.ndim != 2 or w_q.shape != w_k.shape:
        raise ValueError(f"incompatible QK shapes: {w_q.shape} and {w_k.shape}")
    return w_q @ w_k.T
