"""Low-rank shared operator motifs learned with CP alternating least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RankOneMotifs:
    """A shared dictionary of rank-one matrices ``left_k right_k.T``."""

    left: Array
    right: Array
    training_loss: float
    iterations: int

    def __post_init__(self) -> None:
        left = np.asarray(self.left, dtype=np.float64)
        right = np.asarray(self.right, dtype=np.float64)
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            raise ValueError("motif sides must be equally shaped matrices")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("motif sides must be finite")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)


def _coefficients(tensor: Array, left: Array, right: Array, ridge: float) -> Array:
    rhs = np.einsum("nij,ik,jk->nk", tensor, left, right, optimize=True)
    gram = (left.T @ left) * (right.T @ right)
    scale = max(float(np.trace(gram)) / max(len(gram), 1), 1.0)
    return np.linalg.solve(gram + ridge * scale * np.eye(len(gram)), rhs.T).T


def encode_rank_one_motifs(
    tensor: Array, motifs: RankOneMotifs, ridge: float = 1e-7
) -> Array:
    """Return least-squares coefficients for a shared rank-one dictionary."""

    return _coefficients(
        np.asarray(tensor, dtype=np.float64), motifs.left, motifs.right, ridge
    )


def reconstruct_rank_one_motifs(tensor: Array, motifs: RankOneMotifs, ridge: float = 1e-7) -> Array:
    """Least-squares encode and reconstruct matrices using shared rank-one motifs."""

    values = np.asarray(tensor, dtype=np.float64)
    coefficients = encode_rank_one_motifs(values, motifs, ridge)
    return np.einsum(
        "nk,ik,jk->nij", coefficients, motifs.left, motifs.right, optimize=True
    )


def fit_rank_one_motifs(
    tensor: Array,
    rank: int,
    *,
    seed: int = 0,
    iterations: int = 100,
    restarts: int = 3,
    ridge: float = 1e-7,
    tolerance: float = 1e-8,
) -> RankOneMotifs:
    """Fit ``X_h ~= sum_k a_hk left_k right_k.T`` by regularized CP-ALS."""

    values = np.asarray(tensor, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("tensor must contain square matrices")
    if rank < 1 or rank > values.shape[1]:
        raise ValueError("rank must be between one and the matrix dimension")
    if iterations < 1 or restarts < 1:
        raise ValueError("iterations and restarts must be positive")

    generator = np.random.default_rng(seed)
    best: RankOneMotifs | None = None
    for _ in range(restarts):
        left, _ = np.linalg.qr(generator.standard_normal((values.shape[1], rank)))
        right, _ = np.linalg.qr(generator.standard_normal((values.shape[2], rank)))
        previous = np.inf
        completed = 0
        for step in range(iterations):
            coefficients = _coefficients(values, left, right, ridge)

            gram = (coefficients.T @ coefficients) * (right.T @ right)
            rhs = np.einsum("nij,nk,jk->ik", values, coefficients, right, optimize=True)
            scale = max(float(np.trace(gram)) / rank, 1.0)
            left = np.linalg.solve(gram + ridge * scale * np.eye(rank), rhs.T).T

            gram = (coefficients.T @ coefficients) * (left.T @ left)
            rhs = np.einsum("nij,nk,ik->jk", values, coefficients, left, optimize=True)
            scale = max(float(np.trace(gram)) / rank, 1.0)
            right = np.linalg.solve(gram + ridge * scale * np.eye(rank), rhs.T).T

            left_norms = np.maximum(np.linalg.norm(left, axis=0), 1e-12)
            right_norms = np.maximum(np.linalg.norm(right, axis=0), 1e-12)
            left /= left_norms
            right /= right_norms

            reconstruction = reconstruct_rank_one_motifs(
                values, RankOneMotifs(left, right, np.nan, step + 1), ridge
            )
            loss = float(np.sum((values - reconstruction) ** 2) / np.sum(values**2))
            completed = step + 1
            if previous - loss >= 0 and previous - loss < tolerance:
                break
            previous = loss

        candidate = RankOneMotifs(left, right, loss, completed)
        if best is None or candidate.training_loss < best.training_loss:
            best = candidate
    if best is None:  # pragma: no cover - guarded by the restart validation
        raise RuntimeError("motif fitting produced no candidate")
    return best
