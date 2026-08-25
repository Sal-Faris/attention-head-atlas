"""Deterministic low-rank bilinear compression of QK margins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class BilinearMarginModel:
    """A rank-r map whose score is ``query @ left @ right.T @ key_difference``."""

    left: Array
    right: Array
    training_mse: float


def qk_margins(queries: Array, key_differences: Array) -> Array:
    """Return the actual scaled QK margin for aligned query/key-difference events."""

    query_array, key_array = _aligned_events(queries, key_differences)
    return np.sum(query_array * key_array, axis=1) / np.sqrt(query_array.shape[1])


def bilinear_scores(queries: Array, key_differences: Array, model: BilinearMarginModel) -> Array:
    """Score aligned events under a low-rank bilinear margin model."""

    query_array, key_array = _aligned_events(queries, key_differences)
    if model.left.shape != model.right.shape or model.left.shape[0] != query_array.shape[1]:
        raise ValueError("model factor width must match the event width")
    return np.sum((query_array @ model.left) * (key_array @ model.right), axis=1) / np.sqrt(
        query_array.shape[1]
    )


def fit_bilinear_margin_model(
    queries: Array,
    key_differences: Array,
    *,
    rank: int,
    ridge: float,
    iterations: int = 400,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> BilinearMarginModel:
    """Fit a rank-r QK-margin compressor with deterministic full-batch Adam."""

    query_array, key_array = _aligned_events(queries, key_differences)
    width = query_array.shape[1]
    if rank < 1 or rank > width or ridge < 0.0 or iterations < 1 or learning_rate <= 0.0:
        raise ValueError("invalid rank, ridge, iteration count, or learning rate")
    target = qk_margins(query_array, key_array)
    initialization = query_array.T @ (key_array * target[:, None]) / len(target)
    left_vectors, _, right_transpose = np.linalg.svd(initialization, full_matrices=False)
    left = left_vectors[:, :rank].copy()
    right = right_transpose[:rank].T.copy()
    first_left = np.zeros_like(left)
    second_left = np.zeros_like(left)
    first_right = np.zeros_like(right)
    second_right = np.zeros_like(right)
    beta1, beta2 = 0.9, 0.999
    for step in range(1, iterations + 1):
        query_features = query_array @ left
        key_features = key_array @ right
        target_scale = max(float(np.std(target)), 1e-6)
        residual = (
            (np.sum(query_features * key_features, axis=1) / np.sqrt(width)) - target
        ) / target_scale
        gradient_left = query_array.T @ (residual[:, None] * key_features) / (
            len(target) * np.sqrt(width)
        ) / target_scale + ridge * left
        gradient_right = key_array.T @ (residual[:, None] * query_features) / (
            len(target) * np.sqrt(width)
        ) / target_scale + ridge * right
        first_left = beta1 * first_left + (1.0 - beta1) * gradient_left
        second_left = beta2 * second_left + (1.0 - beta2) * gradient_left**2
        first_right = beta1 * first_right + (1.0 - beta1) * gradient_right
        second_right = beta2 * second_right + (1.0 - beta2) * gradient_right**2
        left -= learning_rate * (first_left / (1.0 - beta1**step)) / (
            np.sqrt(second_left / (1.0 - beta2**step)) + 1e-8
        )
        right -= learning_rate * (first_right / (1.0 - beta1**step)) / (
            np.sqrt(second_right / (1.0 - beta2**step)) + 1e-8
        )
    model = BilinearMarginModel(left=left, right=right, training_mse=0.0)
    mse = float(np.mean((bilinear_scores(query_array, key_array, model) - target) ** 2))
    return BilinearMarginModel(left=left, right=right, training_mse=mse)


def projected_identity_scores(queries: Array, key_differences: Array, basis: Array) -> Array:
    """Score QK margins after applying one orthogonal projection to the query."""

    query_array, key_array = _aligned_events(queries, key_differences)
    basis_array = np.asarray(basis, dtype=np.float64)
    if basis_array.ndim != 2 or basis_array.shape[0] != query_array.shape[1]:
        raise ValueError("basis width must match the event width")
    orthonormal, _ = np.linalg.qr(basis_array, mode="reduced")
    return np.sum((query_array @ orthonormal) * (key_array @ orthonormal), axis=1) / np.sqrt(
        query_array.shape[1]
    )


def r_squared(prediction: Array, target: Array) -> float:
    """Return population R-squared, allowing an explicit negative result."""

    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 1 or len(predicted) < 2:
        raise ValueError("prediction and target must be aligned one-dimensional arrays")
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    if denominator <= 0.0:
        raise ValueError("target has no variance")
    return float(1.0 - np.sum((predicted - observed) ** 2) / denominator)


def _aligned_events(queries: Array, key_differences: Array) -> tuple[Array, Array]:
    query_array = np.asarray(queries, dtype=np.float64)
    key_array = np.asarray(key_differences, dtype=np.float64)
    if query_array.ndim != 2 or query_array.shape != key_array.shape or query_array.shape[0] < 2:
        raise ValueError("queries and key_differences must be matching event-by-width arrays")
    if not np.isfinite(query_array).all() or not np.isfinite(key_array).all():
        raise ValueError("events must be finite")
    return query_array, key_array
