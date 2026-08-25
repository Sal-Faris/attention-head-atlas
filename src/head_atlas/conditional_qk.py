"""Gauge-aware low-rank analysis of matched conditional QK events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ConditionalQKSubspace:
    """Whitened conditional QK factors for one head and one selected rank."""

    singular_values: Array
    query_basis: Array
    key_basis: Array
    query_mean: Array
    key_mean: Array
    shrinkage: float


def trace_shrinkage(covariance: Array, shrinkage: float) -> Array:
    """Shrink a covariance matrix toward its trace-scaled identity matrix."""

    matrix = _symmetric_square(covariance, "covariance")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    width = len(matrix)
    target = np.eye(width) * np.trace(matrix) / width
    return (1.0 - shrinkage) * matrix + shrinkage * target


def psd_inverse_sqrt(matrix: Array, *, tolerance: float = 1e-10) -> Array:
    """Return a symmetric inverse square root of a positive semidefinite matrix."""

    covariance = _symmetric_square(matrix, "matrix")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    values, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if np.min(values) < -tolerance * scale:
        raise ValueError("matrix is not positive semidefinite")
    retained = values > tolerance * scale
    if not np.any(retained):
        raise ValueError("matrix has no numerically positive direction")
    return (vectors[:, retained] / np.sqrt(values[retained])) @ vectors[:, retained].T


def conditional_cross_covariance(
    queries: Array,
    positive_keys: Array,
    negative_keys: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Estimate the matched-event cross-covariance and its two side covariances."""

    query_array = _event_matrix(queries, "queries")
    positive_array = _event_matrix(positive_keys, "positive_keys")
    negative_array = _event_matrix(negative_keys, "negative_keys")
    if query_array.shape != positive_array.shape or query_array.shape != negative_array.shape:
        raise ValueError("queries and matched key arrays must have the same shape")
    if len(query_array) < 2:
        raise ValueError("at least two matched events are required")

    query_mean = query_array.mean(axis=0)
    key_mean = np.concatenate((positive_array, negative_array), axis=0).mean(axis=0)
    centered_queries = query_array - query_mean
    centered_keys = np.concatenate((positive_array - key_mean, negative_array - key_mean), axis=0)
    key_difference = positive_array - negative_array
    count = len(query_array)
    cross_covariance = centered_queries.T @ key_difference / count
    query_covariance = centered_queries.T @ centered_queries / count
    key_covariance = centered_keys.T @ centered_keys / len(centered_keys)
    return cross_covariance, query_covariance, key_covariance, query_mean, key_mean


def fit_conditional_qk_subspace(
    queries: Array,
    positive_keys: Array,
    negative_keys: Array,
    *,
    rank: int,
    shrinkage: float,
) -> ConditionalQKSubspace:
    """Fit whitened low-rank query/key directions distinguishing matched sources."""

    cross, query_covariance, key_covariance, query_mean, key_mean = conditional_cross_covariance(
        queries, positive_keys, negative_keys
    )
    width = cross.shape[0]
    if rank < 1 or rank > width:
        raise ValueError("rank must lie between 1 and the head width")
    query_inverse_sqrt = psd_inverse_sqrt(trace_shrinkage(query_covariance, shrinkage))
    key_inverse_sqrt = psd_inverse_sqrt(trace_shrinkage(key_covariance, shrinkage))
    left, singular_values, right_transpose = np.linalg.svd(
        query_inverse_sqrt @ cross @ key_inverse_sqrt,
        full_matrices=False,
    )
    return ConditionalQKSubspace(
        singular_values=singular_values,
        query_basis=query_inverse_sqrt @ left[:, :rank],
        key_basis=key_inverse_sqrt @ right_transpose.T[:, :rank],
        query_mean=query_mean,
        key_mean=key_mean,
        shrinkage=shrinkage,
    )


def orthogonal_projector(basis: Array, *, tolerance: float = 1e-10) -> Array:
    """Construct the Euclidean projector onto a full-column-rank basis."""

    matrix = _event_matrix(basis, "basis")
    if matrix.shape[1] > matrix.shape[0]:
        raise ValueError("basis cannot have more columns than rows")
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values[-1] <= tolerance * max(float(singular_values[0]), 1.0):
        raise ValueError("basis is numerically rank deficient")
    orthonormal, _ = np.linalg.qr(matrix, mode="reduced")
    return orthonormal @ orthonormal.T


def mapped_residual_projector(
    reader: Array,
    coefficient_basis: Array,
    *,
    center_residual: bool = True,
) -> Array:
    """Map a head-coordinate basis into residual coordinates before comparison.

    `reader` should be the processed query/key factor with layer-norm gain
    folded in.  It therefore represents the same residual-space convention as
    the project-wide static QK analysis.
    """

    reader_array = _event_matrix(reader, "reader")
    basis_array = _event_matrix(coefficient_basis, "coefficient_basis")
    if reader_array.shape[1] != basis_array.shape[0]:
        raise ValueError("reader width must equal the coefficient basis width")
    mapped = reader_array @ basis_array
    if center_residual:
        mapped = mapped - mapped.mean(axis=0, keepdims=True)
    return orthogonal_projector(mapped)


def normalized_chordal_distance(first: Array, second: Array, *, tolerance: float = 1e-8) -> float:
    """Return a rank-normalized chordal distance between equal-rank projectors."""

    first_projector = _symmetric_square(first, "first")
    second_projector = _symmetric_square(second, "second")
    if first_projector.shape != second_projector.shape:
        raise ValueError("projectors must have matching shape")
    rank_first = float(np.trace(first_projector))
    rank_second = float(np.trace(second_projector))
    if rank_first <= tolerance or abs(rank_first - rank_second) > tolerance:
        raise ValueError("projectors must have the same positive rank")
    squared = max(rank_first - float(np.trace(first_projector @ second_projector)), 0.0)
    return float(np.sqrt(squared / rank_first))


def query_feature_margin(
    queries: Array,
    positive_keys: Array,
    negative_keys: Array,
    subspace: ConditionalQKSubspace,
) -> tuple[Array, Array, Array]:
    """Split pre-RoPE matched margins into learned-query and residual parts.

    This is a coordinate-space diagnostic.  The final analysis applies the
    identical decomposition before using post-RoPE keys for exact logits.
    """

    query_array = _event_matrix(queries, "queries")
    positive_array = _event_matrix(positive_keys, "positive_keys")
    negative_array = _event_matrix(negative_keys, "negative_keys")
    if query_array.shape != positive_array.shape or query_array.shape != negative_array.shape:
        raise ValueError("query and key arrays must have matching shape")
    projector = orthogonal_projector(subspace.query_basis)
    centered = query_array - subspace.query_mean
    key_difference = positive_array - negative_array
    feature = np.sum((centered @ projector) * key_difference, axis=1)
    residual = np.sum((centered @ (np.eye(query_array.shape[1]) - projector)) * key_difference, axis=1)
    mean = np.sum(subspace.query_mean * key_difference, axis=1)
    return feature, residual, mean


def document_bootstrap_mean(
    values: Array,
    document_ids: Array,
    *,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Return a document-resampling interval for an event-level mean."""

    value_array = np.asarray(values, dtype=np.float64)
    document_array = np.asarray(document_ids)
    if value_array.ndim != 1 or document_array.shape != value_array.shape:
        raise ValueError("values and document_ids must be aligned one-dimensional arrays")
    if not np.isfinite(value_array).all() or repetitions < 1:
        raise ValueError("values must be finite and repetitions positive")
    unique_documents = np.unique(document_array)
    if len(unique_documents) < 2:
        raise ValueError("at least two documents are required for a document bootstrap")
    per_document = np.asarray(
        [value_array[document_array == document].mean() for document in unique_documents]
    )
    draws = rng.integers(0, len(per_document), size=(repetitions, len(per_document)))
    samples = per_document[draws].mean(axis=1)
    return {
        "mean": float(per_document.mean()),
        "bootstrap_standard_deviation": float(samples.std(ddof=1)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def benjamini_hochberg(p_values: Array) -> Array:
    """Return monotone Benjamini--Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("p_values must be a non-empty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _event_matrix(values: Array, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _symmetric_square(values: Array, name: str) -> Array:
    array = _event_matrix(values, name)
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(array, array.T, atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} must be symmetric")
    return array
