"""Prompt-free definitions of matched conditional QK routing events."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Array = np.ndarray

DEFAULT_OFFSET_BINS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 63),
)


def qk_logits(queries: Array, keys: Array, *, scale: float | None = None) -> Array:
    """Return scaled QK logits for equally shaped query and key arrays.

    The final two axes are position and head-coordinate width; all preceding
    axes are batch-like and preserved.  Inputs should already include RoPE
    when reproducing a model's actual attention logits.
    """

    query_array = np.asarray(queries, dtype=np.float64)
    key_array = np.asarray(keys, dtype=np.float64)
    if query_array.ndim < 2 or query_array.shape != key_array.shape:
        raise ValueError("queries and keys must have matching shape (..., position, width)")
    if not np.isfinite(query_array).all() or not np.isfinite(key_array).all():
        raise ValueError("queries and keys must be finite")
    width = query_array.shape[-1]
    if width < 1:
        raise ValueError("head width must be positive")
    resolved_scale = np.sqrt(width) if scale is None else float(scale)
    if not np.isfinite(resolved_scale) or resolved_scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return np.einsum("...id,...jd->...ij", query_array, key_array) / resolved_scale


def causal_softmax(logits: Array) -> Array:
    """Apply the strict causal mask followed by a numerically stable softmax."""

    logit_array = _square_logits(logits)
    sequence_length = logit_array.shape[-1]
    allowed = np.tri(sequence_length, dtype=bool)
    masked = np.where(allowed, logit_array, -np.inf)
    row_maximum = np.max(masked, axis=-1, keepdims=True)
    exponentiated = np.exp(masked - row_maximum)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def relative_offset_statistics(logits: Array) -> tuple[Array, Array]:
    """Return mean and standard deviation of logits for every exact offset.

    The last axis indexes `delta = destination - source`.  All batch-like
    axes are aggregated, so the result has shape `(sequence_length,)`.
    """

    logit_array = _square_logits(logits)
    sequence_length = logit_array.shape[-1]
    means = np.empty(sequence_length, dtype=np.float64)
    standard_deviations = np.empty(sequence_length, dtype=np.float64)
    for offset in range(sequence_length):
        values = np.diagonal(logit_array, offset=-offset, axis1=-2, axis2=-1)
        means[offset] = np.mean(values)
        standard_deviations[offset] = np.std(values)
    return means, standard_deviations


def residualize_by_offset(
    logits: Array,
    means: Array,
    standard_deviations: Array,
    *,
    epsilon: float = 1e-8,
) -> Array:
    """Center and scale every causal logit by frozen exact-offset statistics."""

    logit_array = _square_logits(logits)
    sequence_length = logit_array.shape[-1]
    mean_array = _offset_vector(means, sequence_length, "means")
    standard_deviation_array = _offset_vector(
        standard_deviations, sequence_length, "standard_deviations"
    )
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    destination, source = np.indices((sequence_length, sequence_length))
    offsets = destination - source
    safe_offsets = np.maximum(offsets, 0)
    residualized = (logit_array - mean_array[safe_offsets]) / (
        standard_deviation_array[safe_offsets] + epsilon
    )
    return np.where(offsets >= 0, residualized, np.nan)


def offset_bin(offset: int, bins: Sequence[tuple[int, int]] = DEFAULT_OFFSET_BINS) -> int:
    """Return the index of the inclusive relative-offset bin, or -1 if absent."""

    if offset < 1:
        return -1
    for index, (minimum, maximum) in enumerate(bins):
        if minimum < 1 or maximum < minimum:
            raise ValueError("offset bins must be ordered inclusive positive intervals")
        if minimum <= offset <= maximum:
            return index
    return -1


def matched_source_events(
    residual_logits: Array,
    *,
    minimum_destination: int = 8,
    bins: Sequence[tuple[int, int]] = DEFAULT_OFFSET_BINS,
) -> Array:
    """Select deterministic high-score versus matched-neutral source events.

    `residual_logits` has shape `(destination, source)`.  Each returned row is
    `(destination, positive_source, negative_source, offset_bin)`.  Only
    strict-past, non-BOS source positions are eligible.
    """

    residual_array = _square_logits(residual_logits, allow_nan=True)
    if minimum_destination < 1:
        raise ValueError("minimum_destination must be positive")
    sequence_length = residual_array.shape[-1]
    if sequence_length < 2:
        return np.empty((0, 4), dtype=np.int64)

    events: list[tuple[int, int, int, int]] = []
    for destination in range(minimum_destination, sequence_length):
        sources = np.arange(1, destination, dtype=np.int64)
        if len(sources) < 2:
            continue
        scores = residual_array[destination, sources]
        finite = np.isfinite(scores)
        sources = sources[finite]
        scores = scores[finite]
        if len(sources) < 2:
            continue
        positive_source = int(sources[np.argmax(scores)])
        positive_offset = destination - positive_source
        bin_index = offset_bin(positive_offset, bins)
        if bin_index < 0:
            continue
        minimum, maximum = bins[bin_index]
        same_bin = sources[
            ((destination - sources) >= minimum) & ((destination - sources) <= maximum)
        ]
        same_bin = same_bin[same_bin != positive_source]
        if len(same_bin) == 0:
            continue
        same_bin_scores = residual_array[destination, same_bin]
        median = float(np.median(same_bin_scores))
        negative_source = int(same_bin[np.argmin(np.abs(same_bin_scores - median))])
        events.append((destination, positive_source, negative_source, bin_index))
    return np.asarray(events, dtype=np.int64).reshape(-1, 4)


def batched_matched_source_events(
    residual_logits: Array,
    *,
    minimum_destination: int = 8,
    bins: Sequence[tuple[int, int]] = DEFAULT_OFFSET_BINS,
) -> list[Array]:
    """Select matched events independently for each leading-index logit matrix."""

    residual_array = _square_logits(residual_logits, allow_nan=True)
    if residual_array.ndim == 2:
        return [matched_source_events(residual_array, minimum_destination=minimum_destination, bins=bins)]
    matrices = residual_array.reshape((-1, *residual_array.shape[-2:]))
    return [
        matched_source_events(matrix, minimum_destination=minimum_destination, bins=bins)
        for matrix in matrices
    ]


def max_attention_difference(first: Array, second: Array) -> float:
    """Return the maximum absolute difference between aligned attention arrays."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape:
        raise ValueError("attention arrays must have matching shapes")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("attention arrays must be finite")
    return float(np.max(np.abs(first_array - second_array)))


def _square_logits(logits: Array, *, allow_nan: bool = False) -> Array:
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2] != array.shape[-1]:
        raise ValueError("logits must have shape (..., destination, source) with square final axes")
    finite = np.isfinite(array)
    if allow_nan:
        if np.any(np.isinf(array)):
            raise ValueError("logits cannot contain infinite values")
    elif not finite.all():
        raise ValueError("logits must be finite")
    return array


def _offset_vector(values: Array, sequence_length: int, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (sequence_length,):
        raise ValueError(f"{name} must have shape (sequence_length,)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array
