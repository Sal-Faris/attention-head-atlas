"""Prompt-independent architectural fingerprints for OV singular channels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import SplineTransformer, StandardScaler

Array = np.ndarray


@dataclass(frozen=True)
class CompartmentFit:
    """A variable-size channel partition selected by BIC."""

    labels: Array
    component_count: int
    bic: float
    confirmation_r2: float


def standardized_features(features: Array) -> Array:
    """Standardize informative columns and retain a valid constant fallback."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("features must contain at least two observations")
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    standard_deviation = np.std(values, axis=0)
    informative = standard_deviation > 1e-12
    if not np.any(informative):
        return np.zeros((len(values), 1), dtype=np.float64)
    return StandardScaler().fit_transform(values[:, informative])


def confirmation_r2(features: Array, labels: Array) -> float:
    """Fraction of held-out architectural variation separated by labels."""

    values = standardized_features(features)
    assignments = np.asarray(labels, dtype=np.int64)
    if assignments.shape != (len(values),):
        raise ValueError("labels do not align with observations")
    total = float(np.sum(values**2))
    if total <= 1e-12 or len(np.unique(assignments)) == 1:
        return 0.0
    between = 0.0
    for label in np.unique(assignments):
        selected = values[assignments == label]
        between += len(selected) * float(np.sum(np.mean(selected, axis=0) ** 2))
    return float(np.clip(between / total, 0.0, 1.0))


def residualize_against_gain(
    features: Array,
    gains: Array,
    *,
    n_knots: int = 6,
) -> Array:
    """Remove a flexible smooth dependence on singular gain from every feature."""

    values = np.asarray(features, dtype=np.float64)
    gain_values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 2 or gain_values.shape != (len(values),):
        raise ValueError("gains must align with feature observations")
    if len(values) < 4 or n_knots < 2 or n_knots > len(values):
        raise ValueError("gain residualization requires valid observations and knots")
    if not np.isfinite(values).all() or not np.isfinite(gain_values).all():
        raise ValueError("features and gains must be finite")
    if np.any(gain_values < 0):
        raise ValueError("singular gains must be nonnegative")

    scale = max(float(np.max(gain_values)), 1e-12)
    log_gain = np.log(np.maximum(gain_values / scale, 1e-12))[:, None]
    if float(np.std(log_gain)) <= 1e-12:
        return values - np.mean(values, axis=0, keepdims=True)
    design = SplineTransformer(
        n_knots=n_knots,
        degree=3,
        knots="quantile",
        include_bias=True,
    ).fit_transform(log_gain)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def fit_compartments(
    discovery: Array,
    confirmation: Array,
    *,
    maximum_components: int = 6,
    maximum_pca_dimensions: int = 8,
    seed: int = 0,
) -> CompartmentFit:
    """Select a diagonal-GMM channel partition by BIC and score held-out features."""

    discovery_values = standardized_features(discovery)
    confirmation_values = np.asarray(confirmation, dtype=np.float64)
    dimensions = min(maximum_pca_dimensions, discovery_values.shape[1], len(discovery_values) - 1)
    embedding = PCA(n_components=dimensions, random_state=seed).fit_transform(discovery_values)
    best: tuple[float, Array, int] | None = None
    for count in range(1, min(maximum_components, len(embedding) // 2) + 1):
        model = GaussianMixture(
            n_components=count,
            covariance_type="diag",
            reg_covar=1e-4,
            n_init=5,
            random_state=seed,
        ).fit(embedding)
        labels = model.predict(embedding)
        if count > 1 and np.min(np.bincount(labels)) < 2:
            continue
        bic = float(model.bic(embedding))
        if best is None or bic < best[0]:
            best = bic, labels, count
    if best is None:
        raise RuntimeError("no valid mixture fit")
    bic, labels, count = best
    return CompartmentFit(
        labels=labels,
        component_count=count,
        bic=bic,
        confirmation_r2=confirmation_r2(confirmation_values, labels),
    )


def weighted_subspace_overlap(modes: Array, anchor_modes: Array, weights: Array) -> Array:
    """Energy with which every mode overlaps a weighted anchor subspace."""

    mode_values = np.asarray(modes, dtype=np.float64)
    anchors = np.asarray(anchor_modes, dtype=np.float64)
    gains = np.asarray(weights, dtype=np.float64)
    if mode_values.ndim != 2 or anchors.ndim != 2 or mode_values.shape[0] != anchors.shape[0]:
        raise ValueError("modes and anchors must share their ambient width")
    if gains.shape != (anchors.shape[1],):
        raise ValueError("weights do not align with anchor modes")
    denominator = float(np.sum(gains**2))
    if denominator <= 1e-20:
        return np.zeros(mode_values.shape[1], dtype=np.float64)
    cross = mode_values.T @ anchors
    return np.sum(cross**2 * gains[None, :] ** 2, axis=1) / denominator


def factor_overlap(modes: Array, factor: Array) -> Array:
    """Normalized sensitivity of a raw reader factor to every orthonormal mode."""

    mode_values = np.asarray(modes, dtype=np.float64)
    factor_values = np.asarray(factor, dtype=np.float64)
    denominator = float(np.sum(factor_values**2))
    if denominator <= 1e-20:
        return np.zeros(mode_values.shape[1], dtype=np.float64)
    return np.sum((mode_values.T @ factor_values) ** 2, axis=1) / denominator
