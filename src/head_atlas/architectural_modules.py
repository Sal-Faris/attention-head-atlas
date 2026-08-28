"""Externally anchored residual-axis modules and module-pair energy statistics."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

Array = np.ndarray


def axis_usage_profiles(covariances: tuple[Array, ...], basis: Array) -> Array:
    """Describe each basis axis by its nonnegative usage across anchor components."""

    vectors = np.asarray(basis, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] < 2:
        raise ValueError("basis must contain at least two axes")
    if not covariances:
        raise ValueError("at least one anchor covariance is required")
    profiles = []
    for covariance in covariances:
        values = np.asarray(covariance, dtype=np.float64)
        if values.shape != (vectors.shape[0], vectors.shape[0]):
            raise ValueError("anchor covariance does not match basis width")
        if not np.isfinite(values).all():
            raise ValueError("anchor covariance must be finite")
        profiles.append(np.maximum(np.diag(vectors.T @ values @ vectors), 0.0))
    usage = np.stack(profiles, axis=1)
    norms = np.linalg.norm(usage, axis=1, keepdims=True)
    return usage / np.maximum(norms, 1e-15)


def partition_axis_profiles(profiles: Array, cluster_count: int, *, seed: int) -> Array:
    """Partition axes by cosine-equivalent anchor-usage profiles."""

    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("profiles must be a finite two-dimensional population")
    if cluster_count < 2 or cluster_count > len(values):
        raise ValueError("cluster count must lie between two and the axis count")
    labels = KMeans(n_clusters=cluster_count, n_init=20, random_state=seed).fit_predict(values)
    if len(np.unique(labels)) != cluster_count:
        raise RuntimeError("axis partition lost a requested cluster")
    return np.asarray(labels, dtype=np.int64)


def module_pair_energy(
    coefficients: Array,
    read_labels: Array,
    write_labels: Array,
    cluster_count: int,
) -> tuple[Array, Array]:
    """Return energy and scalar area for every read-module/write-module pair."""

    values = np.asarray(coefficients, dtype=np.float64)
    read = np.asarray(read_labels, dtype=np.int64)
    write = np.asarray(write_labels, dtype=np.int64)
    if values.ndim != 2 or read.shape != (values.shape[0],) or write.shape != (values.shape[1],):
        raise ValueError("coefficient matrix and module labels do not align")
    if cluster_count < 1 or np.any(read < 0) or np.any(write < 0):
        raise ValueError("module labels must be nonnegative")
    if np.any(read >= cluster_count) or np.any(write >= cluster_count):
        raise ValueError("module label exceeds the cluster count")
    energy = np.zeros((cluster_count, cluster_count), dtype=np.float64)
    area = np.zeros_like(energy)
    for read_label in range(cluster_count):
        rows = read == read_label
        for write_label in range(cluster_count):
            columns = write == write_label
            block = values[np.ix_(rows, columns)]
            energy[read_label, write_label] = float(np.sum(block**2))
            area[read_label, write_label] = int(np.sum(rows)) * int(np.sum(columns))
    return energy, area


def top_pair_statistics(
    energy: Array,
    area: Array,
    pair_count: int,
) -> dict[str, float | Array]:
    """Summarize energy captured by the strongest externally defined module pairs."""

    energy_values = np.asarray(energy, dtype=np.float64)
    area_values = np.asarray(area, dtype=np.float64)
    if energy_values.shape != area_values.shape or energy_values.ndim != 2:
        raise ValueError("energy and area matrices must have equal two-dimensional shape")
    if pair_count < 1 or pair_count > energy_values.size:
        raise ValueError("invalid module-pair count")
    total_energy = float(np.sum(energy_values))
    total_area = float(np.sum(area_values))
    if total_energy <= 0.0 or total_area <= 0.0:
        raise ValueError("module pairs need positive energy and area")
    flat = energy_values.ravel()
    selected = np.argpartition(flat, -pair_count)[-pair_count:]
    energy_fraction = float(np.sum(flat[selected]) / total_energy)
    area_fraction = float(np.sum(area_values.ravel()[selected]) / total_area)
    probabilities = flat / total_energy
    positive = probabilities > 0.0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return {
        "energy_fraction": energy_fraction,
        "area_fraction": area_fraction,
        "energy_enrichment": energy_fraction / max(area_fraction, 1e-15),
        "effective_pair_count": float(np.exp(entropy)),
        "selected_flat_indices": np.sort(selected),
    }
