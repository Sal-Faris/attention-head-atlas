"""Low-dimensional embeddings derived from pairwise operator distances."""

import numpy as np

Array = np.ndarray


def classical_mds(
    distance_matrix: Array,
    dimensions: int | None = None,
    tolerance: float = 1e-10,
) -> dict[str, Array | float]:
    """Embed a distance matrix with classical multidimensional scaling.

    Classical MDS, also called principal coordinates analysis (PCoA), converts
    squared distances into a centered Gram matrix

    ``B = -0.5 * H @ D**2 @ H``

    and eigendecomposes ``B``.  For Euclidean distances, its positive
    eigenvalues and eigenvectors recover the centered points up to an
    orthogonal transformation.

    Args:
        distance_matrix: Square, symmetric matrix with a zero diagonal.
        dimensions: Number of positive-eigenvalue coordinates to retain. If
            omitted, retain every numerically positive coordinate.
        tolerance: Relative numerical tolerance used when classifying
            eigenvalues as positive, zero, or negative.

    Returns:
        A dictionary containing the coordinates, all eigenvalues in descending
        order, their positive explained-variance ratios, and the fraction of
        absolute eigenvalue mass lying in significantly negative eigenvalues.
        Negative mass near zero is expected for Euclidean input distances;
        substantial negative mass indicates that Euclidean coordinates cannot
        reproduce the supplied distances exactly.
    """

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    if distances.shape[0] < 2:
        raise ValueError("distance matrix must contain at least two items")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if dimensions is not None and dimensions < 1:
        raise ValueError("dimensions must be positive")
    if not np.isfinite(distances).all():
        raise ValueError("distance matrix contains non-finite values")
    if not np.allclose(distances, distances.T, rtol=0.0, atol=tolerance):
        raise ValueError("distance matrix must be symmetric")
    if not np.allclose(np.diag(distances), 0.0, rtol=0.0, atol=tolerance):
        raise ValueError("distance matrix diagonal must be zero")
    if np.min(distances) < -tolerance:
        raise ValueError("distance matrix contains negative values")

    squared_distances = np.maximum(distances, 0.0) ** 2
    row_means = np.mean(squared_distances, axis=1, keepdims=True)
    column_means = np.mean(squared_distances, axis=0, keepdims=True)
    grand_mean = np.mean(squared_distances)
    gram = -0.5 * (squared_distances - row_means - column_means + grand_mean)
    gram = 0.5 * (gram + gram.T)

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    cutoff = tolerance * eigenvalue_scale
    positive = eigenvalues > cutoff
    negative = eigenvalues < -cutoff
    positive_count = int(np.count_nonzero(positive))
    retained_count = positive_count if dimensions is None else min(dimensions, positive_count)

    retained_eigenvalues = eigenvalues[:retained_count]
    coordinates = eigenvectors[:, :retained_count] * np.sqrt(retained_eigenvalues)

    positive_mass = float(np.sum(eigenvalues[positive]))
    explained_variance_ratio = np.zeros_like(eigenvalues)
    if positive_mass > 0:
        explained_variance_ratio[positive] = eigenvalues[positive] / positive_mass

    absolute_mass = float(np.sum(np.abs(eigenvalues)))
    negative_mass = float(np.sum(np.abs(eigenvalues[negative])))
    negative_mass_ratio = negative_mass / absolute_mass if absolute_mass > 0 else 0.0

    return {
        "coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "negative_eigenvalue_mass_ratio": negative_mass_ratio,
    }
