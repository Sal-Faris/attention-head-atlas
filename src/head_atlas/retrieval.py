"""External-family retrieval metrics for unsupervised head geometry."""

from collections.abc import Mapping, Sequence

import numpy as np

Array = np.ndarray
Location = tuple[int, int]


def _validated_population(
    distance_matrix: Array,
    layers: Sequence[int],
    heads: Sequence[int],
    tolerance: float,
) -> tuple[Array, Array, Array, dict[Location, int]]:
    distances = np.asarray(distance_matrix, dtype=np.float64)
    layer_array = np.asarray(layers, dtype=np.int64)
    head_array = np.asarray(heads, dtype=np.int64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    if layer_array.shape != (len(distances),) or head_array.shape != (len(distances),):
        raise ValueError("layers and heads must contain one value per item")
    if not np.isfinite(distances).all():
        raise ValueError("distance matrix contains non-finite values")
    if not np.allclose(distances, distances.T, rtol=0.0, atol=tolerance):
        raise ValueError("distance matrix must be symmetric")
    if not np.allclose(np.diag(distances), 0.0, rtol=0.0, atol=tolerance):
        raise ValueError("distance matrix diagonal must be zero")

    index_by_location: dict[Location, int] = {}
    for index, (layer, head) in enumerate(zip(layer_array, head_array, strict=True)):
        location = (int(layer), int(head))
        if location in index_by_location:
            raise ValueError(f"duplicate head location L{layer}H{head}")
        index_by_location[location] = index
    return distances, layer_array, head_array, index_by_location


def _average_precision(distances: Array, anchor: int, positives: set[int]) -> tuple[float, int]:
    candidates = np.asarray([index for index in range(len(distances)) if index != anchor])
    order = candidates[np.argsort(distances[anchor, candidates], kind="stable")]
    positive_ranks = [rank for rank, index in enumerate(order, start=1) if index in positives]
    average_precision = float(
        np.mean([(positive_number + 1) / rank for positive_number, rank in enumerate(positive_ranks)])
    )
    return average_precision, min(positive_ranks)


def _rank_matrix(distances: Array) -> Array:
    """Return one-based neighbour ranks, with zero on the diagonal."""

    item_count = len(distances)
    ranks = np.zeros((item_count, item_count), dtype=np.int64)
    for anchor in range(item_count):
        candidates = np.asarray([index for index in range(item_count) if index != anchor])
        order = candidates[np.argsort(distances[anchor, candidates], kind="stable")]
        ranks[anchor, order] = np.arange(1, item_count)
    return ranks


def _family_balanced_map(rank_matrix: Array, index_families: Mapping[str, Sequence[int]]) -> float:
    family_scores = []
    for indices in index_families.values():
        anchor_scores = []
        for anchor in indices:
            positive_ranks = np.sort(
                [rank_matrix[anchor, target] for target in indices if target != anchor]
            )
            precisions = np.arange(1, len(positive_ranks) + 1) / positive_ranks
            anchor_scores.append(float(np.mean(precisions)))
        family_scores.append(float(np.mean(anchor_scores)))
    return float(np.mean(family_scores))


def _pair_percentile(value: float, reference: Array) -> float:
    less = np.count_nonzero(reference < value)
    equal = np.count_nonzero(reference == value)
    return float((less + 0.5 * equal) / reference.size)


def evaluate_family_retrieval(
    distance_matrix: Array,
    layers: Sequence[int],
    heads: Sequence[int],
    families: Mapping[str, Sequence[Location]],
    tolerance: float = 1e-10,
) -> dict[str, object]:
    """Measure whether known family members retrieve one another locally.

    Labels are used only after the distance matrix is fixed. Unknown heads are
    retained in the ranking, making average precision conservative when the
    published family labels are incomplete.
    """

    distances, layer_array, _, index_by_location = _validated_population(
        distance_matrix, layers, heads, tolerance
    )
    if not families:
        raise ValueError("at least one family is required")

    index_families: dict[str, list[int]] = {}
    used_indices: set[int] = set()
    for family_id, locations in families.items():
        indices = []
        for location in locations:
            normalized_location = (int(location[0]), int(location[1]))
            if normalized_location not in index_by_location:
                raise ValueError(
                    f"family {family_id} references missing head "
                    f"L{normalized_location[0]}H{normalized_location[1]}"
                )
            index = index_by_location[normalized_location]
            if index in used_indices:
                raise ValueError("primary retrieval families cannot overlap")
            used_indices.add(index)
            indices.append(index)
        if len(indices) < 2:
            raise ValueError(f"family {family_id} must contain at least two heads")
        index_families[family_id] = indices

    all_pair_distances = distances[np.triu_indices(len(distances), k=1)]
    family_results: dict[str, dict[str, int | float | list[float]]] = {}
    for family_id, indices in index_families.items():
        family_set = set(indices)
        within_distances = np.asarray(
            [
                distances[first, second]
                for position, first in enumerate(indices)
                for second in indices[position + 1 :]
            ]
        )
        pair_percentiles = [
            _pair_percentile(float(value), all_pair_distances) for value in within_distances
        ]

        average_precisions = []
        reciprocal_ranks = []
        nearest_hits = []
        recall_at_five = []
        layer_matched_percentiles = []
        for anchor in indices:
            positives = family_set - {anchor}
            average_precision, first_positive_rank = _average_precision(
                distances, anchor, positives
            )
            average_precisions.append(average_precision)
            reciprocal_ranks.append(1.0 / first_positive_rank)
            nearest_hits.append(float(first_positive_rank == 1))

            nearest_order = np.argsort(distances[anchor], kind="stable")
            top_five = [index for index in nearest_order if index != anchor][:5]
            recall_at_five.append(len(positives.intersection(top_five)) / len(positives))

            for target in positives:
                layer_candidates = np.flatnonzero(layer_array == layer_array[target])
                layer_candidates = layer_candidates[layer_candidates != anchor]
                target_distance = distances[anchor, target]
                layer_matched_percentiles.append(
                    _pair_percentile(target_distance, distances[anchor, layer_candidates])
                )

        family_results[family_id] = {
            "family_size": len(indices),
            "pair_count": int(within_distances.size),
            "mean_pair_distance": float(np.mean(within_distances)),
            "mean_global_pair_percentile": float(np.mean(pair_percentiles)),
            "median_global_pair_percentile": float(np.median(pair_percentiles)),
            "mean_layer_matched_pair_percentile": float(
                np.mean(layer_matched_percentiles)
            ),
            "mean_average_precision": float(np.mean(average_precisions)),
            "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
            "nearest_neighbour_hit_rate": float(np.mean(nearest_hits)),
            "mean_recall_at_5": float(np.mean(recall_at_five)),
            "pair_distances": [float(value) for value in within_distances],
            "pair_percentiles": pair_percentiles,
        }

    aggregate_names = (
        "mean_global_pair_percentile",
        "mean_layer_matched_pair_percentile",
        "mean_average_precision",
        "mean_reciprocal_rank",
        "nearest_neighbour_hit_rate",
        "mean_recall_at_5",
    )
    aggregate = {
        name: float(np.mean([family_results[family_id][name] for family_id in family_results]))
        for name in aggregate_names
    }
    aggregate["family_count"] = len(family_results)
    aggregate["labelled_head_count"] = len(used_indices)
    return {"families": family_results, "aggregate": aggregate}


def permutation_retrieval_test(
    distance_matrix: Array,
    layers: Sequence[int],
    heads: Sequence[int],
    families: Mapping[str, Sequence[Location]],
    permutations: int = 9999,
    seed: int = 0,
    stratify_by_layer: bool = False,
) -> dict[str, int | float | bool]:
    """Test family-balanced mean average precision under label permutation."""

    if permutations < 1:
        raise ValueError("permutations must be positive")
    distances, layer_array, _, index_by_location = _validated_population(
        distance_matrix, layers, heads, tolerance=1e-10
    )
    original_index_families = {
        family_id: [index_by_location[(int(layer), int(head))] for layer, head in locations]
        for family_id, locations in families.items()
    }
    rank_matrix = _rank_matrix(distances)
    observed = _family_balanced_map(rank_matrix, original_index_families)

    rng = np.random.default_rng(seed)
    null_values = np.empty(permutations, dtype=np.float64)
    for repetition in range(permutations):
        mapping = np.arange(len(distances))
        if stratify_by_layer:
            for layer in np.unique(layer_array):
                layer_indices = np.flatnonzero(layer_array == layer)
                mapping[layer_indices] = rng.permutation(layer_indices)
        else:
            mapping = rng.permutation(mapping)

        permuted_index_families = {
            family_id: [int(mapping[index]) for index in indices]
            for family_id, indices in original_index_families.items()
        }
        null_values[repetition] = _family_balanced_map(
            rank_matrix, permuted_index_families
        )

    return {
        "permutations": permutations,
        "seed": seed,
        "stratify_by_layer": stratify_by_layer,
        "observed_family_balanced_map": float(observed),
        "null_mean": float(np.mean(null_values)),
        "null_standard_deviation": float(np.std(null_values)),
        "null_minimum": float(np.min(null_values)),
        "null_maximum": float(np.max(null_values)),
        "upper_tail_p_value": float(
            (1 + np.count_nonzero(null_values >= observed)) / (permutations + 1)
        ),
    }
