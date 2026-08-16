"""Human-readable diagnostics for individual head operators."""

from collections.abc import Iterable, Sequence

import numpy as np

from .operators import HeadOperator

DEFAULT_ENERGY_CUTOFFS = (1, 4, 8, 16, 32, 64)


def operator_record(
    operator: HeadOperator,
    energy_cutoffs: Sequence[int] = DEFAULT_ENERGY_CUTOFFS,
) -> dict[str, int | float | str]:
    """Summarize one operator using a single singular-value decomposition."""

    matrix = np.asarray(operator.matrix)
    input_epsilon = np.finfo(matrix.dtype).eps
    singular_values = np.linalg.svd(matrix.astype(np.float64), compute_uv=False)
    squared_values = singular_values**2
    total_energy = squared_values.sum()
    if total_energy == 0:
        raise ValueError("cannot summarize a zero operator")

    tolerance = max(matrix.shape) * input_epsilon * singular_values[0]
    resolved_values = singular_values[singular_values > tolerance]
    probabilities = resolved_values / resolved_values.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))

    record: dict[str, int | float | str] = {
        "layer": operator.layer,
        "head": operator.head,
        "kind": operator.kind,
        "frobenius_norm": float(np.sqrt(total_energy)),
        "spectral_norm": float(singular_values[0]),
        "effective_rank": float(np.exp(entropy)),
    }
    cumulative_energy = np.cumsum(squared_values) / total_energy
    for cutoff in energy_cutoffs:
        if cutoff < 1:
            raise ValueError("energy cutoffs must be positive")
        index = min(cutoff, singular_values.size) - 1
        record[f"top_{cutoff}_energy"] = float(cumulative_energy[index])
    return record


def operator_table(
    operators: Iterable[HeadOperator],
    energy_cutoffs: Sequence[int] = DEFAULT_ENERGY_CUTOFFS,
) -> list[dict[str, int | float | str]]:
    """Return one diagnostic record for every operator."""

    return [operator_record(operator, energy_cutoffs) for operator in operators]
