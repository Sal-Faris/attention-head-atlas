"""Compare real operator geometry with spectrum-matched null populations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from head_atlas.distance_audit import summarize_distance_matrix
from head_atlas.distances import normalized_frobenius_distances
from head_atlas.model_io import load_operator_bundle
from head_atlas.nulls import resolved_singular_values, sample_spectrum_matched
from head_atlas.operators import HeadOperator
from head_atlas.structure import pcoa_spectrum_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def empirical_tail_probabilities(
    real_summary: dict[str, int | float],
    null_summaries: list[dict[str, int | float]],
) -> dict[str, dict[str, float]]:
    """Return plus-one lower- and upper-tail probabilities for float statistics."""

    probabilities = {}
    denominator = len(null_summaries) + 1
    for name, real_value in real_summary.items():
        if not isinstance(real_value, float):
            continue
        null_values = np.asarray([summary[name] for summary in null_summaries])
        probabilities[name] = {
            "lower_tail": float((1 + np.sum(null_values <= real_value)) / denominator),
            "upper_tail": float((1 + np.sum(null_values >= real_value)) / denominator),
        }
    return probabilities


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")

    operators, source_metadata = load_operator_bundle(args.input)
    real_distances = normalized_frobenius_distances(operators)
    real_summary = summarize_distance_matrix(real_distances)
    real_structure_summary = pcoa_spectrum_summary(real_distances)

    spectra = [resolved_singular_values(operator.matrix) for operator in operators]
    rng = np.random.default_rng(args.seed)
    null_summaries = []
    null_structure_summaries = []
    for repetition in range(args.repetitions):
        null_operators = [
            HeadOperator(
                layer=operator.layer,
                head=operator.head,
                kind=operator.kind,
                matrix=sample_spectrum_matched(spectrum, operator.matrix.shape, rng).astype(
                    operator.matrix.dtype
                ),
            )
            for operator, spectrum in zip(operators, spectra, strict=True)
        ]
        null_distances = normalized_frobenius_distances(null_operators)
        null_summaries.append(summarize_distance_matrix(null_distances))
        null_structure_summaries.append(pcoa_spectrum_summary(null_distances))
        del null_operators, null_distances
        print(f"completed null repetition {repetition + 1}/{args.repetitions}")

    result = {
        "metric": "normalized_frobenius",
        "null_model": "per-head-spectrum-matched-independent-random-directions",
        "repetitions": args.repetitions,
        "seed": args.seed,
        "source": str(args.input),
        "source_metadata": source_metadata,
        "real_summary": real_summary,
        "null_summaries": null_summaries,
        "empirical_tail_probabilities": empirical_tail_probabilities(
            real_summary, null_summaries
        ),
        "real_structure_summary": real_structure_summary,
        "null_structure_summaries": null_structure_summaries,
        "structure_empirical_tail_probabilities": empirical_tail_probabilities(
            real_structure_summary, null_structure_summaries
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved null audit to {args.output}")


if __name__ == "__main__":
    main()
