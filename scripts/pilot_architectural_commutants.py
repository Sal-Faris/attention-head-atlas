"""Search architectural covariance families for nontrivial reducing subspaces."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.commutant import (
    commutator_energy,
    fit_approximate_commutant,
    projector_from_mode,
    random_traceless_mode,
    spectrum_rotated_covariances,
)
from head_atlas.factor_io import load_factor_bundle
from head_atlas.restricted_maps import (
    architectural_anchor_covariances,
    architectural_operator_bases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ov",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/step143000/ov_factors.npz"),
    )
    parser.add_argument(
        "--qk",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped/step143000/qk_factors.npz"),
    )
    parser.add_argument("--target-layers", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--mode-count", type=int, default=6)
    parser.add_argument("--null-repetitions", type=int, default=19)
    parser.add_argument("--random-projectors", type=int, default=99)
    parser.add_argument("--seed", type=int, default=16180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_commutants_v1.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/pythia-70m-deduped/architectural_commutants_v1.png"),
    )
    return parser.parse_args()


def project_family(
    covariances: tuple[np.ndarray, ...], basis: np.ndarray
) -> tuple[np.ndarray, ...]:
    return tuple(basis.T @ covariance @ basis for covariance in covariances)


def haar_projector(dimension: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, rank)), mode="reduced")
    return basis @ basis.T


def crossfit_projector(
    family: tuple[np.ndarray, ...],
    *,
    seed: int,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    directions = []
    learned_projectors = []
    for offset in (0, 1):
        training = family[offset::2]
        held_out = family[1 - offset :: 2]
        fit = fit_approximate_commutant(training, mode_count=1, seed=seed + offset)
        projector, rank, gap = projector_from_mode(fit.modes[0], minimum_rank=2)
        learned_projectors.append(projector)
        fitted_mode_energy = commutator_energy(fit.modes[0], held_out)
        fitted_energy = commutator_energy(projector, held_out)
        random_mode_energies = [
            commutator_energy(random_traceless_mode(len(projector), rng), held_out)
            for _ in range(repetitions)
        ]
        random_energies = [
            commutator_energy(haar_projector(len(projector), rank, rng), held_out)
            for _ in range(repetitions)
        ]
        directions.append(
            {
                "training_anchor_offset": offset,
                "training_anchor_count": len(training),
                "held_out_anchor_count": len(held_out),
                "projector_rank": rank,
                "mode_relative_eigengap": gap,
                "held_out_mode_commutator_energy": fitted_mode_energy,
                "random_mode_mean": float(np.mean(random_mode_energies)),
                "held_out_mode_energy_ratio_to_random": fitted_mode_energy
                / max(float(np.mean(random_mode_energies)), 1e-15),
                "mode_randomization_p_value": float(
                    (1 + np.sum(np.asarray(random_mode_energies) <= fitted_mode_energy))
                    / (repetitions + 1)
                ),
                "held_out_commutator_energy": fitted_energy,
                "size_matched_random_mean": float(np.mean(random_energies)),
                "held_out_energy_ratio_to_random": fitted_energy
                / max(float(np.mean(random_energies)), 1e-15),
                "randomization_p_value": float(
                    (1 + np.sum(np.asarray(random_energies) <= fitted_energy)) / (repetitions + 1)
                ),
            }
        )
    ranks = [round(np.trace(projector)) for projector in learned_projectors]
    projector_overlap = float(np.trace(learned_projectors[0] @ learned_projectors[1]) / min(ranks))
    random_overlaps = []
    for _ in range(repetitions):
        first = haar_projector(len(learned_projectors[0]), ranks[0], rng)
        second = haar_projector(len(learned_projectors[0]), ranks[1], rng)
        random_overlaps.append(float(np.trace(first @ second) / min(ranks)))
    return {
        "directions": directions,
        "mean_held_out_energy_ratio_to_random": float(
            np.mean([item["held_out_energy_ratio_to_random"] for item in directions])
        ),
        "mean_held_out_mode_energy_ratio_to_random": float(
            np.mean([item["held_out_mode_energy_ratio_to_random"] for item in directions])
        ),
        "both_mode_directions_p_at_most_0.05": all(
            item["mode_randomization_p_value"] <= 0.05 for item in directions
        ),
        "both_directions_p_at_most_0.05": all(
            item["randomization_p_value"] <= 0.05 for item in directions
        ),
        "split_projector_overlap": projector_overlap,
        "random_split_projector_overlap_mean": float(np.mean(random_overlaps)),
        "split_projector_overlap_upper_tail_p_value": float(
            (1 + np.sum(np.asarray(random_overlaps) >= projector_overlap)) / (repetitions + 1)
        ),
    }


def analyze_family(
    family: tuple[np.ndarray, ...],
    *,
    label: str,
    mode_count: int,
    null_repetitions: int,
    random_projectors: int,
    seed: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    fit = fit_approximate_commutant(family, mode_count=mode_count, seed=seed)
    projector, rank, gap = projector_from_mode(fit.modes[0], minimum_rank=2)
    null_lowest = []
    null_spectra = []
    for repetition in range(null_repetitions):
        null_fit = fit_approximate_commutant(
            spectrum_rotated_covariances(family, rng),
            mode_count=mode_count,
            seed=seed + 1_000 + repetition,
        )
        null_lowest.append(float(null_fit.eigenvalues[0]))
        null_spectra.append(null_fit.eigenvalues.tolist())
    lowest = float(fit.eigenvalues[0])
    return {
        "label": label,
        "anchor_count": len(family),
        "commutant_eigenvalues": fit.eigenvalues.tolist(),
        "lowest_nontrivial_eigenvalue": lowest,
        "spectrum_rotation_lowest_mean": float(np.mean(null_lowest)),
        "spectrum_rotation_lowest_samples": null_lowest,
        "spectrum_rotation_mean_spectrum": np.mean(null_spectra, axis=0).tolist(),
        "lower_tail_p_value": float(
            (1 + np.sum(np.asarray(null_lowest) <= lowest)) / (null_repetitions + 1)
        ),
        "full_family_projector": {
            "rank": rank,
            "complement_rank": len(projector) - rank,
            "relative_mode_eigengap": gap,
            "commutator_energy": commutator_energy(projector, family),
        },
        "crossfit": crossfit_projector(
            family,
            seed=seed + 20_000,
            repetitions=random_projectors,
            rng=rng,
        ),
    }


def synthetic_family(
    rng: np.random.Generator,
    *,
    dimension: int = 12,
    block_dimensions: tuple[int, ...] = (3, 4, 5),
    count: int = 12,
) -> tuple[np.ndarray, ...]:
    family = []
    boundaries = np.cumsum((0, *block_dimensions))
    for _ in range(count):
        covariance = np.zeros((dimension, dimension))
        for start, stop in pairwise(boundaries):
            values = rng.standard_normal((stop - start, stop - start))
            covariance[start:stop, start:stop] = values @ values.T
        family.append(covariance)
    return tuple(family)


def synthetic_calibration(
    rng: np.random.Generator,
    *,
    null_repetitions: int,
    seed: int,
) -> dict[str, object]:
    family = synthetic_family(rng)
    fit = fit_approximate_commutant(family, mode_count=4, seed=seed)
    null_values = []
    for repetition in range(null_repetitions):
        null = fit_approximate_commutant(
            spectrum_rotated_covariances(family, rng),
            mode_count=1,
            seed=seed + repetition + 1,
        )
        null_values.append(float(null.eigenvalues[0]))
    return {
        "definition": "three planted invariant blocks of dimensions 3, 4, and 5 with arbitrary dense PSD maps inside",
        "lowest_nontrivial_eigenvalues": fit.eigenvalues.tolist(),
        "spectrum_rotation_lowest_mean": float(np.mean(null_values)),
        "lower_tail_p_value": float(
            (1 + np.sum(np.asarray(null_values) <= fit.eigenvalues[0])) / (null_repetitions + 1)
        ),
    }


def population_null_test(records: list[dict[str, object]]) -> dict[str, float]:
    observed = float(np.mean([record["lowest_nontrivial_eigenvalue"] for record in records]))
    null = np.asarray([record["spectrum_rotation_lowest_samples"] for record in records])
    null_population = np.mean(null, axis=0)
    return {
        "observed_mean_lowest_eigenvalue": observed,
        "null_mean_lowest_eigenvalue": float(np.mean(null_population)),
        "observed_to_null_ratio": observed / max(float(np.mean(null_population)), 1e-15),
        "lower_tail_p_value": float(
            (1 + np.sum(null_population <= observed)) / (len(null_population) + 1)
        ),
    }


def plot_report(report: dict[str, object], output: Path) -> None:
    records = report["families"]
    labels = [record["label"] for record in records]
    x = np.arange(len(records))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    axes[0].bar(
        x - 0.2, [record["lowest_nontrivial_eigenvalue"] for record in records], 0.4, label="Real"
    )
    axes[0].bar(
        x + 0.2,
        [record["spectrum_rotation_lowest_mean"] for record in records],
        0.4,
        label="Rotated null",
    )
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("Lowest non-scalar commutant eigenvalue")
    axes[0].set_title("A  Do nontrivial commuting modes exist?", loc="left")
    axes[0].legend()

    axes[1].bar(
        x - 0.2,
        [record["crossfit"]["mean_held_out_mode_energy_ratio_to_random"] for record in records],
        0.4,
        label="Continuous mode",
    )
    axes[1].bar(
        x + 0.2,
        [record["crossfit"]["mean_held_out_energy_ratio_to_random"] for record in records],
        0.4,
        label="Thresholded projector",
    )
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Held-out energy / matched random")
    axes[1].set_title("B  Do fitted compartments generalize?", loc="left")
    axes[1].legend(fontsize=8)

    axes[2].bar(x, [record["full_family_projector"]["rank"] for record in records])
    axes[2].set_xticks(x, labels, rotation=35, ha="right")
    axes[2].set_ylabel("Selected projector rank (of 32)")
    axes[2].set_title("C  Candidate split dimensions (in-sample)", loc="left")

    for record in records:
        axes[3].plot(record["commutant_eigenvalues"], marker="o", alpha=0.75, label=record["label"])
    axes[3].set_xlabel("Non-scalar mode index")
    axes[3].set_ylabel("Commutant eigenvalue")
    axes[3].set_title("D  Low approximate-commutant spectrum", loc="left")
    axes[3].legend(fontsize=7, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ov, metadata = load_factor_bundle(args.ov)
    qk, _ = load_factor_bundle(args.qk)
    target_layers = tuple(sorted(set(args.target_layers)))
    rng = np.random.default_rng(args.seed)
    records = []
    for parity in (0, 1):
        for layer in target_layers:
            read_family, write_family = architectural_anchor_covariances(
                ov,
                qk,
                target_layer=layer,
                anchor_head_parity=parity,
            )
            read_basis, write_basis = architectural_operator_bases(
                ov,
                qk,
                target_layer=layer,
                anchor_head_parity=parity,
                dimension=args.dimension,
            )
            for side, family, basis in (
                ("read", read_family, read_basis),
                ("write", write_family, write_basis),
            ):
                label = f"p{parity}-L{layer}-{side}"
                records.append(
                    analyze_family(
                        project_family(family, basis),
                        label=label,
                        mode_count=args.mode_count,
                        null_repetitions=args.null_repetitions,
                        random_projectors=args.random_projectors,
                        seed=args.seed + 100 * parity + 10 * layer + (side == "write"),
                        rng=rng,
                    )
                )
                print(f"completed {label}", flush=True)
    population = {}
    for parity in (0, 1):
        for side in ("read", "write"):
            selected = [
                record
                for record in records
                if record["label"].startswith(f"p{parity}-") and record["label"].endswith(side)
            ]
            population[f"parity_{parity}_{side}"] = population_null_test(selected)
    stable_compartments = [
        record["label"]
        for record in records
        if record["crossfit"]["both_directions_p_at_most_0.05"]
        and record["crossfit"]["split_projector_overlap_upper_tail_p_value"] <= 0.05
    ]
    report = {
        "status": "approximate-commutant compartment-existence pilot",
        "model": metadata.get("model", "EleutherAI/pythia-70m-deduped"),
        "revision": metadata.get("revision", "step143000"),
        "discovery": "weights only; no target OV matrix, prompts, activations, labels, or transformation classes",
        "target_layers": list(target_layers),
        "architectural_dimension": args.dimension,
        "mathematical_test": (
            "lowest non-scalar eigenmodes of mean double-commutator L(X)="
            "mean_a [G_a,[G_a,X]] over normalized architectural covariances"
        ),
        "interpretation_rule": (
            "a low non-scalar eigenvalue indicates an operator whose spectral subspaces are "
            "approximately reducing compartments for every covariance in the family"
        ),
        "stable_compartment_gate": (
            "both reciprocal held-out projector tests have p<=0.05 and the two learned "
            "projectors overlap above size-matched random pairs at p<=0.05"
        ),
        "stable_compartment_families": stable_compartments,
        "synthetic_calibration": synthetic_calibration(
            rng, null_repetitions=args.null_repetitions, seed=args.seed + 50_000
        ),
        "families": records,
        "population_tests": population,
        "null_repetitions": args.null_repetitions,
        "random_projectors_per_crossfit": args.random_projectors,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_report(report, args.figure)
    print(f"saved result to {args.output}")
    print(f"saved figure to {args.figure}")


if __name__ == "__main__":
    main()
