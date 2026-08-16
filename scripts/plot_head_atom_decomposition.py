"""Show coefficient-scaled dictionary atoms beside the head they reconstruct."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.atoms import coordinate_atom_coefficients, materialize_operator_atoms
from head_atlas.factor_io import load_factor_bundle
from head_atlas.factors import factorized_frobenius_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/pythia-70m-deduped-pilot.json"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/pythia-70m-deduped"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/head_atom_decomposition.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/pythia-70m-deduped/head_atom_decomposition.json"),
    )
    parser.add_argument(
        "--projection-data",
        type=Path,
        default=Path("results/pythia-70m-deduped/head_atom_decomposition_data.json"),
    )
    parser.add_argument("--projection-rank", type=int, default=32)
    return parser.parse_args()


def load_dictionary(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def load_population(manifest: dict, kind: str) -> list:
    operators = []
    for record in manifest["records"]:
        checkpoint_operators, _ = load_factor_bundle(record["factors"][kind]["path"])
        operators.extend(checkpoint_operators)
    return operators


def validate_alignment(artifact: dict[str, np.ndarray], operators: list) -> None:
    if len(operators) != len(artifact["coordinates"]):
        raise ValueError("operator population and dictionary coordinates differ in length")
    operator_layers = np.asarray([operator.layer for operator in operators])
    operator_heads = np.asarray([operator.head for operator in operators])
    if not np.array_equal(operator_layers, artifact["layers"]):
        raise ValueError("operator layers do not align with dictionary metadata")
    if not np.array_equal(operator_heads, artifact["heads"]):
        raise ValueError("operator heads do not align with dictionary metadata")


def materialize_dictionary_terms(
    artifact: dict[str, np.ndarray], operators: list
) -> tuple[np.ndarray, np.ndarray]:
    """Return the operator-space baseline and every dictionary atom."""

    directions = np.concatenate(
        [artifact["coordinate_mean"], artifact["atoms"]], axis=0
    )
    direction_weights = coordinate_atom_coefficients(
        artifact["coordinates"], directions
    )
    global_mean_weights = np.full(
        (len(operators), 1), 1.0 / len(operators), dtype=np.float64
    )
    matrices = materialize_operator_atoms(
        np.concatenate([global_mean_weights, direction_weights], axis=1),
        operators,
    )
    baseline = matrices[0] + matrices[1]
    atoms = matrices[2:]
    return baseline, atoms


def select_representative(artifact: dict[str, np.ndarray]) -> tuple[int, np.ndarray]:
    """Select the final-checkpoint head with the smallest compact-code residual."""

    final_step = int(np.max(artifact["checkpoint_values"]))
    final_indices = np.flatnonzero(artifact["checkpoint_values"] == final_step)
    centered = artifact["coordinates"][final_indices] - artifact["coordinate_mean"]
    reconstructions = artifact["codes"][final_indices] @ artifact["atoms"]
    errors = np.linalg.norm(centered - reconstructions, axis=1) / np.maximum(
        np.linalg.norm(centered, axis=1), 1e-12
    )
    best_local = int(np.argmin(errors))
    return int(final_indices[best_local]), errors


def singular_projection(matrix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left.T @ matrix @ right


def analyze_view(
    view: str,
    artifact: dict[str, np.ndarray],
    operators: list,
    projection_rank: int,
) -> dict:
    validate_alignment(artifact, operators)
    baseline, atoms = materialize_dictionary_terms(artifact, operators)
    sample_index, final_errors = select_representative(artifact)
    operator = operators[sample_index]
    actual = operator.materialize(dtype=np.float64) / factorized_frobenius_norm(operator)
    code = artifact["codes"][sample_index]
    active_atoms = np.flatnonzero(np.abs(code) > 1e-12)
    contributions = [code[atom] * atoms[atom] for atom in active_atoms]
    reconstruction = baseline + np.sum(contributions, axis=0)
    residual = actual - reconstruction

    centered_norm = np.linalg.norm(actual - baseline)
    matrix_relative_error = float(np.linalg.norm(residual) / max(centered_norm, 1e-12))
    coordinate_relative_error = float(
        np.linalg.norm(
            artifact["coordinates"][sample_index]
            - artifact["coordinate_mean"][0]
            - code @ artifact["atoms"]
        )
        / max(
            np.linalg.norm(
                artifact["coordinates"][sample_index]
                - artifact["coordinate_mean"][0]
            ),
            1e-12,
        )
    )
    if not np.isclose(matrix_relative_error, coordinate_relative_error, atol=2e-6):
        raise RuntimeError(
            "operator reconstruction does not match the exact PCoA reconstruction: "
            f"{matrix_relative_error} versus {coordinate_relative_error}"
        )

    left, _, right_t = np.linalg.svd(actual, full_matrices=False)
    rank = min(projection_rank, actual.shape[0])
    left = left[:, :rank]
    right = right_t[:rank].T
    terms = [baseline, *contributions, reconstruction, actual, residual]
    projected = [singular_projection(term, left, right) for term in terms]

    return {
        "view": view,
        "sample_index": sample_index,
        "checkpoint": int(artifact["checkpoint_values"][sample_index]),
        "layer": int(artifact["layers"][sample_index]),
        "head": int(artifact["heads"][sample_index]),
        "active_atoms": active_atoms.tolist(),
        "coefficients": [float(code[atom]) for atom in active_atoms],
        "matrix_relative_error": matrix_relative_error,
        "centered_energy_captured": float(1.0 - matrix_relative_error**2),
        "population_final_median_relative_error": float(np.median(final_errors)),
        "population_final_median_energy_captured": float(1.0 - np.median(final_errors) ** 2),
        "full_matrix_term_norms": [float(np.linalg.norm(term)) for term in terms],
        "projection_rank": rank,
        "projected_terms": projected,
    }


def panel_titles(record: dict) -> list[str]:
    active = record["active_atoms"]
    coefficients = record["coefficients"]
    contribution_titles = [
        f"{coefficient:+.3f} × atom {atom}"
        for atom, coefficient in zip(active, coefficients, strict=True)
    ]
    return [
        "Population baseline",
        *contribution_titles,
        "Dictionary reconstruction",
        f"Actual {record['view']} L{record['layer']}H{record['head']}",
        "Unexplained residual",
    ]


def plot(records: list[dict], output: Path) -> None:
    column_count = max(len(record["projected_terms"]) for record in records)
    figure, axes = plt.subplots(
        len(records),
        column_count,
        figsize=(3.05 * column_count, 3.45 * len(records)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, record in enumerate(records):
        projected = record["projected_terms"]
        titles = panel_titles(record)
        shared_limit = max(float(np.max(np.abs(matrix))) for matrix in projected[:-1])
        images = []
        for column, (matrix, title) in enumerate(zip(projected, titles, strict=True)):
            axis = axes[row, column]
            image = axis.imshow(
                matrix,
                cmap="coolwarm",
                vmin=-shared_limit,
                vmax=shared_limit,
                interpolation="nearest",
            )
            images.append(image)
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            norm = record["full_matrix_term_norms"][column]
            axis.set_xlabel(f"full ‖·‖F = {norm:.3f}", fontsize=9)
        for column in range(len(projected), column_count):
            axes[row, column].axis("off")
        captured = 100.0 * record["centered_energy_captured"]
        median = 100.0 * record["population_final_median_energy_captured"]
        axes[row, 0].set_ylabel(
            f"{record['view']} best compact example\n"
            f"{captured:.1f}% centered energy captured\n"
            f"final-head median: {median:.1f}%",
            fontsize=10,
        )
        figure.colorbar(
            images[-1],
            ax=axes[row, : len(projected)].tolist(),
            fraction=0.012,
            pad=0.012,
            label="operator entry in the head's singular-coordinate basis",
        )
    figure.suptitle(
        "What coefficient-scaled atoms contribute to a final Pythia head\n"
        "All terms share one color scale per row; matrices use the actual head's top singular modes",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def serializable_record(record: dict, include_projection: bool) -> dict:
    result = {key: value for key, value in record.items() if key != "projected_terms"}
    if include_projection:
        result["panel_titles"] = panel_titles(record)
        result["projected_terms"] = [
            np.round(matrix, 6).tolist() for matrix in record["projected_terms"]
        ]
    return result


def main() -> None:
    args = parse_args()
    if args.projection_rank < 2:
        raise ValueError("projection rank must be at least two")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = []
    for view in ("QK", "OV"):
        artifact = load_dictionary(
            args.artifact_root / f"{view.lower()}_compact_dictionary.npz"
        )
        operators = load_population(manifest, view)
        records.append(analyze_view(view, artifact, operators, args.projection_rank))
        del operators

    plot(records, args.output)
    report = {
        "analysis_status": "descriptive compact-dictionary decomposition",
        "selection_rule": (
            "lowest compact coordinate reconstruction error among final-checkpoint heads"
        ),
        "equation": "normalized head = population baseline + scaled atoms + residual",
        "projection": (
            "each displayed matrix is projected into the actual head's leading left/right "
            "singular-vector coordinates; addition remains exact inside the displayed block"
        ),
        "views": {record["view"]: serializable_record(record, False) for record in records},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    projection_data = {
        "analysis_status": "inline visualization data",
        "views": {record["view"]: serializable_record(record, True) for record in records},
    }
    args.projection_data.write_text(
        json.dumps(projection_data, separators=(",", ":")), encoding="utf-8"
    )
    print(f"saved decomposition figure to {args.output}")
    print(f"saved decomposition report to {args.report}")


if __name__ == "__main__":
    main()
