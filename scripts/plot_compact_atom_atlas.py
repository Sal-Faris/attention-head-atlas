"""Plot every compact QK, OV, and joint dictionary atom in shared operator bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from head_atlas.atoms import coordinate_atom_coefficients, materialize_operator_atoms
from head_atlas.factor_io import load_factor_bundle


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
        "--emergence",
        type=Path,
        default=Path("results/pythia-70m-deduped/compact_atom_emergence.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pythia-70m-deduped/compact_atom_atlas.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/pythia-70m-deduped/compact_atom_atlas.json"),
    )
    parser.add_argument("--projection-rank", type=int, default=48)
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


def materialize_atoms(
    coordinates: np.ndarray,
    atoms: np.ndarray,
    operators: list,
) -> np.ndarray:
    coefficients = coordinate_atom_coefficients(coordinates, atoms)
    return materialize_operator_atoms(coefficients, operators)


def shared_projection(
    atoms: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project atoms into common read/write bases ranked by total atom energy."""

    left_gram = np.einsum("aij,akj->ik", atoms, atoms, optimize=True)
    right_gram = np.einsum("aji,ajk->ik", atoms, atoms, optimize=True)
    left_values, left_vectors = np.linalg.eigh(left_gram)
    right_values, right_vectors = np.linalg.eigh(right_gram)
    left = left_vectors[:, np.argsort(left_values)[::-1][:rank]]
    right = right_vectors[:, np.argsort(right_values)[::-1][:rank]]
    projected = np.einsum("di,aij,jk->adk", left.T, atoms, right, optimize=True)
    fractions = np.sum(projected**2, axis=(1, 2)) / np.sum(atoms**2, axis=(1, 2))
    return projected, fractions, np.asarray([np.linalg.norm(atom) for atom in atoms])


def temporal_lookup(emergence: dict, view: str) -> dict[int, dict]:
    return {int(record["atom"]): record for record in emergence["views"][view]["atoms"]}


def plot_row(
    figure: plt.Figure,
    axes: np.ndarray,
    row: int,
    label: str,
    projected: np.ndarray,
    fractions: np.ndarray,
    norms: np.ndarray,
    temporal: dict[int, dict],
) -> None:
    limit = float(np.max(np.abs(projected)))
    images = []
    for atom, axis in enumerate(axes[row]):
        image = axis.imshow(
            projected[atom],
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        images.append(image)
        record = temporal[atom]
        rho = float(record["spearman_training_correlation"])
        direction = "↑" if rho > 0.75 else "↓" if rho < -0.75 else "→"
        axis.set_title(
            f"Atom {atom}  {direction}  ρ={rho:+.2f}\n"
            f"‖A‖F={norms[atom]:.2f} · shown {100 * fractions[atom]:.0f}%",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axes[row, 0].set_ylabel(label, fontsize=12, fontweight="bold")
    figure.colorbar(
        images[-1],
        ax=axes[row].tolist(),
        fraction=0.008,
        pad=0.008,
        label="operator entry",
    )


def main() -> None:
    args = parse_args()
    if args.projection_rank < 2:
        raise ValueError("projection rank must be at least two")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    emergence = json.loads(args.emergence.read_text(encoding="utf-8"))
    qk = load_dictionary(args.artifact_root / "qk_compact_dictionary.npz")
    ov = load_dictionary(args.artifact_root / "ov_compact_dictionary.npz")
    joint = load_dictionary(args.artifact_root / "joint_compact_dictionary.npz")
    qk_dimensions = qk["coordinates"].shape[1]

    qk_operators = load_population(manifest, "QK")
    qk_coefficients = coordinate_atom_coefficients(qk["coordinates"], qk["atoms"])
    joint_qk_coefficients = coordinate_atom_coefficients(
        joint["coordinates"][:, :qk_dimensions], joint["atoms"][:, :qk_dimensions]
    )
    qk_materialized = materialize_operator_atoms(
        np.concatenate([qk_coefficients, joint_qk_coefficients], axis=1), qk_operators
    )
    del qk_operators

    ov_operators = load_population(manifest, "OV")
    ov_coefficients = coordinate_atom_coefficients(ov["coordinates"], ov["atoms"])
    joint_ov_coefficients = coordinate_atom_coefficients(
        joint["coordinates"][:, qk_dimensions:], joint["atoms"][:, qk_dimensions:]
    )
    ov_materialized = materialize_operator_atoms(
        np.concatenate([ov_coefficients, joint_ov_coefficients], axis=1), ov_operators
    )
    del ov_operators

    atom_count = qk["atoms"].shape[0]
    groups = {
        "QK atoms": (qk_materialized[:atom_count], "QK"),
        "OV atoms": (ov_materialized[:atom_count], "OV"),
        "Joint atoms — QK half": (qk_materialized[atom_count:], "JOINT"),
        "Joint atoms — OV half": (ov_materialized[atom_count:], "JOINT"),
    }
    figure, axes = plt.subplots(
        len(groups),
        atom_count,
        figsize=(22, 11.5),
        constrained_layout=True,
        squeeze=False,
    )
    report_groups = {}
    for row, (label, (atoms, temporal_view)) in enumerate(groups.items()):
        projected, fractions, norms = shared_projection(atoms, args.projection_rank)
        plot_row(
            figure,
            axes,
            row,
            label,
            projected,
            fractions,
            norms,
            temporal_lookup(emergence, temporal_view),
        )
        report_groups[label] = {
            "frobenius_norms": norms.tolist(),
            "displayed_energy_fractions": fractions.tolist(),
            "shared_color_limit": float(np.max(np.abs(projected))),
        }
    figure.suptitle(
        "All compact Pythia-70m attention-head atoms\n"
        "Each row has one shared read/write basis and color scale; dictionary-atom sign is arbitrary",
        fontsize=16,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)

    report = {
        "analysis_status": "descriptive compact-atom atlas",
        "atom_count_per_dictionary": atom_count,
        "projection_rank": args.projection_rank,
        "projection": (
            "within each row, atoms are projected into shared left/right bases ranked by "
            "the total energy of all eight atoms"
        ),
        "sign_warning": "each dictionary atom is identifiable only up to a global sign flip",
        "groups": report_groups,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved compact atom atlas to {args.output}")


if __name__ == "__main__":
    main()
