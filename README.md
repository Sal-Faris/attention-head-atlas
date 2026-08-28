# Attention Head Atlas

An unsupervised study of the population geometry of transformer attention-head
operators.

The project asks whether trained attention heads exhibit stable, non-random,
functionally predictive organization. It does **not** assume in advance that
heads form discrete types: discrete clusters, continuous manifolds, sparse
operator atoms, and activation-dependent structure are competing hypotheses.

## Research direction

The development experiment studies both GPT-2-small operator views

\[
M_{OV}^{\ell,h} = W_V^{\ell,h} W_O^{\ell,h}, \qquad
M_{QK}^{\ell,h} = W_Q^{\ell,h} (W_K^{\ell,h})^T
\]

and asks whether their geometry is better explained by discrete clusters,
continuous factors, or sparse mixtures. Published functional labels are
withheld during discovery and used as an incomplete external retrieval
benchmark.

See [`docs/research_protocol_v2.md`](docs/research_protocol_v2.md) for the
current hypotheses, nulls, validation rules, and decision gates. The original
OV-only contract remains in [`docs/analysis_contract.md`](docs/analysis_contract.md).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[analysis,dev,models]"
```

Run the dependency-free mathematical tests with:

```powershell
python -m unittest discover -s tests -v
```

Extract a pinned operator population with:

```powershell
# Replace OV with QK for the other operator view.
python scripts/extract_operators.py `
  --model gpt2-small `
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e `
  --kind OV `
  --device cpu `
  --output artifacts/gpt2-small/ov_operators.npz
```

## Status

Checkpoint 0032 discovers residual-axis modules solely from which earlier OV
heads write them and which later Q/K/V components read them. Coarse module
boundaries recur modestly across disjoint anchor sets, but untouched OV heads
align with them only at two-module resolution in layer 3, and one reciprocal
split misses the strongest partition-specificity gate. This is evidence for
weak reusable architectural organization, not yet for several crisp OV
compartments.

Read the methods, results, and revised gate in
[`docs/checkpoints/0032-architectural-module-pair-pilot.md`](docs/checkpoints/0032-architectural-module-pair-pilot.md).
The main Pythia figures are:

- `results/pythia-70m-deduped/activation_validation_pilot.png`
- `results/pythia-70m-deduped/recurrent_pair_behavior.png`
- `results/pythia-70m-deduped/subspace_family_audit.png`
- `results/pythia-70m-deduped/dictionary_residual_null.png`
- `results/pythia-70m-deduped/factor_subspace_atlas.png`
- `results/pythia-70m-deduped/trajectory_geometry.png`
- `results/pythia-70m-deduped/intrinsic_core_mdl_audit_v1.png`
- `results/pythia-70m-deduped/intrinsic_core_profiles_v1.png`
- `results/pythia-70m-deduped/architectural_ov_compartments_v1.png`
- `results/pythia-70m-deduped/architectural_ov_compartment_confounders_v1.png`
- `results/pythia-70m-deduped/gain_residualized_ov_compartments_v1.png`
- `results/pythia-70m-deduped/direct_ov_restricted_map_summary_v1.png`
- `results/pythia-70m-deduped/architectural_ov_restricted_map_pilot_v1.png`
- `results/pythia-70m-deduped/architectural_module_pairs_v1.png`
- `results/pythia-70m-deduped/atom_emergence.png`
- `results/pythia-70m-deduped/atom_reuse.png`
- `results/pythia-70m-deduped/rotary_qk_robustness.png`

The GPT-2 development figures remain:

- `results/gpt2-small/null_structure_comparison.png`
- `results/gpt2-small/functional_operator_atlas.png`
- `results/gpt2-small/representation_comparison.png`
- `results/gpt2-small/mixture_model_comparison.png`
- `results/gpt2-small/dictionary_stability.png`
