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

Checkpoint 0005 tests what remains after sparse dictionary reconstruction and
separates exact query/key and read/write subspace geometry. Final-checkpoint
dictionary residuals are close to isotropic, while all tested factor subspaces
remain non-random. OV read and write geometries are only weakly coupled,
suggesting that reusable structure may live in separately recombined subspaces
rather than complete operator atoms.

Read the methods, results, caveats, and causal gate in
[`docs/checkpoints/0005-residuals-and-subspaces.md`](docs/checkpoints/0005-residuals-and-subspaces.md).
The preceding atom-discovery checkpoint remains in
[`docs/checkpoints/0004-pythia-factor-atoms.md`](docs/checkpoints/0004-pythia-factor-atoms.md).
The main Pythia figures are:

- `results/pythia-70m-deduped/dictionary_residual_null.png`
- `results/pythia-70m-deduped/factor_subspace_atlas.png`
- `results/pythia-70m-deduped/trajectory_geometry.png`
- `results/pythia-70m-deduped/atom_emergence.png`
- `results/pythia-70m-deduped/atom_reuse.png`
- `results/pythia-70m-deduped/rotary_qk_robustness.png`

The GPT-2 development figures remain:

- `results/gpt2-small/null_structure_comparison.png`
- `results/gpt2-small/functional_operator_atlas.png`
- `results/gpt2-small/representation_comparison.png`
- `results/gpt2-small/mixture_model_comparison.png`
- `results/gpt2-small/dictionary_stability.png`
