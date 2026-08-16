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

Checkpoint 0002 finds strong non-random population directions and statistically
significant local recovery of published functional families, while global
cluster quality is weak. Cross-validated sparse mixtures reconstruct held-out
operators substantially better than hard cluster centroids.

Read the methods, numbers, caveats, and next confirmation gate in
[`docs/checkpoints/0002-operator-geometry.md`](docs/checkpoints/0002-operator-geometry.md).
The main generated figures are:

- `results/gpt2-small/null_structure_comparison.png`
- `results/gpt2-small/functional_operator_atlas.png`
- `results/gpt2-small/representation_comparison.png`
- `results/gpt2-small/mixture_model_comparison.png`
- `results/gpt2-small/dictionary_stability.png`
