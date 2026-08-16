# Attention Head Atlas

An unsupervised study of the population geometry of transformer attention-head
operators.

The project asks whether trained attention heads exhibit stable, non-random,
functionally predictive organization. It does **not** assume in advance that
heads form discrete types: discrete clusters, continuous manifolds, sparse
operator atoms, and activation-dependent structure are competing hypotheses.

## First milestone

The first confirmatory experiment studies GPT-2 small OV operators

\[
M_{OV}^{\ell,h} = W_V^{\ell,h} W_O^{\ell,h}
\]

and asks whether their geometry contains more stable structure than null
operators matched on layer, rank, norm, and singular spectrum. Functional
labels are withheld during discovery and used only for validation.

See [`docs/analysis_contract.md`](docs/analysis_contract.md) for the frozen
initial design and criteria for interpreting results.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,models]"
```

Run the dependency-free mathematical tests with:

```powershell
python -m unittest discover -s tests -v
```

The model-facing pipeline will be added only after the extraction and null
generators pass their correctness tests.

Extract the pinned GPT-2 small OV operators with:

```powershell
python scripts/extract_operators.py `
  --model gpt2-small `
  --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e `
  --kind OV `
  --device cpu `
  --output artifacts/gpt2-small/ov_operators.npz
```

## Status

Research scaffold. No empirical claim has yet been made.
