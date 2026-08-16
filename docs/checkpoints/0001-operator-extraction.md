# Checkpoint 0001: Reproducible Operator Extraction

Date: 2026-08-16  
Checkpoint tag: `checkpoint-0001`

## Working capabilities

- Construct row-vector-convention OV and QK operators.
- Extract every head operator from a TransformerLens model.
- Check every extracted operator against direct factored computation before
  saving it.
- Serialize operators with model, environment, and Git provenance.
- Generate rank/norm-matched Gaussian and exactly spectrum-matched rotation
  null operators.
- Compute raw, spectral, subspace, activation-action, and QK-score views.

## Canonical local artifact

- Model: GPT-2 small
- Hugging Face revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Operators: all 144 OV heads
- Shape: `(144, 768, 768)`
- Dtype: `float32`
- Maximum direct-computation relative error: `4.1347860246787604e-07`
- Manifest: `manifests/gpt2-small-ov.json`

The 316 MB matrix bundle is intentionally excluded from Git and is reproduced
with the command in the README. The tracked manifest records its checksum and
provenance.

## Validation

- 11 unit/integration tests pass.
- Ruff formatting and lint checks pass.
- Model extraction aborts if maximum relative action error exceeds `1e-5`.

## Claims not made at this checkpoint

This checkpoint establishes trustworthy data construction only. It does not
claim that OV operators cluster, that discovered geometry is non-random, or
that any grouping predicts function. Those questions begin in checkpoint 0002.

