# Checkpoint 0003: exact compact operator pipeline

## Outcome

QK and OV operators can now be stored as their native skinny factors rather
than as dense residual-stream matrices. The factor bundles retain the model
location, operator kind, architecture metadata, checkpoint provenance, and
weight-processing convention.

The memory-safe safetensors reader supports GPT-2 and GPT-NeoX. It reproduces
TransformerLens' default interpretability transforms by folding the pre-attention
LayerNorm scale into Q/K/V, centering residual-stream readers, and centering O
writers. It reads one layer's required tensors at a time and does not construct
the complete language model.

## GPT-2 regression gate

The factor-native pipeline was compared with the frozen dense-operator distance
artifacts for all 144 GPT-2 small heads. Across every entry of both 144 by 144
distance matrices:

- QK maximum absolute error: `6.6170580126367895e-09`
- OV maximum absolute error: `1.3066516757476165e-09`
- entries above the `1e-6` acceptance tolerance: zero

The full numerical report is in
`results/gpt2-small/factorized_regression.json`.

## Resource behavior

The original TransformerLens compatibility loader exceeded the available
Windows commit allowance during this run. The replacement reader accesses
weights tensor-by-tensor. Pairwise distances use a temporary disk-backed
float32 cache and small float64 multiplication blocks; the scratch file is
removed automatically and dense matrices are never retained as artifacts.

## Interpretation boundary

For GPT-NeoX models, QK factors describe the content bilinear form before the
position-dependent rotary transform. Analyses using Pythia must label this as
the zero-relative-rotation view and separately test whether conclusions survive
explicit relative-position rotations.
