# Factorized Atom Discovery Protocol

Date frozen: 2026-08-16  
Development model: GPT-2 small  
External pilot: EleutherAI Pythia-70M-deduped

## Objective

Discover reusable QK, OV, and paired QK–OV operator atoms without using
functional labels, then determine whether those atoms recur and evolve across
training checkpoints.

## Compact operator representation

Every operator is stored as two skinny factors:

\[
M = LR^T,
\]

with

\[
(L,R)=(W_Q,W_K)\quad\text{for QK}
\]

and

\[
(L,R)=(W_V,W_O^T)\quad\text{for OV}.
\]

Full residual-stream matrices are materialized only for verification or a
specific atom inspection. Model caches and factor bundles remain outside Git.

For GPT-NeoX/Pythia, the static QK operator is explicitly the content bilinear
form before rotary position transforms (equivalently, the zero-relative-rotation
view). Any QK claim must also be checked over a preregistered set of relative
position offsets. A conclusion present only at offset zero is reported as such
rather than generalized to effective attention scores.

## Correctness gate

Before external model acquisition:

1. Factorized actions must match direct TransformerLens head computation.
2. Factor-native inner products and normalized Frobenius distances must match
   explicit matrix calculations on synthetic cases.
3. The factor pipeline must reproduce the frozen GPT-2-small QK and OV
   distance matrices within a documented numerical tolerance.

Failure at any gate stops Pythia acquisition.

## Unsupervised discovery contract

Published functional head labels cannot influence:

- operator normalization;
- atom count;
- sparsity strength;
- checkpoint selection;
- atom retention;
- representation weighting.

Choices use only held-out reconstruction, description length, sparsity, and
stability. Labels and semantic prompt inspection begin only after the selected
dictionary and consensus atoms are frozen.

## Coupled dictionary

The primary paired model uses one sparse coefficient vector per head and
separate QK and OV atom matrices:

\[
\min_{C,D_Q,D_O}
w_Q\lVert X_Q-CD_Q\rVert_F^2
+w_O\lVert X_O-CD_O\rVert_F^2
+\lambda\lVert C\rVert_1.
\]

The weights are fixed to give normalized QK and OV views equal total influence.
Separate QK-only and OV-only dictionaries are required baselines.

## Checkpoint leakage controls

The same layer/head identity at nearby checkpoints is a correlated trajectory,
not an independent observation. Validation therefore uses grouped splits:

- hold out complete head trajectories when testing generalization to heads;
- hold out contiguous checkpoint regions when testing temporal generalization;
- never randomly distribute adjacent observations of one trajectory across
  training and test sets for a primary score.

## Pilot data

The checkpoint list is frozen in
`configs/pythia70m_deduped_pilot.json`. It covers initialization, early
training, middle training, and the final checkpoint. Expansion to denser
checkpoint sampling happens only after the pilot decision gate.

## Nulls and baselines

- step-0 randomly initialized model;
- per-head spectrum-matched random singular directions;
- checkpoint-label permutation after discovery;
- time-shuffled coefficient trajectories;
- hard clustering at matched dictionary size;
- dense PCA at matched component count;
- independently initialized and bootstrap-resampled dictionaries.

## Pilot success criteria

The pilot advances if all of the following hold:

1. Sparse dictionaries outperform hard centroids on grouped held-out
   reconstruction in at least one operator view.
2. Consensus-atom bootstrap similarity exceeds its matched random-dictionary
   baseline by at least 0.10 absolute cosine.
3. At least one consensus atom recurs across three adjacent sampled checkpoints
   or exhibits a stable monotonic coefficient trajectory exceeding a
   time-permutation null.
4. The result is not explained solely by singular spectra or operator norm.

Failure is a valid result. It triggers representation diagnosis rather than
automatic acquisition of more checkpoints.

## Storage discipline

- Use `D:/Laptop/AI/model-cache/huggingface` as the dedicated external-model
  cache.
- Keep model weights, factor bundles, and activation caches out of Git.
- Track manifests, checksums, compact JSON summaries, and figures in Git.
- Report disk usage before expansion beyond 50 GB.
- Avoid duplicate model caches and avoid saving full operator populations.
