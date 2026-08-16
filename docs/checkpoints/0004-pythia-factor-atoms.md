# Checkpoint 0004: Pythia factor atoms across training

Date: 2026-08-16  
Model: `EleutherAI/pythia-70m-deduped`  
Checkpoints: 0, 64, 512, 1,000, 4,000, 16,000, 64,000, 143,000

## Result in one sentence

Pythia attention operators develop stable, sparse, continuously evolving
population structure that is poorly described by hard clusters, but current
controls do not support claiming a cross-layer periodic table of functional
head species: much of the strongest structure is shared developmental layer
geometry.

## Exact compact pipeline

Each head is retained as skinny factors, `M = L R^T`, rather than a dense
512-by-512 matrix. A safetensors reader reproduces TransformerLens' LayerNorm
folding and centering without constructing the language model. On GPT-2 small,
all entries of the frozen distance matrices were reproduced with maximum error
`6.62e-9` for QK and `1.31e-9` for OV.

The eight Pythia snapshots occupy 2.10 GB in the dedicated cache. Sixteen QK/OV
factor bundles occupy 0.174 GB. Each checkpoint contributes 48 heads, giving
384 observations per view. Model weights and factors remain outside Git; the
tracked manifest records immutable snapshot commits and factor SHA-256 hashes.

## Population geometry over training

At initialization, independently random heads are almost orthogonal: mean
pairwise QK and OV distance is approximately `sqrt(2)`. By step 143,000, the
means fall to 1.390 for QK and 1.400 for OV while their dispersion increases.
This is the signature expected when shared population directions emerge.

Individual trajectories remain highly identifiable. Across adjacent sampled
checkpoints, the matching `(layer, head)` is the nearest observation at the
next checkpoint for every QK transition and all but one OV transition. Even the
largest same-head changes remain far below the approximately 1.4 distance
between different heads.

## Sparse mixtures versus clusters

Discovery uses only checkpoints at step 1,000 or later. Six-fold validation
holds out complete head trajectories. Blocked-time validation separately holds
out contiguous checkpoint regions.

The reconstruction-optimal profile selects 32 atoms with 8 active per head:

| View | Trajectory error | Matched k-means | Dense PCA |
| --- | ---: | ---: | ---: |
| QK | 0.9353 | 1.0446 | 0.9294 |
| OV | 0.9619 | 1.1262 | 0.9579 |
| Joint shared-code | 0.9606 | 1.1208 | 0.9560 |

Values are held-out squared error relative to the training-mean baseline. The
sparse model beats hard clusters substantially but remains slightly behind
dense PCA, favoring sparse directions in a continuous space rather than clean
discrete species.

Materialization showed that 32 atoms can partially memorize individual heads.
A second, explicitly parsimonious profile therefore chooses the smallest
preregistered model that beats matched k-means: 8 atoms with 2 active in every
view. It gives errors 0.9498 (QK), 0.9764 (OV), and 0.9728 (joint), sacrificing
only 1–1.5 percentage points for a fourfold reduction in dictionary size and
per-head composition.

## Stability and temporal emergence

Twenty bootstraps resample complete head trajectories. All compact dictionaries
clear the frozen +0.10 margin over dimension-matched random dictionaries:

| View | Bootstrap cosine | Random | Advantage |
| --- | ---: | ---: | ---: |
| QK | 0.314 | 0.084 | +0.230 |
| OV | 0.288 | 0.083 | +0.205 |
| Joint | 0.273 | 0.058 | +0.215 |

After normalizing coefficient magnitude within each checkpoint, 6/8 QK, 6/8
OV, and 4/8 joint atoms have Benjamini-Hochberg-significant ordered time
trajectories. The heatmaps show redistribution from a few initialization-
aligned directions toward a broader trained dictionary, not the sudden birth
of sharply separated clusters.

## RoPE does not erase the late QK atlas

The base QK atlas is the zero-relative-rotation content view. Explicit GPT-NeoX
RoPE rotations were tested from 1 to 1,024 tokens. At the final checkpoint,
offset 1,024 retains 0.957 rank correlation with zero-offset distances and
83.3% of nearest-neighbour identities, with mean absolute distance change
0.0037. At step 1,000 the same values are 0.784, 39.6%, and 0.0084. Training
makes the static QK geometry more position-robust, although early-checkpoint QK
claims must remain offset-specific.

## The layer-confound diagnosis

The compact atoms initially looked reusable, but exemplar heads revealed
strong layer concentration. An explicit participation audit finds five of
eight compact atoms reusable across trajectories in each view, but only three
per view pass the stricter cross-layer rule.

An exploratory follow-up subtracts every checkpoint-by-layer centroid before
learning dictionaries. Layer centroids account for 16.6% of QK, 12.2% of OV,
and 14.4% of joint coordinate variance. Residual compact dictionaries still
beat hard clusters:

| View | Residual sparse error | Residual k-means | Residual PCA |
| --- | ---: | ---: | ---: |
| QK | 0.9726 | 1.0004 | 0.9710 |
| OV | 0.9820 | 1.0028 | 0.9790 |
| Joint | 0.9813 | 1.0029 | 0.9793 |

They also remain more stable than random. However, no residual atom currently
passes stability, FDR-controlled temporal change, and cross-layer reuse at the
same time. Residual QK has no significant time trajectories. This prevents the
strong periodic-table claim.

## Current interpretation

Supported:

- training produces non-random, continuous QK and OV population geometry;
- sparse mixtures generalize better than mutually exclusive hard clusters;
- compact atom directions are reproducible under trajectory bootstrap;
- developmental layer structure is a major, scientifically meaningful part of
  the atlas;
- OV retains a weak within-layer mixture signal after layer residualization.

Not supported yet:

- a universal set of cross-layer functional head species;
- semantic names for weight-only atoms;
- causal interchangeability of heads with similar coefficients;
- the claim that every atom in the 32-element reconstruction atlas is reusable.

## Conditional causal gate

Five raw compact candidates pass stability, temporal, trajectory-reuse, and
cross-layer-reuse filters. They are recorded with same-layer low-loading
controls in `results/pythia-70m-deduped/causal_validation_plan.json`. Because no
layer-residual atom passes every gate, these interventions are falsification
tests of provisional raw candidates, not confirmation. The complete protocol
is in `docs/causal_validation_protocol.md`.
