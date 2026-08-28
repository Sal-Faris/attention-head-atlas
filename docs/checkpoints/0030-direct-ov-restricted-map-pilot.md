# Checkpoint 0030: direct restricted-map pilot finds shared axes, not compartments

## The hypothesis tested

This checkpoint directly tests an identifiable weight-only version of the
restricted-transformation hypothesis:

\[
M_h \approx \sum_c U_{hc} A_{hc} V_{hc}^{\mathsf T} + R_h.
\]

Each proposed block has a variable-dimensional read support, a
variable-dimensional write support, an arbitrary learned low-rank core, and an
explicit residual. Heads may select different supports, dimensions, ranks, and
numbers of blocks. No copy, suppression, projection, rotation, or other
semantic operator type is imposed.

Completely arbitrary private subspaces cannot identify compartments from an
isolated linear map: any collection of blocks can be merged and SVD can
diagonalize the result. The pilot therefore supplies the minimum independent
structure needed for a weight-only test: read and write coordinates learned
from *different heads*. A block counts only if it compresses held-out heads in
those population coordinates.

## Direct model

For training operators, the read basis is learned from the leading eigenspace
of

\[
\sum_h M_hM_h^{\mathsf T},
\]

and the write basis from

\[
\sum_h M_h^{\mathsf T}M_h.
\]

For a held-out head,

\[
C_h=D_{\mathrm{read}}^{\mathsf T}M_hD_{\mathrm{write}}.
\]

The algorithm searches the energy geometry of the full coefficient matrix for
rectangular supports. Inside every support it learns the best low-rank core
directly from the residual, subtracts the selected transformation, and repeats
up to six times. Supports range from 4 to 64 dimensions per side and core ranks
from 1 to 8. This is not a partition of SVD channels.

The scalar-equivalent cost includes:

- the degrees of freedom of both population Stiefel bases, conservatively
  amortized only over held-out heads;
- combinatorial read/write support codes;
- arbitrary low-rank core parameters;
- a per-block overhead.

Rate--distortion curves compare explained Frobenius energy at fixed total
costs. The residual is always reported rather than silently discarded.

## Data split and controls

The real pilot uses all 48 final Pythia-70M OV heads in two reciprocal splits:

- learn population coordinates from 24 even-numbered heads and evaluate 24 odd
  heads;
- learn new coordinates from the odd heads and evaluate the even heads.

Discovery uses only weights. There are no prompts, activations, token classes,
or semantic labels.

The baselines are:

1. spectrum-matched independent rotations in the same projected space;
2. individually retained sparse coefficients in the same population basis;
3. one dense low-rank map in the same population basis;
4. full-space truncated SVD.

The first control asks whether trained orientation matters. The second asks
whether apparent blocks are merely generic axis sparsity. The third asks
whether splitting into compartments is better than one map.

## Calibration

Twelve synthetic matrices contain three planted transformations with variable
8- and 16-dimensional supports, ranks 2 and 3, plus small dense noise. At a
scalar-equivalent budget of 250, the restricted-map model recovers 90.0% of
operator energy, versus 19.8% after spectrum-matched rotation and 0% for the
allowed dense low-rank baseline. At budget 500 it recovers 99.1%, versus 44.1%
and 42.1%.

Thus the search and complexity accounting can detect the kind of block
structure being tested when it is present at this scale.

## Real result at dimension 128 and budget 8,000

| Training heads | Restricted blocks | Rotated-block null | Sparse coefficients | Rotated-sparse null | One dense map | Full SVD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Even | 7.23% | 6.23% | **8.92%** | 7.46% | **10.77%** | 34.19% |
| Odd | 6.10% | 5.02% | **7.44%** | 6.30% | **9.16%** | 33.62% |

Restricted maps beat all 19 spectrum-matched block-null populations in both
reciprocal splits (`p=0.05`, the minimum available). This establishes a small
but reproducible orientation-specific signal.

However, the block model fails the more important model-comparison gate:

- unstructured sparse coefficients explain more energy at the same cost;
- one dense projected low-rank map also explains more;
- 59.8% and 52.3% of selected blocks touch the maximum allowed 64-dimensional
  support on at least one side.

With 99 fresh null populations, unstructured shared-axis sparsity itself is
strongly non-random in both reciprocal splits (`p=0.01`). At budget 8,000 it
exceeds its rotated null by 1.45 and 1.14 percentage points. This simpler
effect accounts for the apparent block advantage.

## Resolution sensitivity

The population coordinates do not form a sharp compact common subspace.
Increasing their dimension captures held-out OV energy gradually:

| Dimension per side | Held-out energy | Basis cost per held-out head |
| ---: | ---: | ---: |
| 32 | 2.60% | 1,321 |
| 64 | 5.60% | 2,557 |
| 128 | 15.17% | 4,773 |
| 256 | 41.64% | 8,181 |
| 384 | 72.24% | 10,224 |
| 512 | 100.00% | 10,901 |

The dimension-64 pilot is indistinguishable from rotations. Dimension 128 is
the first resolution with a reciprocal orientation signal, but it remains
inferior to non-block baselines. A higher-dimensional run would buy coverage
primarily by transmitting most of the ambient basis, not by revealing a
compact periodic table.

## Conclusion

This pilot finds a real reusable feature of OV weights:

> Population-derived read/write axes make held-out heads modestly sparser than
> spectrum-matched rotations.

It does **not** find evidence that those coefficients assemble into compact
restricted transformations. At the tested resolutions and costs, the best
description is shared-axis sparsity plus head-specific low-rank structure, not
several clean compartments.

This is a direct test of variable-support, variable-rank transformations, not
the earlier SVD-channel proxy. What remains untested is a compartment whose
subspaces are meaningful only relative to architectural producers/consumers or
input distributions. That requires an independent anchor: otherwise a fully
private block decomposition of one matrix is non-identifiable up to arbitrary
basis rotations and merging.

## Next justified direction

The next model should preserve the direct block objective but replace
population covariance coordinates with independently meaningful architectural
coordinates:

- upstream component write spaces and embeddings on the read side;
- downstream Q/K/V and MLP readers plus unembedding directions on the write
  side;
- complementary components held out during discovery;
- private corrections charged at full cost.

That would test whether head-specific compartments exist relative to what can
actually produce their inputs and consume their outputs, while remaining
prompt-independent. Merely increasing the population-basis dimension is not
the priority.

## Reproduction

```powershell
python scripts/pilot_direct_ov_restricted_maps.py `
  --basis-dimension 128 `
  --null-repetitions 19 `
  --output results/pythia-70m-deduped/direct_ov_restricted_map_pilot_dim128_v1.json `
  --figure results/pythia-70m-deduped/direct_ov_restricted_map_pilot_dim128_v1.png

python scripts/pilot_direct_ov_restricted_maps.py `
  --basis-dimension 128 `
  --training-head-parity 1 `
  --null-repetitions 19 `
  --output results/pythia-70m-deduped/direct_ov_restricted_map_pilot_dim128_reciprocal_v1.json `
  --figure results/pythia-70m-deduped/direct_ov_restricted_map_pilot_dim128_reciprocal_v1.png

python scripts/audit_direct_ov_shared_axis_sparsity.py
python scripts/summarize_direct_ov_restricted_map_pilot.py
```

Primary outputs:

- `results/pythia-70m-deduped/direct_ov_restricted_map_summary_v1.json`
- `results/pythia-70m-deduped/direct_ov_restricted_map_summary_v1.png`
