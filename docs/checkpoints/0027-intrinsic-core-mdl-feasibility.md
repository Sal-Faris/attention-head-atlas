# Checkpoint 0027: learned core variation is real, but arbitrary frames dominate MDL

## Question

Before fitting variable OV compartments, this checkpoint audits a hidden
identifiability issue: if every candidate module can use unrelated head-specific
read and write bases, how much can reuse of the transformation core actually
save, and do those intrinsic cores contain learned population structure?

The analysis is weight-only and uses no functional labels, prompts, tokens, or
activations. The complete protocol is in
`docs/intrinsic_core_mdl_protocol.md`.

## Mathematical result

A rank-`k` operator in residual width `d` has `2dk-k^2` degrees of freedom.
Under independent changes of input and output bases, only its singular values
remain intrinsic. Sharing a normalized singular profile can therefore remove
at most `k-1` degrees of freedom.

For Pythia's `d=512` operators:

| Rank | Full rank-manifold dimensions | Maximum saved by a perfectly shared normalized core | Fraction |
| ---: | ---: | ---: | ---: |
| 16 | 16,128 | 15 | 0.0930% |
| 32 | 31,744 | 31 | 0.0977% |
| 64 | 61,440 | 63 | 0.1025% |

Thus a model with arbitrary head-specific frames cannot obtain a meaningful
full-weight MDL advantage merely by declaring the internal transformations
equivalent up to independent rotations. The locations of the read/write
subspaces dominate its description.

## Empirical spectral structure

Ordinary per-head SVD recovers 57.1% of QK and 54.0% of OV energy at rank 16;
rank 32 recovers 81.8% and 76.8%. This is substantial low-rank coverage but is
not reuse.

When the exact singular frames of each unseen head are supplied for free, the
training-population mean singular profile recovers 97.58% of QK and 96.79% of
OV energy. Those apparently spectacular values are not themselves evidence of
learned structure: initialization recovers 99.99%, as do Gaussian factor
products, because their normalized spectra are nearly identical.

The nontrivial signal is in the *variation around the mean*. Four spectral PCA
corrections learned from disjoint heads recover:

| View | Final trained profile variation | Initialization | Gaussian-factor null |
| --- | ---: | ---: | ---: |
| QK | 95.08% | 38.78% | 32.26% |
| OV | 96.97% | 32.72% | 32.26% |

The same four-dimensional spans recover 95.46% of QK and 97.26% of OV profile
variation when complete alternating layers are held out. Training therefore
creates a genuinely low-dimensional family of singular-profile changes that
transfers across heads and layers.

## What the learned variations look like

QK's first spectral mode explains 81.6% of total profile variation and is
strongly ordered by layer (Spearman `rho=-0.887`). It largely trades a strong
leading spectrum against a flatter bulk. Much of the simplest QK spectral
family is therefore developmental layer geometry.

OV is different. Its first three modes explain 65.2%, 24.5%, and 8.1% of
profile variation (97.8% jointly), while the first mode has only weak layer
association (`rho=-0.228`). The modes vary the strength of a leading spike,
the top several directions, and the slope of the remaining bulk. This is a
real cross-head pattern in how OV operators allocate gain, although it is not
yet a semantic mechanism or a compartment boundary.

## Why this does not solve compartment discovery

The conditional core scores hand every test head its exact singular frames.
In contrast, the existing shared-support model reconstructs only 3.93% of a
complete unseen QK operator and 0.63% of OV. The gap is the central result:
intrinsic gain profiles are simple, but locating the subspaces on which those
gains act is the hard and information-rich part.

This checkpoint neither rejects nor confirms bespoke compartments. It shows
that a proposed MDL model must charge for subspace location and cannot claim
large compression from gauge-equivalent cores alone.

## Revised next gate

The next compartment run should compare three honest frame codes:

1. fully bespoke singular frames;
2. population- or layer-reused ambient subspaces;
3. prompt-independent architectural anchors derived from upstream writers and
   downstream readers.

Only after frame cost is included should variable module counts, dimensions,
and reusable versus bespoke cores be selected. The learned OV spectral modes
provide a compact core code for that experiment, not the compartments by
themselves.

## Reproduction

```powershell
python scripts/audit_intrinsic_core_mdl.py
```

Outputs:

- `results/pythia-70m-deduped/intrinsic_core_mdl_audit_v1.json`
- `results/pythia-70m-deduped/intrinsic_core_mdl_audit_v1.png`
- `results/pythia-70m-deduped/intrinsic_core_profiles_v1.png`

