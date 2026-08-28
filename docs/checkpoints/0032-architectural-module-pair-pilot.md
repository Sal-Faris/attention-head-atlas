# Checkpoint 0032: reproducible coarse architectural modules, weak OV alignment

## Question

Checkpoint 0031 found that OV maps are localized in architectural coordinates,
but several fitted rectangular blocks did not outperform one dense low-rank
map. This checkpoint adds a stronger definition of a compartment: its
boundaries must be discovered from surrounding components without inspecting
the target OV matrix.

For each target layer, the read basis is still derived from earlier OV outputs
and the write basis from later Q/K/V readers. For basis direction \(u_i\) and
individual normalized architectural covariance \(G_a\), define its usage
profile

\[
p_i(a)=u_i^{\mathsf T}G_a u_i,
\qquad
\widehat p_i=\frac{p_i}{\lVert p_i\rVert_2}.
\]

Read and write axes are clustered separately by these profiles. Thus two axes
belong to the same module when the same upstream producers write them or the
same downstream consumers read them. No target-head coefficient is used to
choose a boundary.

For target coefficients \(C_h\), module-pair energy is

\[
E_{ab}^{(h)}=
\left\lVert
C_h[\mathcal R_a,\mathcal W_b]
\right\rVert_F^2.
\]

At resolution \(k\), the statistic is the energy in the strongest \(k\) of the
\(k^2\) possible read-module/write-module pairs, divided by total projected
energy. The selected pairs may be arbitrary: the method does not require an
identity matching or a predeclared transformation type.

## Design and controls

- Model: final Pythia-70M-deduped checkpoint.
- Targets: all 16 OV heads in layers 2 and 3.
- Architectural basis dimension: 128 per side.
- Resolutions: 2, 3, 4, 6, and 8 modules per side.
- Reciprocal constructions: even anchor heads and odd anchor heads.
- Discovery remains weight-only and prompt-independent.

Two separately refitted controls are used:

1. **Spectrum rotation:** preserve every nonzero projected singular value but
   independently rotate its read and write orientations.
2. **Matched partition shuffle:** retain the real target matrix and every
   discovered group size, but permute which architectural axes receive each
   membership.

Every null repetition is allowed to choose its best resolution exactly as the
real population does. An additional test permits selection over both layer and
resolution. All reported selection tests therefore include the relevant
look-elsewhere correction over the tested grid. There are 99 repetitions, so
the minimum attainable one-sided value is `p=0.01`.

## Are the externally discovered modules reproducible?

Yes, weakly. Module projectors from even and odd anchors are optimally matched
in ambient residual space. Their dimension-weighted overlap exceeds
group-size-matched permutations at every resolution:

| Modules per side | Real matched overlap | Permuted overlap | Advantage |
| ---: | ---: | ---: | ---: |
| 2 | 0.269 | 0.241 | 0.028 |
| 3 | 0.234 | 0.210 | 0.024 |
| 4 | 0.188 | 0.172 | 0.015 |
| 6 | 0.147 | 0.134 | 0.012 |
| 8 | 0.120 | 0.108 | 0.012 |

The maximum-over-resolution test gives `p=0.01`. Therefore the surrounding
weights contain some reusable producer/consumer organization. The effect is
small and strongest at the coarsest resolution; this is not evidence for a
sharp fine-grained modular basis.

## Do untouched OV heads route through a few module pairs?

Across both target layers, the best resolution is \(k=2\).

| Anchor heads | Real concentration | Rotated-null advantage | Rotation p | Partition-shuffle advantage | Shuffle p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Even | 0.809 | +0.016 | 0.01 | +0.004 | 0.86 |
| Odd | 0.838 | +0.057 | 0.01 | +0.040 | 0.01 |

The spectrum result is reciprocal, but the harder architectural-specificity
test is not. Even-anchor modules explain essentially no more concentration
than arbitrary assignments with the same dimensions.

The effect is also sharply layer-dependent:

| Anchors | Layer | Real k=2 | Rotation mean | Shuffle mean |
| --- | ---: | ---: | ---: | ---: |
| Even | 2 | 0.796 | 0.816 | 0.816 |
| Even | 3 | 0.823 | 0.767 | 0.791 |
| Odd | 2 | 0.899 | 0.940 | 0.939 |
| Odd | 3 | 0.776 | 0.625 | 0.657 |

When the test is allowed to choose both layer and resolution, both anchor
constructions select layer 3 at \(k=2\). Both beat spectrum rotations
(`p=0.01`). Odd anchors beat matched partition shuffles (`p=0.01`), while even
anchors remain suggestive but nonsignificant (`p=0.11`). This correction means
the layer-3 observation is not merely an uncorrected post-hoc comparison, but
it still fails the preregistered reciprocal gate.

At \(k\ge3\), real concentration is below both null families. Increasing
resolution does not uncover progressively cleaner compartments. The selected
layer-3 pairs contain only 6.13% and 3.95% of full OV energy because the
128-dimensional architectural projection itself is limited.

## Interpretation

This checkpoint separates two claims that earlier analyses conflated:

1. **Reusable architectural organization exists.** Directions grouped by
   their producer/consumer usage recur modestly across disjoint head subsets.
2. **OV heads decompose along those groups.** Evidence for this is coarse,
   layer-specific, and not reciprocal against the strongest control.

The first claim is a genuine new structural observation within this project.
The second remains a hypothesis. The data favor a small amount of continuous
or coarse architectural organization rather than a hierarchy of many crisp
independent compartments.

The result does not rule out head-private compartments. Such decompositions
remain non-identifiable from a single linear matrix unless another source of
information anchors their boundaries. It also does not test MLP, embedding,
unembedding, positional, or activation-conditional anchors.

## Best next direction

Hard clustering is now the weakest assumption in the pipeline. The next
weight-only model should replace discrete memberships by nonnegative or sparse
soft memberships learned from the full individual anchor covariances,
including off-diagonal coupling rather than only diagonal usage. The untouched
target then supplies a module-to-module transport matrix. Cross-fitting anchors
and matched rotations can remain exactly as in this checkpoint.

This tests whether the observed organization is a continuous overlapping
factorization—the most plausible interpretation after fine hard partitions
failed—while preserving target independence and allowing bespoke transport
cores.

## Reproduction

```powershell
python scripts/pilot_architectural_module_pairs.py
```

Outputs:

- `results/pythia-70m-deduped/architectural_module_pairs_v1.json`
- `results/pythia-70m-deduped/architectural_module_pairs_v1.png`
