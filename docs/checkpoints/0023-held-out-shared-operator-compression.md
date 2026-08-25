# Checkpoint 0023: shared structure predicts unseen heads, but only modestly

This checkpoint directly tests the central reuse question without requiring
discrete classes or named operator types:

> Can structure learned from some attention heads compress complete unseen
> heads better than matched unstructured operators?

Each final-checkpoint QK or OV operator is normalized to unit Frobenius norm.
Two strict folds hold out four of eight heads in every layer.  On the remaining
heads, the procedure learns shared input/read and output/write support bases,
projects operators into their shared core, and learns dense PCA patterns of
the core coupling.  Neither support nor coupling sees the held-out heads.

The frozen primary model uses a 64-dimensional shared support on each side and
16 core coupling components.  Two nulls are rerun end-to-end:

1. independent Haar rotations preserving every operator's singular spectrum;
2. within-layer shuffling of output sides, preserving the population of read
   and write subspaces while breaking their head-specific pairing.

| View | Shared-support energy in held-out heads | Full held-out variance recovered by 16 coupling components | Haar-null mean | Side-pairing-null mean |
| --- | ---: | ---: | ---: | ---: |
| QK | 9.91% | **3.93%** | -0.48% | -0.36% |
| OV | 5.76% | **0.63%** | -0.54% | -0.01% |

All 19 repetitions of both nulls fall below the corresponding observation
(finite-null upper-tail p-value 0.05, the minimum available at this run size).
The positive result is therefore consistent across two qualitatively different
controls, but the small null count and exploratory resolution grid should be
kept explicit.

At a 128-dimensional support, the bases capture 20.51% of held-out QK energy
and 14.62% of OV energy, while 16 shared core patterns recover only 5.03% and
0.56% of the complete operators.  Thus broad support geometry is more reusable
than the detailed coupling inside that support.

**Conclusion.** There is genuine reusable statistical structure: information
learned from half the heads predicts complete unseen heads better than
spectrum-matched matrices and better than operators with their side pairings
destroyed.  The effect is substantially stronger for QK than OV.  It is also
modest in absolute magnitude.  Most weight variation is head-specific at this
resolution, lies in finer conditional/compositional structure, or is not
compressible by a single shared linear-support-plus-PCA model.

This is a more defensible result than the earlier extreme-edge demonstrations:
it tests reuse out of sample rather than merely showing that a constructed
singular channel has high gain.  The appropriate next step, if resources
permit, is not another broad method sweep.  It is one refinement of the QK
case: replace dense core PCA with a multiresolution sparse or nonlinear latent
model and require improvement on the same frozen held-out-head/null benchmark.
