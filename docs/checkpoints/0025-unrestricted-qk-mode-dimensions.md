# Checkpoint 0025: reusable QK structure is medium-rank and multiscale

This checkpoint removes the rank-one motif assumption.  It asks whether the
dimensions of recurring QK transformations emerge when the transformations
are learned without an internal rank or block constraint.

Each of the 48 final-checkpoint Pythia-70M QK matrices is normalized to unit
Frobenius norm.  On one half of the heads, full 512-by-512 population modes are
learned by uncentered operator PCA.  No input support, output support, or rank
is selected before learning.  Only afterward is each unrestricted mode
singular-value decomposed.  The modes are tested on complete unseen heads and,
separately, on complete alternating layers.

## Weight-only held-out transfer

| Learned unrestricted modes | Unseen-head variance | Unseen-layer variance |
| ---: | ---: | ---: |
| 1 | 4.32% | 1.26% |
| 2 | 5.96% | 1.40% |
| 4 | **7.17%** | **1.65%** |
| 8 | 8.10% | 1.78% |
| 16 | 9.11% | 2.02% |

For the frozen four-mode comparison, independently truncating every learned
mode to rank 64 retains 6.42% for unseen heads and 1.54% for unseen layers:
89.5% and 93.2% of the corresponding unrestricted transfer.  Thus the
held-out-relevant part of these unrestricted transformations is concentrated,
but not rank one.

Nine end-to-end null repetitions give only 0.0015% for both the unrestricted
and rank-64 spectrum-matched-Haar comparisons.  Within-layer query/key
side-pairing shuffles give means of 0.21% and 0.21% in the unseen-head split,
and 0.037% and 0.038% in the unseen-layer split.  Every null repetition falls
below the observation (finite-null upper-tail p-value 0.10, the minimum at
this inexpensive run size).

## Dimensions that actually emerge

For unseen-head folds, the first four unrestricted modes require approximately
66, 103, 138, and 143 singular directions to retain 90% of their own energy.
Their entropy-effective ranks are approximately 71, 101, 129, and 149.  The
leading shared transformation is therefore more concentrated than later
population variation, producing a dimensional hierarchy rather than a single
compartment size.

The corresponding 90%-energy ranks when training on disjoint layer sets are
approximately 48, 70, 135, and 118.  There are no strong internal spectral
gaps: the largest adjacent singular-value ratios before 99% cumulative energy
are only about 1.07--1.29.  Consequently, the data do not reveal crisp natural
boundaries such as a canonical 16D or 64D compartment.  They reveal nested,
smoothly decaying subspaces.

The real modes are much lower-dimensional than independent spectrum-matched
Haar controls, whose top-four median 90%-energy rank is about 190--193.
However, within-layer side-pairing shuffles have median ranks near the real
ones.  Therefore the reduced dimensionality itself mostly comes from the
marginal query and key geometries.  Correct query-key pairing is what produces
the large held-out reconstruction advantage.

## Stability without assuming individual atoms

The leading unrestricted mode has matrix cosine 0.614 when learned from two
disjoint halves of the heads and 0.297 across disjoint layer sets.  For the
four-mode spans, the basis-invariant overlap fractions are 20.0% and 3.25%,
versus pairing-shuffle null means of 0.75% and 0.11%.  The first global mode is
the most reproducible; subsequent structure increasingly reflects layer-local
variation.  Comparing spans is important because individual PCA modes may
rotate within a real multidimensional family.

## Revised interpretation

This is stronger evidence for reusable weight structure than the constrained
rank-one experiment, but it argues against a simple discrete-compartment
picture.  QK weights contain a dominant, moderately low-dimensional global
transformation plus a hierarchy of progressively broader and more
layer-specific variations.  The structure is continuous and multiresolution.

The appropriate next decomposition is therefore not to force a fixed number
of blocks.  It is to follow the nested singular subspaces of the stable leading
mode and its stable population span, then measure their typed upstream and
downstream architectural connections.  Any proposed boundary should be
treated as a resolution parameter and retained only if it improves held-out
behavior or compression.
