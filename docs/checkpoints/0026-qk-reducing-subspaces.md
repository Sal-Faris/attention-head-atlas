# Checkpoint 0026: fixed QK compartments do not survive a smooth-trajectory null

This checkpoint tests the most assumption-light version of the proposed
within-head compartment hypothesis.  It asks whether a head's QK matrices
across training admit fixed input and output subspaces in which two completely
unrestricted transformations evolve with little cross-block leakage.  The
transformations inside the blocks are not assumed to be rank one, diagonal,
copy-like, mutually similar, or functionally named.

For each unit-Frobenius operator trajectory, the estimator learns projectors
`P` and `Q` from training checkpoints only and approximates

    A_t ~= P A_t Q + (I-P) A_t (I-Q).

The active input and output supports are also learned from training only.  The
frozen primary resolution is a 64-dimensional active support divided into two
32-dimensional sides.  A 12-configuration grid spanning support dimensions
32, 64, and 96 is used as a resolution audit.

## Design and correctness

The experiment uses all 48 processed QK heads in Pythia-70M-deduped at eight
checkpoints.  The main forward split trains through step4000, validates at
step16000, and confirms at steps64000 and 143000.  A late sensitivity split
trains at steps1000, 4000, and 16000, validates at step64000, and confirms at
step143000.

Synthetic tests verify recovery of fixed blocks containing independently
changing dense maps, rejection on a dense random family, independent
input/output gauge invariance, exclusion of inactive kernels, fixed-seed
determinism, and exact equality between dense and compact-factorized
calculations.  The final implementation computes supports and projected blocks
directly from exact compact singular factors, avoiding dense 512-dimensional
Gram calculations without changing the result.

The primary null hierarchy contains 19 end-to-end repetitions each:

1. independent spectrum-Haar trajectories;
2. within-layer query/key side-trajectory pairing with a constant donor across
   checkpoints;
3. smooth singular-frame trajectories preserving every checkpoint spectrum
   and every exact adjacent-checkpoint left- and right-frame overlap matrix.

Every null relearns its supports and reducing projectors.  The smallest
attainable add-one p-value is 0.05.

## Absolute held-out scores

| Split | Phase | Active-support energy | Block-diagonal/full energy | Within-support concentration | Gain over matched random projectors |
| --- | --- | ---: | ---: | ---: | ---: |
| Main | Validation | 13.94% | 11.41% | 79.93% | 29.94 points |
| Main | Confirmation | 5.75% | 3.93% | 64.47% | 14.46 points |
| Late | Validation | 26.70% | 22.78% | 84.25% | 34.24 points |
| Late | Confirmation | 19.76% | 16.01% | 79.10% | 29.11 points |

These numbers look highly structured in isolation.  They are not sufficient
evidence because a fitted split can exploit smooth low-rank motion.

## Matched-null results

The table reports the real minus null population mean at confirmation for the
frozen primary resolution.

| Null | Split | Active-support energy difference | Within-support gain difference | Upper-tail p |
| --- | --- | ---: | ---: | ---: |
| Independent spectrum-Haar | Main | +4.19 points | +14.46 points | 0.05 |
| Independent spectrum-Haar | Late | +18.19 points | +29.13 points | 0.05 |
| Within-layer side pairing | Main | +0.84 points | +12.00 points | 0.05 |
| Within-layer side pairing | Late | +2.19 points | +23.16 points | 0.05 |
| Smooth singular frames | Main | **-1.48 points** | **-2.04 points** | 1.00 |
| Smooth singular frames | Late | **-0.82 points** | **-0.75 points** | 1.00 |

The same pattern appears in complete block-diagonal energy.  Real trajectories
beat independent Haar and side-pairing nulls, showing temporal persistence and
nontrivial within-head query/key co-development.  They do not beat the strongest
null.  All 19 smooth-null population means exceed the real confirmation mean in
both forward splits.

Disjoint-checkpoint projector stability is 0.427 for the real heads, 0.064 for
independent Haar, 0.426 for side pairing, and 0.429 for smooth frames.  Thus the
apparently stable ambient partition is also explained by marginal temporal
coherence.  It does not provide independent evidence for fixed compartments.

An instructive transient occurs at main validation: real within-support gain is
0.32 points above the smooth-null mean (p=0.05), but real active-support reach is
1.73 points below it.  At later confirmation the gain reverses to 2.04 points
below the null.  A feature seen at one intermediate checkpoint is therefore not
a persistent compartment.

## Multiresolution audit

All 12 real-data resolutions have positive gain over dimension-matched random
projectors.  Population-wide validation selects support 32 split 16/16.  Its
main confirmation gain is 17.42 points, compared with 14.46 points for the
fixed 64/32/32 resolution, but it covers only 2.17% of full confirmation energy
instead of 5.75%.  Support 96 covers 12.43% and has 14.19 points of gain at its
48/48 split.  This is a smooth coverage-versus-concentration tradeoff, not a
natural compartment boundary.

The failed fixed-primary smooth-null gate is already decisive.  The selected
32-dimensional resolution is exploratory and was not substituted into the
confirmatory null test after seeing validation.

All 1,152 real grid fits reached the 60-iteration cap rather than the strict
projector-change tolerance.  This does not preferentially favor the real data,
because every null receives the identical optimizer and budget, but different
convergence rates could still affect a close real/null difference.  Two focused
audits therefore test iteration depth and local initialization.

At 60, 120, and 240 iterations, main-split real-minus-smooth confirmation gain
is respectively -2.19, -2.48, and -2.37 points.  Late-split differences are
-0.84, -0.68, and -0.73 points.  The sign is unchanged as the optimizer budget
quadruples.  At 120 iterations, expanding from one to four random starts (plus
the deterministic energy and mean starts) changes the main gap from -2.48 to
-2.37 points and the late gap from -0.68 to -0.67 points.  The negative result
is therefore not an iteration-cap or obvious local-start artifact.

## Head and layer heterogeneity

Only 8 of 48 heads on the main split and 12 of 48 on the late split have a
positive real-minus-mean-smooth-null confirmation gain.  Most layer means are
negative; the main split becomes less negative in later layers.  L3H5 is the
only head at the minimum per-head finite-null p-value of 0.05 in both splits,
with residual gains of about 2.48 and 2.72 points.  This does not survive a
48-head multiple-comparison claim and is recorded only as a candidate for an
independently designed follow-up.

## Scientific conclusion

The experiment rejects the population-level claim that isolated QK head
trajectories contain fixed orthogonal read/write compartments beyond what is
expected from their spectra and local smooth frame motion.  It does **not** say
that QK weights are unstructured.  Two weaker but real forms of structure
remain:

- trained trajectories persist far more than independent orientations;
- the matched query and key sides are much more reducible than sides borrowed
  from another head.

The key correction is that a high block score is not itself interpretable
structure.  Low rank and smooth rotation make fitted blocks look convincing.
The strongest null converts that visual impression into a negative result.

This test also does not exhaust the broader hypothesis.  It rules out fixed,
orthogonal, trajectory-stable compartments at the tested resolutions.  It does
not rule out co-moving subspaces, overlapping or oblique compartments,
architecturally anchored producer/consumer channels, conditional activation
regions, or reusable transformations that recur across different heads rather
than across one head's checkpoints.

## Best next move

The next section should replace fixed ambient projectors with typed,
architecturally anchored or co-moving coordinates.  In particular, compare
intrinsic transformations after transporting each head's active frames through
training, and anchor candidate read/write directions by the upstream components
that can produce them and downstream components that can read them.  The same
smooth-frame null should remain in the protocol.  This directly tests whether
the reusable object is a transformation in a moving channel rather than a
stationary slice of residual space.
