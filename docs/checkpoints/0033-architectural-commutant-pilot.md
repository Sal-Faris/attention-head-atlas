# Checkpoint 0033: shared geometry without stable hard compartments

## Question

Can the surrounding weight architecture define residual-stream compartments
without choosing their number, dimensions, memberships, or internal
transformation type in advance?

This checkpoint deliberately postpones transformation reuse, clustering,
dictionary learning, token interpretation, and activation analysis. It asks
only whether stable read or write compartments exist in a small weight-only
pilot.

## Test

For a family of normalized symmetric architectural covariances (G_a), define

\[
\mathcal L(X)=\frac{1}{A}\sum_a [G_a,[G_a,X]],
\qquad [G_a,X]=G_aX-XG_a.
\]

Its quadratic form is

\[
\langle X,\mathcal L(X)\rangle
=\frac{1}{A}\sum_a\lVert[G_a,X]\rVert_F^2.
\]

The identity always commutes and is discarded. A low-energy non-scalar
symmetric mode (X) indicates common geometry. If the family has an exact
common reducing decomposition, a nontrivial orthogonal projector onto one of
its compartments commutes with every (G_a). Conversely, spectral projectors
of a commuting symmetric (X) provide such a decomposition. The test is
therefore agnostic to the dense transformation performed *inside* each block.

For approximate structure, the lowest non-scalar eigenmode of
(\mathcal L) is fitted. Its largest admissible eigengap proposes a hard
two-sided split; the smaller side is reported so that a compartment and its
complement are not double-counted.

## Data and controls

- Model: final Pythia-70M-deduped checkpoint.
- Target layers: 2 and 3.
- Architectural dimension: 32.
- Reciprocal populations: even and odd anchor heads.
- Read families: individual earlier-OV output covariances.
- Write families: individual later-Q/K/V reader covariances.
- The target OV matrix, prompts, activations, labels, and predefined operator
  classes are not used.

Three validation levels are kept separate:

1. **In-sample geometry.** Compare the lowest real commutant eigenvalue with
   independently spectrum-rotated covariance families.
2. **Held-out action.** Fit on alternating anchors and test the continuous
   mode and its thresholded projector on the opposite anchors against matched
   random modes/projectors.
3. **Boundary recurrence.** Compare the two split-fitted projectors with
   Haar-random projector pairs of identical ranks.

A stable compartment is declared only when the hard projector beats random
projectors in both held-out directions *and* the two independently learned
projectors overlap above the matched random distribution, all at
`p <= 0.05`.

There are 19 spectrum rotations (`p_min=0.05`) and 99 random comparisons per
cross-fit statistic (`p_min=0.01`).

## Calibration

The method was first tested on 12-dimensional synthetic covariance families
with three planted invariant blocks of dimensions 3, 4, and 5. The maps within
each block were arbitrary dense positive-semidefinite matrices.

The two expected non-scalar block-indicator modes were recovered at numerical
zero (`7.1e-17` and `1.3e-16`), while the independently rotated null's lowest
mode averaged `0.0263` (`p=0.05`). Thus the method detects compartments without
requiring simple or repeated transformations inside them.

## Real-weight results

At the population level, all four reciprocal read/write comparisons have
lower commutant energy than every matched spectrum-rotation draw:

| Population | Real / rotated-null energy | p |
| --- | ---: | ---: |
| Even anchors, read | 0.860 | 0.05 |
| Even anchors, write | 0.718 | 0.05 |
| Odd anchors, read | 0.772 | 0.05 |
| Odd anchors, write | 0.747 | 0.05 |

This is evidence that the architectural covariance families have more shared
orientation than independent matrices with the same spectra. It is not yet a
compartment result.

The full-family eigengaps propose small candidate sides of dimensions 2, 4,
3, 3, 2, 6, 2, and 4 across the eight configurations. These dimensions are
descriptive in-sample outputs, not established modules.

The stricter cross-fit gives the decisive result:

- Two families have continuous modes that beat random modes in both
  directions: odd-layer-3 read and odd-layer-3 write.
- Two families have thresholded projectors that beat random projectors in
  both directions: even-layer-2 write and even-layer-3 write.
- Those two hard-projector candidates have split-projector overlaps of only
  `0.03` and `0.01`, respectively, below their matched random means near
  `0.10` (`p=0.98` and `p=1.00`).
- Three other families have above-random split overlap (`p=0.02`), but their
  projectors fail reciprocal held-out generalization.
- Therefore **zero of eight families passes the preregistered stable-hard-
  compartment gate**.

## Conclusion

This pilot finds nontrivial shared architectural geometry, but it does not
find a stable hard compartment decomposition in the tested 32-dimensional
read/write slices. The distinction matters: low commutator energy says several
operators prefer related orientations; it does not imply that one crisp
boundary consistently separates them.

This is a useful negative result rather than evidence that the weights are
unstructured. It rules out the strongest version of the local hypothesis at
this resolution: a few shared, sharply bounded architectural subspaces that
are recoverable from either half of the anchor population.

It does **not** rule out:

- overlapping or soft compartments;
- head-private compartments;
- compartments stable across a single head's training trajectory rather than
  across different surrounding heads;
- structure outside the top 32 architectural dimensions;
- more than one low commutant mode jointly defining a block algebra;
- input-conditional structure, tokens, activations, MLPs, or paths.

## Best next test

The closest test of the user's original single-head modularity hypothesis is a
trajectory cross-fit. For each head, use its OV operators across checkpoints
to form read families (M_tM_t^{\mathsf T}) and write families
(M_t^{\mathsf T}M_t). Fit common reducing subspaces on alternating
checkpoints, test them on held-out checkpoints, and compare with independently
rotated trajectory nulls. This supplies multiple observations of the *same
head*, so stable head-private compartments become identifiable without
requiring global sharing across heads.

Only after a compartment survives that gate should its restricted internal
maps be tested for low description length, reuse across heads, token meaning,
activation usage, and causal effect.

## Reproduction

```powershell
python scripts/pilot_architectural_commutants.py
```

Outputs:

- `results/pythia-70m-deduped/architectural_commutants_v1.json`
- `results/pythia-70m-deduped/architectural_commutants_v1.png`
