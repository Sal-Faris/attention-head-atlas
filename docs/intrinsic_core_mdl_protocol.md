# Intrinsic-core MDL feasibility protocol

## Question

Suppose every attention operator is allowed its own input/read and output/write
subspaces. Can reuse of a transformation *inside* those subspaces materially
compress the complete operators?

This is a prerequisite for a variable-compartment MDL model. Without it, an
optimizer can call gauge-dependent cores reusable while charging no realistic
cost for locating their subspaces.

## Identifiability calculation

For a rank-`k` square operator of residual width `d`, the rank-manifold
dimension is

\[
    2dk-k^2.
\]

If independent orthogonal changes of input and output coordinates are allowed,
the normalized singular spectrum is the complete intrinsic invariant. Fixing
that spectrum removes only `k-1` dimensions because one overall scale remains
head-specific. Thus perfect reuse of intrinsic shape can save at most

\[
    \frac{k-1}{2dk-k^2}
\]

of a full rank-`k` parameter description before library overhead.

This is a manifold-dimension feasibility bound, not a claim of exact bitwise
MDL at a chosen quantization precision.

## Empirical audit

The exact singular spectra of all 48 processed Pythia-70M-deduped QK and OV
operators are measured at initialization and at step 143,000. Every spectrum
is normalized to unit L2 norm.

Two complete-observation parity splits hold out alternating heads or layers.
On the training observations, the mean spectrum and PCA corrections are
learned. Evaluation supplies each test operator's true singular frames for
free, so the resulting score measures only conditional core-profile
predictability:

\[
    1-\lVert \sigma_h-\widehat\sigma_h\rVert_2^2.
\]

The fraction of *between-head spectral variation* recovered by PCA corrections
is also reported relative to the held-out mean-spectrum error.

Controls are:

1. the actual initialization population;
2. twenty populations of normalized Gaussian factor products at matched
   residual width, factor rank, population size, and split;
3. a synthetic shared-profile versus heterogeneous-profile recovery gate.

Exact-spectrum Haar rotations are not a control for this particular score:
they preserve the object being measured by definition. Previous complete-
operator held-out reconstruction is included only as a clearly labeled
reference for the cost of locating frames.

## Decision rule

A high absolute profile score alone is not evidence of learned reuse. The
trained population must exhibit low-dimensional *variation beyond* the
initialization and Gaussian-factor controls. Even if it does, core reuse is
not a complete compartment result unless a reusable or cheaply anchored frame
code produces a material full-description saving.

