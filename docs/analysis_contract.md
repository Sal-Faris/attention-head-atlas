# Initial Analysis Contract

Status: **frozen before inspecting empirical clusters**  
Version: 0.1  
Date: 2026-08-16

Changes to this document must be committed with a rationale before the affected
analysis is run. Analyses invented after observing results are exploratory and
must be replicated on untouched evidence before they support a claim.

## Research question

Do trained attention-head OV operators exhibit stable, non-random population
structure that predicts held-out functional measurements beyond layer, norm,
rank, and singular spectrum?

## Competing hypotheses

- **H0 — matched random geometry:** apparent organization is explained by
  dimensionality, low rank, layer, norm, and singular spectrum.
- **H1 — discrete families:** operators form a small number of stable groups.
- **H2 — continuous geometry:** operators occupy a reproducible low-dimensional
  continuum without natural cluster boundaries.
- **H3 — reusable atoms:** heads are sparse mixtures of recurring sub-operators.
- **H4 — activation-conditional structure:** useful organization appears only
  after weighting operators by states visited by the model.

The study must not describe H2, H3, or H4 as failed clustering.

## First experimental population

- Model: GPT-2 small, using a pinned model revision recorded in run metadata.
- Units: all 144 attention heads.
- Primary object: `M_OV = W_V @ W_O`, separately for each layer and head.
- QK is a planned second study, not evidence for the first OV claim.

Extraction is accepted only after tests verify tensor orientation against the
model's direct head computation on random inputs.

## Discovery representations

### Primary

Frobenius-normalized OV matrices, compared using Frobenius distance. This is a
deliberately simple baseline measuring orientation in matrix space, not the
definitive functional metric.

### Prespecified secondary views

1. Normalized singular-value vectors.
2. Leading read-subspace projectors (right singular vectors).
3. Leading write-subspace projectors (left singular vectors).
4. Empirical action on held-out layer-appropriate residual activations.
5. Vocabulary/logit projection on a prespecified token subset.

UMAP and t-SNE are visualizations only and cannot establish structure.

## Null hierarchy

For every observed population statistic, compare against repeated draws from:

1. Gaussian matrices with matching dimensions and rank.
2. Per-head norm-and-rank-matched random matrices.
3. Per-head singular-value-matched Haar rotations.
4. Layer-stratified singular-value-matched rotations.
5. Random-initialization operators from the same architecture, when available.

The primary interpretation uses the strongest applicable null. Beating an
isotropic Gaussian null alone does not support operator families.

## Nuisance variables

Before naming any structure, quantify its association with:

- layer and normalized depth;
- Frobenius and spectral norm;
- algebraic and effective rank;
- leading singular-value fractions.

Functional prediction must be compared with a nuisance-only baseline. A family
adds evidence only if it improves held-out prediction beyond these variables.

## Separation of evidence

- **Development:** pipeline debugging and representation sanity checks.
- **Validation:** hyperparameter and model-selection decisions.
- **Confirmation:** untouched seed/checkpoint/model or prespecified held-out
  diagnostic evidence.

Functional diagnostics are unavailable to the discovery algorithm. Labels are
revealed only after representation and clustering choices are frozen.

## Required stability checks

- repeated clustering initializations;
- bootstrap resampling of heads;
- feature or projection subsampling;
- modest neighborhood/dimensionality choices;
- agreement across prespecified representations;
- consensus/co-clustering estimates rather than raw label equality.

An unstable group remains unnamed.

## Primary evidence categories

No single metric is sufficient. A strong finding requires:

1. **Excess structure:** a prespecified statistic exceeds the matched-null
   distribution.
2. **Stability:** neighborhoods, factors, or co-clustering relationships recur
   under resampling.
3. **Incremental prediction:** the discovered representation predicts held-out
   diagnostics beyond nuisance covariates.
4. **Causal specificity:** same-family replacements preserve a predicted
   behavior better than different-family and spectrum-matched replacements.

## Outcome language

- **Positive:** all applicable evidence categories support a specific form of
  organization.
- **Negative:** estimates are precise and consistent with the strongest null.
- **Inconclusive:** estimates are too unstable or underpowered to distinguish
  the hypotheses.

Cluster plots alone are never a positive result.

## Researcher-degrees-of-freedom log

Every run must save the Git commit, configuration, model revision, dependency
versions, random seeds, representation version, null generator, and complete
metrics. Failed and exploratory runs remain in the registry.

