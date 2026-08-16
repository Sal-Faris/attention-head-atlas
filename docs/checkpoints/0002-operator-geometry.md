# Checkpoint 0002: Operator Geometry and Functional Retrieval

Date: 2026-08-16  
Checkpoint tag: `checkpoint-0002`

## Result in one sentence

GPT-2-small attention operators contain reproducible population structure and
locally retrieve several published functional families, but the population is
better described by continuous directions and sparse mixtures than by a clean
global partition into head species.

## Scope and frozen baseline

- Population: all 144 heads of GPT-2 small.
- Static operators: normalized QK and OV residual-stream matrices.
- Primary distance: Frobenius distance after normalizing every operator to unit
  Frobenius norm. Sign and directional alignment are retained.
- Joint view: equal-weight Euclidean product of QK and OV distances.
- External benchmark: primary validated members of published GPT-2-small
  functional families. Unlabelled heads are unknown, not negative examples.
- The copy-suppression pair was inspected before the benchmark was frozen and
  is excluded from confirmatory permutation tests.

The full analysis rules and decision gates are in
`docs/research_protocol_v2.md`.

## Finding 1: structure exists, but global clusters are weak

Twenty null populations independently randomize every head's singular-vector
directions while preserving its complete resolved singular spectrum.

| View | Real effective dimension | Null mean | Real top-10 variance | Null mean |
| --- | ---: | ---: | ---: | ---: |
| QK | 72.72 | 142.97 | 25.25% | 7.18% |
| OV | 114.74 | 142.97 | 16.47% | 7.18% |

Thus, real heads share population directions that are absent from independent
random subspaces, especially in QK. This does not imply discrete clusters.
Average-linkage sweeps support that distinction:

- QK best silhouette: 0.0666 at 16 clusters, including 9 singletons.
- OV best silhouette: 0.0147 at 2 clusters.
- Joint best silhouette: 0.0342 at 15 clusters, including 7 singletons.

These values are too weak to treat a dendrogram cut as a discovered taxonomy.

## Finding 2: external functional families are locally recoverable

Published labels were applied only after the operator distances were fixed.
Family-balanced mean average precision (mAP) was compared with 9,999 random
label assignments, including a stricter null that preserves every labelled
head's layer.

| View | Uninspected-family mAP | Layer-null mean | Layer-stratified p |
| --- | ---: | ---: | ---: |
| QK | 0.3289 | 0.0642 | 0.0003 |
| OV | 0.3602 | 0.0675 | 0.0001 |
| Joint | 0.3602 | 0.0667 | 0.0001 |

The result survives removing each family in turn. When induction is removed,
the weakest leave-one-family-out result remains significant: QK p=0.0134, OV
p=0.0019, and joint p=0.0011 under layer-stratified permutations.

Recovery is heterogeneous rather than universal:

- The primary induction heads are mutual nearest neighbours in QK and OV.
- Name movers are strongest in the joint view.
- Previous-token heads are more recoverable in OV than QK.
- The primary duplicate-token pair is not recovered by these static metrics.

## Finding 3: layer and spectra do not fully explain retrieval

Distance-based variance partitioning finds a strong categorical layer effect
for QK (R²=0.1608, p=0.0001), a smaller joint effect (R²=0.1036, p=0.0001),
and no positive categorical layer separation for OV (R²=0.0496, p=1.0).

The entire implicit feature space was then linearly residualized against layer
and per-head spectral diagnostics before distances were recomputed. Retrieval
remains significant in every view:

| View | Raw mAP | Layer + spectrum residual mAP | Layer-stratified p |
| --- | ---: | ---: | ---: |
| QK | 0.3289 | 0.3318 | 0.0002 |
| OV | 0.3602 | 0.2050 | 0.0001 |
| Joint | 0.3602 | 0.1830 | 0.0001 |

Spectra explain a meaningful part of OV and joint similarity, but they are not
the whole signal.

## Finding 4: different operator parts carry different families

Projector distances compare leading singular subspaces without depending on a
particular choice of basis vectors inside each subspace. The rank comparison is
exploratory because benchmark labels were used to compare representations.

- QK singular values alone: mAP 0.082, layer-p=0.1239.
- QK rank-32 query/read subspace: mAP 0.393, layer-p=0.0001.
- OV singular values alone: mAP 0.218, layer-p=0.0143; this is driven by near-
  perfect name-mover retrieval.
- OV rank-16 write subspace: mAP 0.320, layer-p=0.0001.

No single invariant summary is sufficient. Singular values, input subspaces,
output subspaces, and the full aligned map retain complementary information.

## Finding 5: sparse mixtures outperform hard clusters

Exact PCoA coordinates preserve the complete normalized-Frobenius geometry.
In six-fold head-level cross-validation, representations learned on training
heads reconstruct held-out heads as either one centroid, a sparse weighted sum
of learned dictionary atoms, or dense PCA factors.

With 32 learned elements:

| View | Hard clusters | 2-atom mixture | 8-atom mixture | Dense PCA |
| --- | ---: | ---: | ---: | ---: |
| QK | 5.0% | 15.3% | 16.5% | 17.5% |
| OV | -8.6% | 5.2% | 6.6% | 7.6% |
| Joint | -4.5% | 7.5% | 8.9% | 10.0% |

Negative values mean the cluster centroid predicts a held-out operator worse
than the training mean. Sparse mixtures approach the dense linear upper
baseline with only a few active atoms. This supports a motif-mixture model over
mutually exclusive global head types, but atom stability and causal meaning
still need validation.

## Figures

- `results/gpt2-small/null_structure_comparison.png`
- `results/gpt2-small/functional_operator_atlas.png`
- `results/gpt2-small/representation_comparison.png`
- `results/gpt2-small/mixture_model_comparison.png`

## Claims deliberately not made

- The static geometry is not claimed to fully determine behavior on prompts.
- The external labels are incomplete and task-derived; unlabelled heads may be
  undiscovered positives.
- Projector rank and representation comparisons are exploratory and require a
  second model or held-out benchmark for confirmation.
- Sparse atoms are not yet named mechanisms. Reconstruction alone does not
  establish causal interchangeability.
- GPT-2 small is now a development population, not an untouched replication.

## Next confirmation gate

1. Measure dictionary-atom stability over folds, seeds, and nearby atom counts.
2. Project atoms back to operator space and identify heads with high loadings.
3. Test whether atom coefficients predict external families without fitting
   representation choices to those labels.
4. Add activation-weighted and causal replacement tests.
5. Replicate frozen choices on another model or GPT-2 checkpoint.
