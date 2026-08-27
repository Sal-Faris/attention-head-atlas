# Checkpoint 0029: OV singular-channel compartments do not survive gain removal

## Question

Checkpoint 0028 found that partitions of OV singular channels predict held-out
architectural connectivity, but singular gain alone recovers most of the
signal. Is there reproducible channel organization beyond this spectral
strength effect?

This is a gate for the broader restricted-transformation hypothesis. If the
same singular-channel representation cannot beat a spectrum-preserving null
after gain removal, it should not be promoted into a more expensive MDL model.

## Method

For every final Pythia-70M OV head, the prompt-independent architectural
fingerprint from checkpoint 0028 was recomputed. The explicit gain column was
removed. Within each head, a cubic quantile spline of log normalized singular
gain, with six knots, was fitted independently to every discovery and held-out
architectural feature. Clustering used only the residuals from these fits.

The remaining procedure was unchanged:

- exact paired OV singular read/write channels;
- reciprocal even-head and odd-head architectural anchor splits;
- PCA followed by a one-to-six-component diagonal Gaussian mixture;
- component count selected by BIC, with minimum component size two;
- held-out architectural R2 evaluated on the opposite anchor split.

Two controls were run:

1. 99 fitted-label permutations per reciprocal split;
2. 19 complete refits after independently permuting the write singular modes
   within every head, preserving all singular gains and marginal read/write
   directions while destroying their trained pairing.

The second control is the decisive one: a proposed restricted transformation
must depend on which read direction is actually connected to which write
direction.

## Population results

| Discovery anchors | Architecture-only R2 before removal | R2 after gain removal | Label-null mean | Re-paired mean | Multi-group heads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Even heads | 9.16% | **0.724%** | 0.431% | **1.870%** | 12/48 |
| Odd heads | 9.23% | **0.827%** | 0.465% | **2.080%** | 13/48 |

The residual partitions exceed the simple permuted-label controls in both
splits (`p=0.01`). This weak effect is not evidence for transformations,
because the spectrum-preserving re-pairing null is substantially stronger than
the trained model in every population repetition (`p=1.0` for an upper-tail
test in both splits).

Thus more than 90% of the previous held-out architectural score disappears
when a flexible within-head gain trend is removed. What remains is reproducible
relative to arbitrary labels but is not specific to the trained read/write
coupling.

This collapse is robust to the chosen spline flexibility. Repeating the
observed reciprocal fits with 3, 4, 5, 6, and 8 gain knots gives population
held-out R2 values from 0.48% to 0.88%; none approaches the 9.2% uncorrected
score. The full null hierarchy above was run at the preregistered six-knot
setting.

## Reciprocal and individual-head checks

Only seven heads select multiple groups in both reciprocal splits. Their mean
adjusted Rand agreement is 0.294, the median is 0.315, and none reaches 0.5.

An exploratory per-head comparison against the 19 re-pairing refits finds:

- two heads above all nulls in the even-anchor split;
- one head above all nulls in the odd-anchor split;
- no head above all nulls in both reciprocal splits.

With 48 simultaneous tests at the minimum attainable `p=0.05`, chance predicts
2.4 hits per split. The observed counts are therefore consistent with false
positives, and the lack of reciprocal survivors supplies no candidate for a
larger confirmatory run.

Residual scores are largest in middle layers, especially layer 3, but this
localization cannot be interpreted as trained compartment structure because
the stronger re-pairing null dominates the population result.

## Conclusion

This checkpoint rejects a specific version of the compartment hypothesis:

> Grouping the exact singular channels of an OV matrix by their architectural
> overlaps does not reveal trained read-transform-write compartments beyond
> smooth spectral-strength organization.

It does **not** reject the broader hypothesis that a weight matrix contains
variable-dimensional restricted transformations. The failed assumptions are
that the relevant elementary units are individual SVD channels and that a
Gaussian partition of channel fingerprints exposes their composition.

No structured-weight energy coverage is reported, because no proposed group
survives the decisive null. Counting the energy of compulsory partitions would
therefore be misleading.

## Decision for the next experiment

Do not build the full MDL model on these channel labels. The next method should
search directly over variable-rank restricted maps rather than partitioning a
preselected SVD basis. It must:

1. learn read and write supports jointly with the transformation core;
2. include an explicit unstructured residual;
3. charge for both support bases and the core, preventing arbitrary subspaces
   from being supplied for free;
4. allow clean bespoke blocks as well as cross-head reuse;
5. compare against SVD, spectral strata, and spectrum-matched rotations on
   held-out matrix entries or held-out architectural contractions.

This is now a better-justified target: the cheap spectral explanation has been
isolated and ruled out as an adequate answer to the original question.

## Reproduction

```powershell
python scripts/analyze_gain_residualized_ov_compartments.py
python scripts/audit_gain_residualization_sensitivity.py
```

Outputs:

- `results/pythia-70m-deduped/gain_residualized_ov_compartments_v1.json`
- `results/pythia-70m-deduped/gain_residualized_ov_compartments_v1.png`
- `results/pythia-70m-deduped/gain_residualization_sensitivity_v1.json`
