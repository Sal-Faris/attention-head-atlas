# Checkpoint 0028: OV channel organization is real but spectrally confounded

## Question

Can a single OV head be partitioned into multiple, variable-dimensional
read-transform-write modules without prompts or functional labels?

Checkpoint 0027 showed that intrinsic gain profiles are too cheap to identify
complete modules when arbitrary read/write frames are supplied for free. This
checkpoint therefore defines a compartment through its prompt-independent
architectural connections.

## Channel and fingerprint construction

Each final Pythia-70M OV operator is exactly decomposed into 64 paired singular
channels. Channel `i` reads its left singular direction, applies gain
`sigma_i`, and writes its paired right singular direction.

For every channel, the discovery fingerprint contains:

- normalized singular gain;
- overlap of its read direction with every earlier OV head's weighted write
  space;
- overlap of its write direction with every later head's Q, K, and V reader
  spaces.

No activations, tokens, prompts, or semantic labels enter the fingerprints.

Anchor heads are split by parity. A PCA plus diagonal Gaussian mixture is fit
only to the discovery-parity fingerprints, with one through six compartments
competing by BIC and a minimum size of two. The frozen channel assignments are
scored by the fraction of architectural fingerprint variation they separate
among the opposite-parity anchor heads. The entire procedure is then repeated
with anchor parities exchanged.

## Initial results

| Discovery anchors | Multi-compartment heads | Mean selected count | Held-out architectural R2 | Shuffled-label mean | Read/write re-pairing mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Even heads | 32/48 | 1.77 | **9.16%** | 1.22% | 4.55% |
| Odd heads | 28/48 | 1.71 | **9.08%** | 1.12% | 4.87% |

For both reciprocal splits, 99 independently shuffled fitted-label populations
fall below the real held-out score (`p=0.01`, the minimum available). Thus the
groups found using one set of architectural components predict connections to
components that were not used to discover them.

The stronger null independently permutes the write singular modes within every
head and reruns PCA, BIC selection, and mixture fitting. It preserves every
read mode, write mode, singular value, head, and layer while breaking which
read direction is transformed into which write direction. All 19 refitted
null populations fall below the real score in both reciprocal splits
(`p=0.05`). Correct read/write pairing accounts for roughly half of the
observed architectural coherence.

## Spectral confound audit

The initial result does not by itself establish compartments. Singular values
are ordered, and strong and weak channels could have systematically different
architectural connectivity without representing distinct transformations. We
therefore reran model selection with four discovery representations while
keeping the opposite-parity architectural fingerprint as confirmation.

| Discovery representation | Even-anchor held-out R2 | Odd-anchor held-out R2 |
| --- | ---: | ---: |
| Architecture plus normalized gain | 9.16% | 9.08% |
| Architecture only | **9.16%** | **9.23%** |
| Normalized gain only | 8.03% | 8.55% |
| Singular rank only | 0.00% | 0.00% |

Architecture-only and gain-only partitions both beat their separately
permuted-label controls (`p=0.01` in both reciprocal splits). Adding gain to
the architectural fingerprint provides no improvement. Gain alone recovers
most, but not all, of the predictive score.

The two partitions are related but not identical. Among the 18 heads assigned
multiple groups by both methods in each split, their mean adjusted Rand
agreement is 0.280 and 0.336. Architecture-only labels explain about 20% of
within-head normalized-gain variation. Thus spectral strength is a major
confound, but it does not completely determine the architectural grouping.

## Dimensions and reciprocal stability

The selected groups are not a disguised rank-one atomization. Across both
splits, multi-component heads produce 131 modules with median dimension 27,
range 2--61, and median operator-energy share 44.24%.

Twenty-five heads select multiple compartments in both reciprocal splits.
Their mean adjusted Rand agreement is 0.359 (median 0.344); nine have agreement
at least 0.5. Thirty-three of all 48 heads select the same compartment count in
both splits. This is moderate evidence for stable coarse organization, not a
canonical fine partition.

The effect is strongest in middle layers. Boundary layers have fewer available
upstream or downstream anchors, which is both an expected limitation and a
possible source of reduced power.

## Interpretation

The defensible positive result is narrower than the compartment hypothesis.
OV singular channels have non-random organization: partitions discovered from
one set of upstream writers and downstream readers predict relationships to an
independent set, and breaking read/write pairing roughly halves this
predictability. Architecture-only partitions are not identical to spectral
partitions, so the signal is not merely singular rank.

However, gain-only partitions recover most of the held-out architectural
score. The current groups are therefore best described as
*architecture-connected spectral strata*, not yet as distinct reusable
read-transform-write mechanisms. This is useful evidence about where
structure lives, but it is not the desired decomposition by itself.

The result also does not give a compartment explained-variance percentage: the
current model partitions all 64 channels, including weakly supported ones, and
the 9% score is held-out *architectural variation*, not matrix reconstruction.
BIC is an approximate description-length selector in fingerprint space, not
the final end-to-end weight MDL. Singular channels can also rotate unstably
inside near-degenerate spectral subspaces, so the partition must eventually be
tested under weight perturbations and subspace-preserving reformulations.

## Next gate

The next version should stop treating a compulsory partition of all singular
channels as the target object. It should fit variable-rank restricted maps
with an explicit background/residual component, penalize the read basis, core
transformation, write basis, and assignments jointly, and bootstrap whole
anchor heads. A proposed compartment should count toward structured weight
coverage only when it:

1. recurs across anchor bootstraps;
2. predicts held-out architectural fingerprints;
3. beats re-paired and rotated-support nulls;
4. yields a shorter full read/core/write code than bespoke SVD;
5. later predicts coherent activation and causal effects.

The spectral-only model must be included as a mandatory baseline. This will
convert the current organization test into the requested
reusable/bespoke/residual variance accounting without mistaking strength tiers
for mechanisms.

## Reproduction

```powershell
python scripts/analyze_architectural_ov_compartments.py
python scripts/analyze_architectural_ov_compartments.py `
  --discovery-parity 1 `
  --output results/pythia-70m-deduped/architectural_ov_compartments_parity1_v1.json `
  --figure results/pythia-70m-deduped/architectural_ov_compartments_parity1_v1.png
python scripts/summarize_architectural_ov_compartments.py
python scripts/audit_architectural_ov_compartment_confounders.py
```

Primary outputs:

- `results/pythia-70m-deduped/architectural_ov_compartments_v1.json`
- `results/pythia-70m-deduped/architectural_ov_compartments_v1.png`
- `results/pythia-70m-deduped/architectural_ov_compartments_parity1_v1.json`
- `results/pythia-70m-deduped/architectural_ov_compartments_summary_v1.json`
- `results/pythia-70m-deduped/architectural_ov_compartment_confounders_v1.json`
- `results/pythia-70m-deduped/architectural_ov_compartment_confounders_v1.png`
