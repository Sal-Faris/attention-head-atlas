# Checkpoint 0031: architectural coordinates reveal localization, not separable compartments

## Why this is a closer test of the actual hypothesis

The target hypothesis is that one OV map can contain several transformations
between restricted residual-stream subspaces,

\[
M_h \approx \sum_c U_{hc} A_{hc} V_{hc}^{\mathsf T}+R_h,
\]

without requiring every core \(A_{hc}\) to recur in other heads. The previous
pilot made the decomposition identifiable using population coordinates learned
from other OV heads. This checkpoint instead uses coordinates supplied by the
transformer's directed architecture. They are still discovered from weights
only and do not impose semantic operator types.

For an OV head in target layer \(\ell\), the read-side covariance is built only
from outputs of earlier anchor heads:

\[
G_{\mathrm{up}}^{(\ell)}=
\sum_{a:\,\operatorname{layer}(a)<\ell}
\frac{(M_{OV}^a)^{\mathsf T}M_{OV}^a}
{\operatorname{tr}((M_{OV}^a)^{\mathsf T}M_{OV}^a)}.
\]

These are directions the architecture can write before the target. The
write-side covariance is built only from Q, K, and V reader factors in later
anchor heads:

\[
G_{\mathrm{down}}^{(\ell)}=
\sum_{b:\,\operatorname{layer}(b)>\ell}
\left(
\frac{W_Q^b(W_Q^b)^{\mathsf T}}{\lVert W_Q^b\rVert_F^2}+
\frac{W_K^b(W_K^b)^{\mathsf T}}{\lVert W_K^b\rVert_F^2}+
\frac{W_V^b(W_V^b)^{\mathsf T}}{\lVert W_V^b\rVert_F^2}
\right).
\]

Every anchor contribution has unit trace, preventing large-norm heads from
dominating. The leading eigenspaces \(D_{\rm read}^{(\ell)}\) and
\(D_{\rm write}^{(\ell)}\) define

\[
C_h=(D_{\rm read}^{(\ell)})^{\mathsf T}M_h
D_{\rm write}^{(\ell)}.
\]

The target layer's matrices never enter either basis. Tests explicitly verify
that arbitrarily changing a target head leaves its bases unchanged.

## Pilot design

- Model: final Pythia-70M-deduped checkpoint.
- Targets: all 16 OV heads in middle layers 2 and 3.
- Primary architectural basis: 128 dimensions per side.
- Reciprocal anchor runs: all bases are constructed once from even heads and
  once from odd heads.
- Restricted maps: 4--64 dimensions per side, ranks 1, 2, 4, or 8, up to six
  blocks, with an explicit residual.
- Discovery: weights only; no tokens, prompts, activations, or semantic labels.
- Primary conditional description-length cap: 4,000 scalar equivalents. The
  architectural bases are side information computed from other model weights.
  Charging both bases conservatively would cost 7,160 scalars per target head,
  which is also reported.

The decision gate requires restricted blocks to beat:

1. singular-value-matched independent rotations;
2. arbitrary sparse coefficients in the same coordinates;
3. one dense low-rank transformation in the same projected space.

The complete gate must pass under both anchor parities.

## How much of the OV maps these coordinates expose

Coverage grows smoothly rather than revealing a tiny privileged architectural
subspace:

| Basis dimension | Even-anchor mean | Odd-anchor mean |
| ---: | ---: | ---: |
| 32 | 1.65% | 1.88% |
| 64 | 3.54% | 3.79% |
| 128 | 8.17% | 8.60% |
| 256 | 23.19% | 25.15% |
| 384 | 51.16% | 53.75% |

At dimension 128, layer 2 coverage is 9.21% and 12.25% for the two anchor
runs, while layer 3 coverage is 7.14% and 4.95%. Thus the primary result is a
test of a real but limited architecturally reachable/readable slice, not a
claim about all OV variance.

## Primary rate--distortion result

At the common 4,000-scalar cap:

| Anchor heads | Restricted blocks | Rotated blocks | Sparse coefficients | One dense map | Full SVD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Even | 5.18% | 4.83% | **5.87%** | **7.05%** | 21.44% |
| Odd | 5.60% | 5.15% | **6.18%** | **7.33%** | 21.44% |

Restricted blocks beat every one of 19 matched rotation populations in both
runs (`p=0.05`, the attainable minimum). The orientation supplied by real
upstream and downstream weights therefore contains reproducible localized
structure that the singular spectrum alone cannot explain.

The stronger compartment gate fails. Arbitrary coefficient sparsity and one
dense low-rank map use the common budget more efficiently in both runs.
Furthermore, 56.3% and 60.3% of selected blocks touch the largest permitted
64-dimensional support on at least one side. The typical model selects about
four blocks, but many are broad rather than cleanly isolated.

## Selected-cost audit

The block frontier does not always spend its full cap: its mean selected costs
are 1,306 and 1,435 scalars. A second audit gives each baseline exactly the
cost selected for that individual head.

| Anchor heads | Blocks | Cost-matched sparse | Cost-matched dense | Cost-matched full SVD |
| --- | ---: | ---: | ---: | ---: |
| Even | **5.18%** | 4.58% | **5.40%** | 5.04% |
| Odd | **5.60%** | 5.00% | **5.74%** | 7.69% |

Blocks beat cost-matched unstructured sparsity in all 16 heads in each anchor
run. They beat the dense projected map in only 7 of 16 heads in each run and
lose slightly on the population mean. This distinction resolves two questions:

- the result is more than isolated large coefficients;
- it is not evidence that several separable transformations describe the
  population better than one coupled low-rank transformation.

The common-budget comparison remains the primary MDL gate. The selected-cost
audit is a morphology diagnostic, not a replacement for it.

## Conclusion

This small real pilot finds a nontrivial intermediate result:

> Independently defined architectural coordinates make trained OV maps more
> block-localized than spectrum-matched rotations, and the selected blocks are
> more efficient than isolated sparse entries at their actual cost.

It does **not** yet establish the stronger claim that an OV head is best
described as several clean restricted transformations. A single dense
low-rank transformation remains equally good or better on average, and most
full OV energy lies outside the 128-dimensional architectural slice.

This rules out neither higher-resolution compartments nor compartments aligned
to MLP, embedding, or unembedding directions omitted from the pilot. It does
rule out declaring success from block-looking coefficients alone: a credible
compartment must beat a dense-map alternative and survive reciprocal external
coordinates.

## Highest-value next experiment

Before adding prompts or semantics, the next weight-only experiment should
separate a single coupled architectural map into statistically independent or
weakly interacting modules, rather than only rectangular supports. A useful
test is joint approximate block diagonalization of the target map together
with its upstream-producer and downstream-consumer Gram operators, with an MDL
penalty and synthetic calibration. This adds the missing criterion that
different proposed compartments are approximately closed under the relevant
family of transformations. If that also collapses to one block, activation-
conditioned structure becomes the justified next level.

## Reproduction

```powershell
python scripts/pilot_architectural_ov_restricted_maps.py
python scripts/audit_architectural_ov_selected_cost.py
```

Primary outputs:

- `results/pythia-70m-deduped/architectural_ov_restricted_map_pilot_v1.json`
- `results/pythia-70m-deduped/architectural_ov_restricted_map_pilot_v1.png`
- `results/pythia-70m-deduped/architectural_ov_selected_cost_audit_v1.json`
