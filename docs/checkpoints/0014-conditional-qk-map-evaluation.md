# Checkpoint 0014: predictive test of class-conditional QK maps

## Question

Checkpoints 0012--0013 established an association between unsupervised
contextual input classes and a global QK channel's component mixture.  Does
that association mean that a head switches between separately learnable QK
maps?

## Fixed comparison

Fit query/key classes on discovery residual inputs, as before.  For each
populated query-class × key-class pair with at least 64 discovery events, fit
a rank-one bilinear QK map; use a shared rank-one map for sparse pairs.  This
conditional model has one rank-one factor for every fitted group plus the
fallback.

Compare its confirmation-set QK-margin R-squared to one global low-rank map
with the **same total number of rank-one factors**.  A global rank-four model
is also reported as a familiar compact reference.  All maps are fitted only
on discovery events; the test uses disjoint confirmation documents.

## Result

The class-conditional maps fail this predictive test:

| Model | Mean held-out R-squared |
| --- | ---: |
| class-conditional rank-one maps | -1.2201 |
| global equal-rank-budget map | 0.1437 |
| global rank-four map | 0.3047 |

Only 4 of 17 heads have a class-conditional map that beats its equal-budget
global control.  The poor average is not a close call; fitting a separate map
inside each coarse class pair substantially overfits the discovery data.

## Interpretation

This corrects the tempting interpretation of checkpoints 0012--0013.
Contextual residual states really are associated with the *usage* of a global
low-rank QK channel, including joint query/key dependence.  But the current
evidence does **not** support a hard routing-rule model in which each input
class pair owns a distinct operator.

The better working hypothesis is a smoothly varying global channel: input
state changes the continuous coordinates with which a common low-rank map is
used.  Any future gated model must beat the equal-description-length global
baseline on held-out margins before it is called a genuine conditional
operation.

## Reproduction

```powershell
python scripts/evaluate_conditional_qk_maps.py --iterations 200
```
