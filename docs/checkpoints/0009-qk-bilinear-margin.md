# Checkpoint 0009: joint QK bilinear-margin compression

## Result

The joint low-rank QK model from the frozen protocol generalizes to a new,
disjoint 64-document confirmation corpus.  It models a matched QK preference
margin as

```text
q^T U V^T (k_plus - k_minus).
```

At rank 4, its mean per-head confirmation R-squared is 0.2789.  The strongest
equal-rank PCA control is key-side PCA, with mean R-squared -0.0879; query-side
PCA is -1.0116.  The joint model exceeds the best PCA control with a positive
document-bootstrap 95% interval for 35 of 48 heads.

Rank 2 is also robust (0.1562, 42 of 48 positive intervals).  Rank 8 increases
mean reconstruction to 0.3577 but only 27 heads retain a positive interval
over the best PCA control.  Rank 16 is less compelling as a compact primitive:
key-side PCA reaches 0.5399 and wins for many heads.

The rank-4 shuffled-pairing control reaches 0.1869, below the real-pair model's
0.2789.  Thus the advantage is not merely a generic low-rank optimizer; it
depends on the actual query--key pairings chosen by the head.

## Interpretation

This is evidence for a compact, prompt-distribution-weighted joint routing
structure.  A head's QK action is better represented by several paired
query--key channels than by an isolated query subspace.  It does not yet give
semantic labels to those channels, establish causality, or prove that the same
channels recur across heads.

The next checkpoint should map rank-2/4 bilinear factors into residual
coordinates and test their recurrence on the already validated QK families,
using exact-layer-pair controls.  No further prompt data are needed for that
test.

## Reproduction

```powershell
python scripts/collect_qk_conditional_events.py --confirmation-seed 2718 --artifact artifacts/qk_bilinear_margin_confirmation_v1.npz
python scripts/confirm_qk_bilinear_margin.py
```
