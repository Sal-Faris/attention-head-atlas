# Checkpoint 0016: recurrent QK channels do not yet transfer as portable maps

The previous audit found a small excess of recurring rank-four key-side QK
neighborhoods across layers.  That established a geometric recurrence, but not
whether a channel learned by one head is a reusable functional object.  This
checkpoint tests that stronger claim directly.

For each of the twelve recurrent cross-layer edges in the rank-four key-side
family audit, a rank-four joint QK bilinear channel was learned on the donor
head's discovery events.  The donor's processed Q and K factors and learned
channel were then applied to the recipient layer's normalized residual states,
with the recipient positions' RoPE rotation.  A linear calibration on the
recipient discovery events was used only to put the transferred score on the
recipient's actual QK-margin scale; held-out R-squared was evaluated on the
recipient confirmation events.

The essential control replaces the selected donor with every *other* head from
the same donor layer.  It preserves source and destination layer, recipient,
rank, residual coordinates, RoPE, and calibration procedure.  A resampling
null then selects one such alternative per edge.

Results:

- selected recurrent-donor mean transfer R-squared: **0.0334**;
- alternative same-layer donor mean R-squared: **0.0284**;
- resampled control mean: **0.0284**, standard deviation **0.01079**;
- upper-tail p-value for selected-donor mean: **0.3115**.

Two edges rank first among their seven exact source-layer alternatives, but
the population average does not beat the control.  The best individual
examples are L2H0 -> L3H4 (R-squared 0.1475) and L3H2 -> L4H5 (0.0747); they
are candidates for later targeted study, not evidence of a general mechanism.

**Conclusion.** The observed local cross-layer QK subspace recurrence does
not, at the current rank-four channel representation, yield a broadly portable
head-independent routing map.  Geometric proximity therefore cannot be
treated as a reusable-atom result.  This rules out an attractive but overly
strong interpretation and directs the next work toward either finer conditional
components or typed compositions, both of which can retain information that a
single globally fitted channel discards.

The result is deliberately a negative control result, not a claim that the
heads have no related role.  A relationship may instead be mediated by a
shared subspace, a conditional input regime, or downstream composition.

## Coordinate correction

An initial version accidentally applied factors that were folded into the
pre-layernorm residual coordinates to the stored *post*-layernorm residual
states.  The final calculation instead estimates each Q/K affine factor in
the artifact's normalized-residual coordinates and verifies it on the held-out
split.  The maximum relative Q/K reconstruction error is **2.31e-7**.  The
numbers above are exclusively from this corrected calculation; the initial
numbers are superseded.
