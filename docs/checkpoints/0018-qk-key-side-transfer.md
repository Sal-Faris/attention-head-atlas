# Checkpoint 0018: recurrent QK key-side geometry is not a portable usable key representation

The static family audit found a small, controlled excess of recurrent rank-four
QK *key-side* neighborhoods across layers.  Whole-channel transfer then failed
(Checkpoints 0016--0017), but that could have happened because the same key
side participates in different query-to-key couplings.  This experiment tests
the narrower, more appropriate side-specific claim.

For each selected cross-layer edge, the recipient query is retained.  The donor
key map is applied to the recipient layer's normalized residual states.  Since
head coordinates are only defined up to a rotation, an orthogonal
donor-key-to-recipient-key alignment is learned on all discovery positions.
The alignment is truncated to ranks 1, 2, 4, 8, or 16, then predicts the
recipient's matched QK margins on held-out confirmation events.  A linear
calibration is fitted on discovery events only.  The exact control swaps in
each other head from the donor layer, preserving donor layer, recipient,
alignment procedure, rank, data, and calibration.

| Alignment rank | Selected donor mean R-squared | Same-layer control mean | Upper-tail p-value |
| ---: | ---: | ---: | ---: |
| 1 | 0.0194 | 0.0207 | 0.6196 |
| 2 | 0.0268 | 0.0254 | 0.4421 |
| 4 | 0.0285 | 0.0490 | 0.8717 |
| 8 | 0.0846 | 0.0733 | 0.2789 |
| 16 | 0.0946 | 0.0824 | 0.2669 |

No rank has a supported excess over its exact source-layer control.  In
particular, the rank-four representation corresponding to the original static
family selection is substantially below control.  The rank-8 and rank-16
tendencies are exploratory and non-significant across the sweep.

**Conclusion.** The recurrent static key-side geometry should not be described
as a portable key feature or reusable key atom.  At this resolution it is a
geometric relation, not a functionally interchangeable representation.  This
is a valuable constraint: the next structural search should prioritize typed
compositions and context-specific rules, where the object of recurrence can be
a relation among components rather than an isolated head side.

This result does not contradict the static audit.  Subspace proximity is an
intrinsic property of the matrices; it simply does not guarantee that applying
the maps to a later layer's reachable states gives the same usable feature.
