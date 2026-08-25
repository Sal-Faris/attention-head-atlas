# Checkpoint 0017: QK channel transfer does not emerge at a different rank

Checkpoint 0016 found no statistically supported functional transfer for
rank-four channels between the geometrically recurrent QK key-side pairs.  A
simple reusable motif could still have been obscured by an arbitrary rank-four
choice.  This pre-registered follow-up reuses the identical twelve edges,
events, native normalized-residual Q/K maps, source-layer alternative-donor
control, and 9,999 resamples at ranks 1, 2, 4, 8, and 16.

| Channel rank | Selected donor mean R-squared | Same-layer control mean | Upper-tail p-value |
| ---: | ---: | ---: | ---: |
| 1 | 0.0285 | 0.0288 | 0.4788 |
| 2 | 0.0344 | 0.0348 | 0.4922 |
| 4 | 0.0274 | 0.0263 | 0.4328 |
| 8 | 0.0353 | 0.0245 | 0.1567 |
| 16 | 0.0443 | 0.0294 | 0.0868 |

No rank crosses the uncorrected 0.05 threshold, and these five exploratory
tests would require a stronger multiple-comparison standard.  The modest
rank-16 tendency is a lead, not evidence: increasing channel complexity can
also make a transfer score less stable and raises the chance of an extreme
result among a small selected set of edges.

The result is informative nevertheless.  A reusable routing primitive should
not need a finely chosen rank to beat matched heads from the same layer.  The
absence of a low-rank result rules out the clearest flat-atom interpretation;
the absence through rank 16 makes full-map transfer a low-priority direction
unless a future conditional or compositional representation provides an
independent reason to revisit it.

The most sensible next target is therefore not a larger rank sweep.  It is a
typed composition or side-specific test, where a recurring key-side subspace
can participate in different query-to-key couplings rather than being asked to
serve as a complete portable head map.
