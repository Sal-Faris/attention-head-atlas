# Checkpoint 0022: low-rank typed composition channels survive causal controls

Checkpoint 0021 showed that whole-source-head ablations influence next-layer
Q/K/V readers in the pattern predicted by typed static edges.  This checkpoint
asks whether that relation remains when the intervention is restricted to a
specific **rank-four channel inside a source head**.

For each of the two strongest adjacent-layer static edges of each type Q, K,
and V, form the 64-by-64 coupling

\[
C_{a\to b}^{R}=W_O^a W_R^b.
\]

The source-side leading four-dimensional singular subspace of \(C\) is
removed from the source head's attention output immediately before its output
projection.  Seven Haar-random rank-four source-coordinate subspaces are
removed as matched controls for the same source head and prompt batch.  The
outcome is RMS change in the specified next-layer target reader, normalized
also by the fraction of source-output energy removed.

| Type | Selected edge(s) | Targeted / random mean raw effect | Targeted / random energy-normalized effect | Targeted rank among 8 subspaces |
| --- | --- | ---: | ---: | --- |
| Q | L2H6→L3H3, L2H6→L3H5 | 14.3x, 12.7x | 6.0x, 5.7x | 1/8, 1/8 |
| K | L2H7→L3H6, L2H7→L3H1 | 3.7x, 3.8x | 2.7x, 2.8x | 1/8, 1/8 |
| V | L4H7→L5H4, L1H1→L2H7 | 1.3x, 2.5x | 1.4x, 2.0x | 1/8, 1/8 |

The frozen selection comes from weights alone; outcomes use a separate set of
eight confirmation sequences.  Each intervention is naturally propagated
through the rest of the source transformer block before measuring the next
layer's raw Q/K/V preactivation.

**Conclusion.** The composition object is not only a whole-head association.
For these selected links, a small source-coordinate subspace obtained
unsupervised from \(W_O^aW_R^b\) has substantially more causal influence on
the intended reader than equally sized random subspaces.  This is the clearest
evidence yet for the project’s intended primitive:

\[
\text{source write channel} \longrightarrow \text{typed target read channel}.
\]

This is still a deliberately small confirmation set: six extreme static edges,
seven random controls each, and eight text sequences.  It establishes a sharp
candidate mechanism, not a population-wide effect size or semantic label.
The next test should pre-register a broader stratified set of edges and use a
new prompt split, then trace validated channels into attention changes and
logits.
