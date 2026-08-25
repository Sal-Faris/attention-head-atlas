# Checkpoint 0020: static typed composition edges predict held-out action

Checkpoint 0019 found that typed OV-writer→Q/K/V-reader edge identities are
highly stable across late training.  This checkpoint tests whether those static
edges correspond to what the mature model actually does on text.

On each held-out sequence, each source head's actual attention-weighted OV
output is calculated.  For every later target reader, this output is passed
through the target's processed Q, K, or V linear reader.  The behavioral edge
strength is the RMS norm of this resulting vector:

\[
A^R_{a\to b}=\sqrt{\mathbb{E}_{x,\,pos}
\lVert \operatorname{HeadResult}_a(x,pos)W_R^b\rVert_2^2}.
\]

The static score from Checkpoint 0019 is evaluated against this action score
within each exact ordered layer pair.  The null independently shuffles source
and target head identity inside each pair.  Thirty-two frozen documents are
used for discovery and a disjoint 32 for confirmation.  No edge was selected
using activations.

| Reader type | Static→action Spearman, discovery | Static→action Spearman, confirmation | Matched-null p-value | Discovery/confirmation action reliability |
| --- | ---: | ---: | ---: | ---: |
| Q | 0.552 | **0.554** | 0.001 | 0.987 |
| K | 0.524 | **0.528** | 0.001 | 0.996 |
| V | 0.433 | **0.439** | 0.001 | 0.993 |

This is strong held-out validation: after controlling for the source and
target layers, a weight-only score ranks connections by their real
activation-weighted linear reach into later readers.  It is especially notable
because the analogous isolated-head and key-side transfer tests were negative.

**What this establishes.** Typed writer→reader edges are a productive
weight-derived structural primitive.  They are stable over training and their
strength has a held-out correspondence to how much a source head can influence
later Q/K/V coordinates on a text distribution.

**What it does not establish.** This is a linearized reachability measure, not
a causal intervention.  It does not include all intervening residual updates,
layer-normalization nonlinearities, or feedback through later attention.
Strong edges are therefore candidate circuit links, not yet claims of necessity
or semantic function.  The next high-value experiment is to test whether a
small set of strong edges composes into predictable two-step paths, followed by
targeted ablation/patching of the most robust paths.
