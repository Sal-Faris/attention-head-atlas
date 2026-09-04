# Checkpoint 0036: weight-discovered writer-to-Q buses causally control held-out attention

## Question

Checkpoint 0035 showed that training aligns OV writer subspaces with a shared
next-layer Q-reader geometry.  Is that geometry merely statically readable, or
does the trained model actually use the discovered channels when computing
attention?

Discovery remains weights-only and unsupervised.  Prompts enter only after the
rank, representation, head splits, intervention, outcome, aggregation, and
null are frozen.

## Frozen channel

For source writer basis `B_a` and each discovery Q-reader basis `R_b`, define

```text
K_ab = B_a.T @ R_b @ R_b.T @ B_a.
```

Normalize each nonzero `K_ab` to unit trace, average over four discovery Q
heads, and let `U_a` contain the top four eigenvectors.  The ambient residual
channel is the rank-four projector

```text
C_a = B_a @ U_a @ U_a.T @ B_a.T.
```

The other four Q heads are never used to construct `C_a`.  The complete test
is then repeated with discovery and held-out head sets exchanged.

## Exact intervention and outcome

Pythia uses parallel residual blocks.  For source head contribution `y_a`, the
intervened next-layer residual is therefore exactly

```text
x' = x - y_a @ C_a.
```

The target layer's input LayerNorm and attention are then reevaluated.  Target
K and V preactivations are patched back to their clean values while the
intervened Q is retained.  This isolates the Q-mediated route.  The primary
outcome is Jensen--Shannon divergence between clean and intervened attention
rows for the four held-out Q heads, excluding the trivial first query
position.

The optimized local replay reproduces the original model attention exactly:
maximum absolute replay error is zero in all five target layers.

## Controls and inference

Each of 199 null draws samples one residual-coordinate permutation per target
layer.  The permutation is shared across all discovery Q heads and every
source writer in that layer.  It retains the dimensions and complete mutual
geometry of the discovery Q population while producing surrogate channels
whose ambient orientation is not privileged by the trained model.  Eight
independent Haar-random rank-four channels inside each identical writer span
provide a diagnostic second null.

All 40 adjacent-layer source heads are included.  Two conjunctive endpoints
are computed:

1. mean attention JS within each source layer, then equal mean over layers;
2. within each layer, summed JS divided by summed removed-energy fraction,
   then equal mean over layers.

The ratio-of-sums avoids unstable per-head division.  Both endpoints must
exceed their coordinate null in both 4/4 head splits.  The final p-value is the
maximum of the four one-sided empirical p-values.  Checkpoints, tokens, heads,
and attention rows are not treated as independent inferential replicates.

## Fresh confirmation corpus

The final run uses 32 length-64 documents sampled deterministically from the
fixed `NeelNanda/pile-10k` revision.  All 192 document rows used by earlier
activation and QK studies were excluded before sampling.  The frozen result
was first seen on an eight-document pilot and then repeated without changing
the method on this unused corpus.

## Results

| Split | Endpoint | Real | Coordinate null | Real/null | p |
|---|---|---:|---:|---:|---:|
| forward | attention JS | 0.001861 | 0.000374 | 4.98x | 0.005 |
| forward | JS / removed energy | 0.012566 | 0.005945 | 2.11x | 0.005 |
| reverse | attention JS | 0.001284 | 0.000273 | 4.70x | 0.005 |
| reverse | JS / removed energy | 0.010118 | 0.004329 | 2.34x | 0.005 |

The joint two-split intersection--union p-value is `0.005`, the minimum
resolvable value with 199 null draws.  The Haar means closely match the
coordinate-null means for every endpoint.

The result is population-wide rather than driven by a few selected edges:

- the real channel beats the per-source mean coordinate null for 39/40 heads
  in raw attention JS in each split;
- it beats the per-source energy-adjusted null for 38/40 heads in each split;
- median per-head advantages are about 4.28--4.32x raw and 2.56--2.65x after
  energy adjustment;
- every source layer has a positive advantage in both endpoints and splits;
- removing any one source layer leaves every endpoint beyond all 199 null
  draws (`IUT p=0.005`).

An earlier eight-document run gave the same conclusion (approximately
4.0--4.7x raw and 2.0--2.2x energy-adjusted), so the fresh corpus is a genuine
replication rather than the only positive sample.

## Conclusion

The communication-bus geometry is not only a static overlap.  A rank-four
ambient channel discovered from the weights of four next-layer Q heads is
preferentially occupied by real source-head outputs, and selectively removing
that component changes the actual attention patterns of four unseen Q heads
roughly five times more than matched surrogate channels.  Greater activation
energy explains part, but not most, of the effect.

This is direct evidence for a reusable population-level primitive:

```text
source-head residual channel -> shared downstream Q interface -> attention
```

## Limits

- The causal outcome is immediate next-layer attention, not downstream loss or
  a semantic behavior.
- K and V are deliberately patched clean, so the intervention establishes a
  Q-mediated route rather than the natural combined downstream effect.
- The primary outcome is measured on held-out Q heads, but an explicit
  target-versus-off-target attention selectivity statistic is not yet included.
- Only one Pythia seed and scale receives causal confirmation here; GPT-2 has
  static replication but not this exact intervention.
- Rank four is a frozen probe and is not claimed to be the intrinsic bus rank.
- The coordinate permutation is a conditional surrogate-channel null, not a
  full retraining null.
- Literature novelty remains to be established separately.

## Next high-value step

Freeze the same channel construction and test two additions hierarchically:

1. natural unpatched propagation into later logits/loss, with matched removed
   energy and whole-head-effect calibration;
2. exact causal replication in GPT-2 small or a second Pythia scale/seed.
