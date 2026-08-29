# Checkpoint 0034: gauge-invariant writer-to-query channels

## Question

Do adjacent attention components possess reusable communication channels that
are easier to detect than similarities between isolated whole-head matrices?
This pilot uses weights only and imposes no token labels, prompts, behavioral
classes, or predefined semantic operator types.

## Representation

For source head `a`, let `B_a` be an orthonormal basis for the residual-stream
row space of `W_O^a`.  For target head `b`, let `R_b` be an orthonormal basis
for the residual-stream column space of `W_Q^b`.  These spaces do not change
under invertible changes of the internal coordinates of either factor.

The pairwise restricted overlap operator is

```text
K_ab = B_a^T R_b R_b^T B_a.
```

Its trace measures total writer/reader overlap.  Its eigenvectors identify
directions inside the source writer space that are preferentially read by the
target.  This is a deliberately coarser invariant for RoPE Q/K factors because
it discards internal Q/K correspondence.

For each source writer and its next-layer population of query readers, the
pilot learns the leading eigenspace of the mean `K_ab` on a balanced half of
the target heads and measures captured energy on the held-out half.  Sixteen
deterministic balanced splits are used.  Rank 4 Q fan-out was frozen as the
single confirmatory endpoint.

Three statistics must all be positive:

1. mean total pair overlap;
2. equal-partner cross-fitted capture after normalizing each `K_ab` to unit
   trace;
3. overlap-weighted cross-fitted capture using the unnormalized operators.

The third statistic prevents a nearly nonexistent but concentrated
intersection from looking important.  The confirmatory p-value is the maximum
of the three one-sided empirical p-values (an intersection-union test).

## Null and controls

For each target layer, one random residual-coordinate permutation is applied
jointly to every Q/K/V reader and every head.  This preserves every individual
subspace dimension and all partner-partner principal angles within the layer,
while breaking their collective alignment with the fixed upstream writer.
The null is therefore a conditional coordinate-alignment benchmark, not a
claim that trained residual coordinates are universally exchangeable.

Additional controls:

- repeat the primary analysis while omitting each source layer;
- compare actual next-layer Q readers with the nearest different target-layer
  Q population for the same writer;
- apply the frozen method to a second model architecture.

## Results

| Model | Total overlap real / null | Equal-partner rank-4 capture real / null | Weighted rank-4 capture real / null | IUT p |
|---|---:|---:|---:|---:|
| Pythia-70m-deduped, step143000 | 0.16160 / 0.12525 | 0.13052 / 0.09742 | 0.12757 / 0.09742 | 0.005 (199 nulls) |
| GPT-2 small | 0.11142 / 0.08341 | 0.14513 / 0.10148 | 0.14453 / 0.10149 | 0.010 (99 nulls) |

No null draw reached the real statistic in either model.  All three statistics
also remained beyond every corresponding null after omitting any one source
layer.

The actual next-layer population exceeded a mismatched target layer on all
three measures in Pythia (`p=0.005` each).  In GPT-2, total overlap and
equal-partner reuse replicated (`p=0.01`), but the weighted actual-minus-
mismatch difference did not (`p=0.11`).  Strict target-layer specificity is
therefore only partially replicated.

## Conclusion

This is positive evidence for structure beyond isolated weight matrices:
trained OV writer subspaces align with next-layer Q-reader populations more
strongly than expected from the complete internal geometry of those reader
populations, and a low-dimensional channel learned from some target heads
predicts where held-out target heads read.  The result replicates across two
architectures and is not driven by one layer.

It does **not** yet establish the transformation, sign, gain, semantic meaning,
activation usage, or causal necessity of a channel.  It establishes reusable
residual communication geometry.

## Deferred invariant-object ladder

The following directions remain available, ordered from the nearest extension
to higher complexity:

1. extract and compare the actual ambient rank-4 channel projectors;
2. test source-specificity against a fitted generic layer reader bus;
3. retain gains with whitened canonical interface operators;
4. study gauge-invariant two-edge path maps such as `M_OV^a M_QK^b`;
5. search for shared invariant subspaces, commutators, and short operator words;
6. combine QK routing and OV transport into conditional rule tensors;
7. validate frozen weight-discovered channels with activations and causal
   interventions only after discovery.

The next experiment should begin with (1) and (2), because they determine
whether the replicated population effect resolves into identifiable reusable
channels or merely a broad layer-level bus.
