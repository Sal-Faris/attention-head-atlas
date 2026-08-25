# Checkpoint 0021: typed composition edges predict causal adjacent-layer influence

Checkpoint 0020 established that static OV-writer→Q/K/V-reader edges predict
activation-weighted linear reach.  The remaining concern was that this might
be only a descriptive correlation.  This checkpoint applies an intervention.

For each source layer, a single forward batch contains one clean condition and
eight conditions, each of which zeroes one source head's post-attention,
pre-output-projection vector.  This is a real head-level ablation: the altered
source output then propagates through the rest of its transformer block and
into the next layer.  The outcome for every target head is the RMS change in
its raw Q, K, or V preactivation.  All 64 source-head→target-head pairs at all
five adjacent layer pairs are tested; no edge is selected using activations or
the intervention outcome.

| Target reader | Static-edge vs causal-effect Spearman | Matched head-identity null mean | Upper-tail p-value |
| --- | ---: | ---: | ---: |
| Q | **0.727** | 0.007 | 0.001 |
| K | **0.571** | 0.002 | 0.001 |
| V | **0.655** | 0.000 | 0.001 |

The control independently shuffles source and target head labels within each
same adjacent layer pair.  Thus the relationship is not explained simply by
some layer pairs being more sensitive than others.

**Conclusion.** The typed static edge score is causally informative: stronger
OV-writer→reader links predict which next-layer Q/K/V coordinates actually
change when the source head is removed.  This is the project’s clearest
evidence yet for a weight-only structural primitive with causal meaning.

The claim is deliberately narrow.  The intervention removes an entire source
head, so it does not yet isolate one low-rank channel inside that head, and it
only tests immediate next-layer influence.  It does not show that a candidate
edge affects a particular semantic behavior or final logits.  The next
refinement should ablate the source-output component that overlaps a selected
target reader, then test whether the effect concentrates in the predicted
reader and propagates to a measurable downstream behavior.
