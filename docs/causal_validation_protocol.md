# Conditional causal validation protocol

Unsupervised geometry generates hypotheses; it does not assign functions. A
head atom receives a semantic name only after a controlled intervention exceeds
a same-layer, low-loading head control and recurs across at least two heads.

## Candidate admission

A compact atom is eligible only if it passes all of the following without
functional labels:

1. trajectory-bootstrap similarity exceeds its matched random baseline by 0.10;
2. checkpoint coefficient share has an FDR-controlled time trend;
3. effective participation spans at least four head trajectories;
4. effective participation spans at least two layers, with no layer supplying
   more than 75% of usage.

Layer-residual atoms are preferred. Raw compact atoms remain provisional when
the corresponding residual analysis does not reproduce their full trajectory.

## Interventions

For each admitted atom, compare its highest-loading final-checkpoint heads with
same-layer heads having minimal absolute atom loading:

- zero the head result at all token positions;
- replace the result with its clean-run mean;
- patch the clean result into matched corrupted sequences.

Primary metrics are next-token cross-entropy change, KL divergence from the
clean distribution, and patching recovery. Correct across all candidate-head
and metric tests with Benjamini-Hochberg. Prompt families and corruptions must
be frozen before inspecting intervention outcomes.

## Current gate status

The raw compact atlas produces provisional candidates, but no layer-residual
atom currently passes stability, temporal change, and cross-layer reuse at the
same time. The generated causal plan is therefore scaffolding, not evidence of
causal function. Running it is justified as a falsification test of the raw
candidates, not as confirmation of named motifs.
