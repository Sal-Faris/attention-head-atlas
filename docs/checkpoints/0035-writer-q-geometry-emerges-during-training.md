# Checkpoint 0035: writer-to-Q geometry emerges during training

## Question

The previous checkpoint found gauge-invariant alignment between OV writer
subspaces and the next layer's population of Q readers.  Is this alignment
already present because of architecture or initialization, or is it created
by training?

This experiment is weights-only.  It uses no prompts, activations, task
labels, token classes, behavioral head names, or semantic supervision.

## Frozen objects and statistics

For source head `a` and target Q head `b`, let `B_a` and `R_b` be orthonormal
bases for the residual-stream writer and reader subspaces.  The restricted
overlap operator is

```text
K_ab = B_a.T @ R_b @ R_b.T @ B_a.
```

At each of eight Pythia-70m checkpoints, the same four balanced partner-head
splits measure three rank-4 population statistics:

1. total writer/Q overlap;
2. equal-partner cross-fitted channel reuse;
3. overlap-weighted cross-fitted channel reuse.

Source layers receive equal weight.  Rank 4 was frozen from checkpoint 0034;
it is not claimed to be the natural or unique dimension.

## Whole-trajectory null

Each of 199 null draws samples one residual-coordinate permutation per Q
layer and applies that same permutation to every Q head at every checkpoint.
Consequently, a null draw preserves:

- every Q head's trajectory;
- checkpoint-to-checkpoint smoothness;
- all Q-Q principal-angle geometry;
- formation of a generic shared Q-reader bus;
- all writer trajectories.

It destroys only the collective residual-coordinate alignment of writers with
the evolving Q population.  Checkpoints are treated as one correlated
trajectory, never as independent replicates.

The confirmatory statistic is final minus initialization for each of the
three endpoints.  All three must be positive and the intersection-union
p-value is the maximum of their one-sided empirical p-values.  A secondary
test compares the fixed log-time slope with slopes of whole null trajectories.

## Results

| Metric | Step 0 | Final | Observed change | Mean null change | Empirical p |
|---|---:|---:|---:|---:|---:|
| total overlap | 0.12543 | 0.16160 | +0.03616 | -0.00001 | 0.005 |
| equal-partner rank-4 reuse | 0.06298 | 0.13363 | +0.07066 | +0.03672 | 0.005 |
| weighted rank-4 reuse | 0.06296 | 0.12974 | +0.06678 | +0.03672 | 0.005 |

The joint endpoint-emergence test passed at `p=0.005`.  The three log-time
slope tests also each had `p=0.005`, giving a joint slope IUT of `p=0.005`.
No null trajectory reached the observed statistic.

Initialization was inside the null distribution for every metric (empirical
percentiles 0.89--0.91).  Its null-relative excess was only 0.5--1.1 percent
of the corresponding final excess.  Thus the final effect was not a large
alignment inherited from initialization.

Removing any one of the five source layers left all three endpoint contrasts
positive and beyond every null draw (`IUT p=0.005` for every omission).

## Interpretation

At initialization, the observed values are almost exactly the isotropic
expectations:

```text
writer/Q overlap: 64 / 512 = 0.125
rank-4 capture inside a 64-dimensional writer: 4 / 64 = 0.0625
```

Training creates two separable effects:

1. Q-reader subspaces become mutually organized into a reusable bus.  The
   null preserves this, explaining about `+0.0367` of the reuse growth.
2. OV writer subspaces additionally align with that learned bus.  This
   explains essentially all `+0.0362` overlap growth and the remaining
   approximately `+0.030--0.034` reuse growth beyond the null.

This is stronger than a static non-randomness result.  It shows a relational,
gauge-invariant communication geometry being constructed during optimization.

## Limits

- Only one Pythia model seed and scale has checkpoint data here.
- The coordinate-permutation null is a conditional alignment null, not every
  imaginable training null.
- Rank 4 is a frozen probe, not an estimated intrinsic dimension.
- The result identifies communication geometry, not its semantic content,
  activation usage, transported transformation, or causal necessity.
- Novelty relative to the complete literature has not yet been established.

## Best next step

Freeze the learned channel projectors without semantic labels, then determine
what residual directions and token/feature families load onto them and whether
projecting out a channel selectively disrupts the downstream Q interfaces it
predicts.  Discovery should remain weight-only; activations and interventions
should be held-out validation.
