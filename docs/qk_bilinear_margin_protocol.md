# QK bilinear-margin compression protocol

## Motivation

Checkpoint 0008 found event-selected conditional QK directions beyond Haar and
shuffled controls, but a one-sided query PCA projection predicted matched QK
margins better.  That comparison does not give a joint query--key mechanism a
fair chance: the prior score discarded the learned key factor when predicting
the margin.

This checkpoint asks a narrower question:

> On the distribution of unsupervised matched routing events, can a rank-r
> bilinear map jointly compress QK margins better than equal-rank PCA
> projections?

This is a prompt-distribution-weighted decomposition of a fixed QK operation,
not a semantic classifier.  It uses no token labels or hand-written features.

## Splits

- Discovery: checkpoint-0008's 32 discovery documents.
- Tuning: checkpoint-0008's 32 tuning documents.
- Confirmation: 64 new documents, deterministically sampled with seed 2718,
  excluding every row used in checkpoints 0007 and 0008.

The existing 64-document confirmation split is not reused because it informed
the decision to test this new representation.

## Model

For each event e, use its actual post-RoPE query `q_e`, matched key difference
`d_e = k_e+ - k_e-`, and actual margin

```text
y_e = q_e dot d_e / sqrt(d_head).
```

Fit a rank-r matrix

```text
M_r = U_r V_r^T
```

by minimizing discovery squared error

```text
mean_e (y_e - q_e M_r d_e / sqrt(d_head))^2 + ridge (||U_r||_F^2 + ||V_r||_F^2).
```

Candidate ranks are 1, 2, 4, 8, and 16; ridge values are `1e-5`, `1e-4`,
`1e-3`, and `1e-2`.  Optimize only on discovery; select the pair by tuning
margin R-squared.  The optimizer must be deterministic and verified on a
synthetic low-rank recovery test.

## Controls

At equal rank compare against:

- query-PCA projector: `M = P_query`;
- key-PCA projector: `M = P_key`;
- independent Haar query/key factors;
- a bilinear model trained after shuffling matched key differences;
- the rank-r truncated identity map (the optimal unweighted Frobenius
  approximation, included as a diagnostic rather than a behavioral baseline).

The primary confirmation statistic is event-level margin R-squared, with a
1,000-document bootstrap confidence interval for the difference from the best
PCA control.  Report both overall results and the six frozen relative-offset
bins.  Do not inspect token strings until the quantitative report is frozen.

## Decision rule

Advance to cross-head recurrence only if the joint bilinear model exceeds the
best equal-rank PCA control on the new confirmation split, with a positive
document-bootstrap 95% interval for the difference, in at least one
preregistered rank.  Otherwise conclude that this event distribution has no
evidence yet for a compact joint bilinear mechanism beyond ordinary variance
directions.
