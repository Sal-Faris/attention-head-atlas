# Checkpoint 0007: held-out activation validation

## Question

Do static QK/OV operator and subspace geometries predict how heads behave on
real tokens, beyond depth effects? Do the recurrent cross-layer neighbors from
checkpoint 0006 behave more similarly than layer-pair-matched controls?

This remains an unsupervised analysis: no published functional labels are used.

## Fixed pilot

The final `step143000` Pythia-70M-deduped checkpoint is evaluated on a pinned
revision of [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k),
a compact sample from The Pile. A deterministic sample of 64 documents supplies
32 train and 32 held-out test sequences of 64 tokens each. Documents do not
cross splits.

Three behavior geometries are accumulated directly as 48 by 48 Gram matrices,
without retaining full activations:

1. actual attention probabilities, centered across heads at every corpus
   feature;
2. bias-free processed OV responses to the actual normalized residual stream,
   centered across heads;
3. those OV responses mixed by each head's actual attention probabilities,
   giving a centered composed head-output geometry.

Predictors include normalized full-operator Frobenius distance and rank-4,
rank-8, and rank-16 query/key or read/write projector distances. For composed
outputs, QK-only, OV-only, and joint QK+OV predictors are compared. Predictor
selection uses train documents only; statistics use held-out documents only.

All primary tests use 499 head-identity permutations within layers. This
preserves the full layer geometry while breaking the specific mapping between a
static head and its behavior.

The recurrent-pair follow-up uses only the held-out behavior matrices. Each
selected cross-layer edge is compared with random pairs having exactly the same
two layers. It uses 999 repetitions and Benjamini-Hochberg correction across
all 18 view/rank/side/target comparisons.

## Results

### The behavior estimates are highly reliable

Train/test distance Spearman correlations are 0.994 for attention patterns,
0.994 for OV responses, and 0.992 for composed outputs. The pilot corpus is
therefore large enough to estimate these population geometries consistently.

### Full operators retain behaviorally relevant information

For centered attention patterns, raw QK distance has held-out Spearman 0.265
versus a within-layer-shuffle mean of 0.159 (p = 0.002). Rank-8 query-subspace
distance is competitive at 0.269, but its independently selected advantage
over raw QK is only 0.004 and is not significant (p = 0.116). Key-only and joint
query/key projector views are weaker.

For centered activation-conditioned OV responses, raw OV distance reaches
0.278 versus a shuffle mean of 0.076 (p = 0.002). The best train-selected
subspace view, rank-16 write distance, is 0.016 worse than raw on test. Its
apparent 0.263 correlation is explained by layer structure (shuffle mean 0.253,
p = 0.414). The rank-16 read view has a small beyond-layer association
(0.071 versus 0.011, p = 0.038).

For actual centered composed head outputs, joint raw QK+OV distance has
held-out Spearman 0.236 versus a shuffle mean of 0.115 (p = 0.002). OV-only raw
distance reaches 0.228 and QK-only raw distance reaches 0.192; both retain
beyond-layer signal (p = 0.002). The best train-selected subspace predictor,
rank-4 OV, is 0.035 worse than the joint raw metric on test.

The absolute depth baseline is strongly correlated with QK behavior (0.459)
and composed behavior (0.235), but its permutation null is identical and its
p-value is 1.0. Depth is a major population axis, not evidence of head-specific
functional prediction.

### Recurrent QK neighborhoods validate strongly; OV validates weakly

All six query/key recurrent-edge sets are behaviorally closer than exact
layer-pair controls after FDR correction. Their held-out attention-pattern
distance ratios range from 0.857 to 0.909: a 9% to 14% reduction, with
q-values from 0.009 to 0.024.

OV effects are smaller. Rank-8 and rank-16 recurrent edges produce OV-response
distance reductions of roughly 3% to 6% after correction. For composed head
outputs, rank-8 read/write and rank-16 write edges show reductions of roughly
2% to 5%; rank-4 edges do not survive correction.

This specifically falsifies the tempting inference that rank-4 OV write is the
best functional representation merely because it had the clearest silhouette.
Visual separation and functional prediction are different criteria.

## Interpretation

The earlier suggestion that most full-matrix variation might be functionally
invisible is too strong. Complete QK/OV maps contain behaviorally useful
information that low-rank subspace identity alone loses. At the same time,
subspace neighborhoods are not epiphenomenal: recurrent QK pairs, and a weaker
subset of OV pairs, transfer to held-out behavior.

The current best representation is multiscale:

- full operators for global functional prediction;
- continuous subspace neighborhoods for reusable local organization;
- recurrent QK pairs as the strongest candidate functional families;
- higher-rank OV write/read neighborhoods as weaker compositional candidates.

Hard cluster labels and whole-matrix atom catalogues remain secondary.

## Limits and next gate

This is one small model, one corpus sample, and correlational rather than causal
evidence. The OV response omits biases to remain aligned with the static
operator definition; composed outputs nevertheless use the model's actual
rotary-aware, bias-aware, softmax attention probabilities. Centering removes
the generic behavior shared across heads, and raw-behavior sensitivity results
are retained in the report.

The next high-value step is to characterize the validated recurrent QK pairs:
which token relations they jointly attend to, when their behavioral similarity
emerges during training, and whether patching or ablating a pair produces
related effects. OV causal work should concentrate on the validated rank-8/16
edges rather than the prettier rank-4 cut.

## Reproduction

```powershell
python scripts/run_activation_validation_pilot.py
python scripts/audit_recurrent_pair_behavior.py
```

Primary outputs:

- `results/pythia-70m-deduped/activation_validation_pilot.json`
- `results/pythia-70m-deduped/activation_validation_pilot.png`
- `results/pythia-70m-deduped/recurrent_pair_behavior.json`
- `results/pythia-70m-deduped/recurrent_pair_behavior.png`
