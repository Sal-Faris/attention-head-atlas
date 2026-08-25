# Checkpoint 0011: held-out examples for recurrent QK channels

## Question

Can the rank-4 joint QK channels validated in checkpoints 0009--0010 be
given a simple input-class interpretation from their strongest uses?  In
particular, do they cleanly reduce to literal token matching or a small,
reusable semantic category?

## Method

Take the 17 heads incident to the previously recurrent rank-4 **key-side**
families.  For each head:

1. Fit its rank-4 bilinear channel on discovery documents only.
2. Canonicalize its four paired query/key components with the SVD of the
   learned channel matrix.
3. On the disjoint 64-document confirmation split, calculate each component's
   signed contribution to the matched QK preference margin.
4. Record its six largest positive and negative token-context examples,
   relative-offset profile, and literal destination-token = selected-source-
   token rate in the largest-magnitude decile.

This is an unsupervised profile: token strings are displayed after component
discovery, not used to fit or label the components.

## Results

Every head has at least one component whose held-out contribution correlates
substantially with its full matched margin: the best per-head correlations
range from 0.29 to 0.81.  This is compatible with the compact bilinear model,
but it alone does not make a component a semantic feature.

Inspection does **not** reveal a clean universal component vocabulary.  The
high-magnitude examples mix ordinary prose, markup, code-like text,
newlines, and document beginnings.  Some components place relatively more
magnitude on long relative offsets, but this varies by head.

Literal token repetition provides only a weak candidate interpretation.  A
few components have a 2--4.5x enrichment of exact destination/source token
matches in their top magnitude decile, but the largest absolute top-decile
match rate is only 6.7%.  That is not enough to call any of them a copy
channel; most high-contribution events are not exact matches.

## Interpretation

The channel result should currently be understood as a compact **continuous
routing geometry**, not yet as a discovered list of semantic rules.  Raw top
examples are useful for avoiding overinterpretation, but they are not a sound
way to infer conditional input classes by eye.

The next appropriate test is automatic and held out: cluster the query-side
and key-side residual inputs on discovery data, assign confirmation inputs to
those clusters, then test whether a channel component's dominant contribution
is associated with a query class, key class, or their pair beyond
offset-stratified permutations.  A positive result would give the first
evidence for the user's proposed conditional routing rules; a negative result
would support smoothly graded rather than discretely gated inputs.

## Reproduction

```powershell
python scripts/profile_recurrent_qk_channels.py
```

Primary compact output:

- `results/pythia-70m-deduped/recurrent_qk_channel_profiles_v1.json`
