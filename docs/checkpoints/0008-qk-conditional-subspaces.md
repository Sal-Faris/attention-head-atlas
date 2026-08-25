# Checkpoint 0008: conditional QK subspaces

## Question

Do the validated recurrent QK pairs from checkpoint 0007 share an
unsupervised, low-dimensional *conditional routing* subspace, beyond their
ordered layer pair?  Can that subspace predict which matched source receives a
higher QK score on fresh documents?

## Fixed design

The detailed preregistered design is in
[`qk_conditional_subspaces_protocol.md`](../qk_conditional_subspaces_protocol.md).
It uses the final Pythia-70M-deduped checkpoint, 32 discovery documents, 32
tuning documents, and 64 newly sampled confirmation documents, all disjoint
at the dataset-row level.

For each head and destination, the source with the largest QK logit after
exact-relative-offset residualization is matched to a neutral source in the
same offset bin.  Whitening the cross-covariance between the query and the
positive-minus-negative key yields low-dimensional query/key directions.

The analysis stores actual pre- and post-RoPE Q/K tensors.  Float32 QK
reconstruction matches model attention exactly.  Float16 changes some softmax
outputs by almost one, so the 453 MB local artifact correctly remains float32
and is not tracked in Git.

The first standalone attribution implementation incorrectly rotated all 64
head dimensions.  Pythia rotates only 16 (`rotary_pct = 0.25`).  An explicit
audit caught the error before interpretation: after correction the standalone
RoPE reconstruction has maximum absolute error `2.27e-6`.  The final result
below uses the corrected implementation and 9,999 controls.

## Results

At rank 4, the average confirmation (R^2) for the learned conditional query
feature was 0.0437.  This exceeds random Haar directions (0.0161) and factors
fit to shuffled matched triplets (0.0291), so event-selected directions are
not arbitrary.  However, unconditional query PCA reaches 0.1007.  Thus the
current conditional estimator does not isolate more predictive structure than
ordinary high-variance query directions.

After mapping factors back into the processed residual basis, five of six
pre-existing recurrent QK families are closer than exact ordered-layer-pair
controls even when every one of the 26 recurrent edges is excluded from the
null pool.  Benjamini--Hochberg adjusted q-values are 0.00012 for rank-4 key,
rank-8 query/key, and rank-16 query/key families.  The rank-4 query family is
not significant (q = 0.075).

The effect sizes are small: recurrent distances are only 0.4%--2.0% below
their exact-layer-pair null means.  This is evidence that the earlier static
QK recurrence has a conditional-subspace counterpart, but not evidence for a
clean, sparse conditional rule vocabulary.

## Interpretation

The results separate two claims:

- **Supported:** recurrent QK neighbors share weak but reproducible mapped
  conditional query/key geometry beyond layer identity.
- **Not supported:** this particular matched-event SVD discovers a behavioral
  representation superior to simple unconditional query PCA.

The next improvement should therefore not be a broader clustering sweep.  It
should test a richer conditional estimator: retain both query and key feature
contributions with their position-specific rotations, use several matched
negative sources per event, and compare a sparse bilinear model directly to
PCA at equal description length.  The present result is a gate for that work,
not a claim that the desired motif decomposition has been found.

## Reproduction

```powershell
python scripts/collect_qk_conditional_events.py
python scripts/analyze_qk_conditional_subspaces.py
```

Primary compact outputs:

- `results/pythia-70m-deduped/qk_conditional_events_v1.json`
- `results/pythia-70m-deduped/qk_conditional_subspaces_v1.json`
- `results/pythia-70m-deduped/qk_conditional_subspaces_v1.png`
