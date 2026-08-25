# Checkpoint 0015: preliminary early-training QK channel replication

Using the identical frozen token sequences, the rank-four joint QK bilinear
protocol was run at Pythia-70M-deduped step 16000.  It remains present early:
mean held-out rank-four R-squared is 0.1412 and 45/48 heads beat their best
equal-rank PCA control by a positive document-bootstrap interval.

The matched-iteration mature control is now complete: rank-four held-out
R-squared rises from **0.1412** at step 16000 to **0.1930** at step 143000,
on identical discovery, tuning, and confirmation token sequences with 200
iterations in both fits.  Thus the compact rank-four joint QK representation
does become more predictive over this interval.

The trajectory is not simply monotonic improvement on every diagnostic.  The
number of heads whose rank-four model beats its best PCA control by a positive
bootstrap interval is 45/48 at 16k but 28/48 at 143k.  Early training has
higher-rank structure that is comparatively easy to beat with the joint model,
whereas the mature model has a stronger absolute rank-four fit but more
competitive PCA baselines.  A middle checkpoint is needed before drawing a
full emergence curve.

The collector now supports reusing a frozen confirmation-token artifact,
allowing all checkpoints to be compared on exactly the same inputs.
