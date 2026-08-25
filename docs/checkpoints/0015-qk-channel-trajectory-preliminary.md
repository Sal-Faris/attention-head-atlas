# Checkpoint 0015: preliminary early-training QK channel replication

Using the identical frozen token sequences, the rank-four joint QK bilinear
protocol was run at Pythia-70M-deduped step 16000.  It remains present early:
mean held-out rank-four R-squared is 0.1412 and 45/48 heads beat their best
equal-rank PCA control by a positive document-bootstrap interval.

This is below the mature checkpoint's 0.2789 rank-four R-squared, consistent
with stronger compact QK organization later in training.  This comparison is
preliminary because the early run used 200 optimization iterations whereas the
original mature confirmation used 400.  The next trajectory action is a
matched-iteration mature control before interpreting the difference as a
learning curve.

The collector now supports reusing a frozen confirmation-token artifact,
allowing all checkpoints to be compared on exactly the same inputs.
