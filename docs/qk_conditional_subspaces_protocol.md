# QK conditional-subspace protocol

## Question

Can an unsupervised, low-dimensional representation of *conditional QK
routing* predict attention relations on fresh documents, and do the recurrent
cross-layer QK neighborhoods from checkpoint 0006 share this representation
beyond their layer pair?

The object of study is not a semantic head label.  For each head, a relational
event pairs one source position receiving an unusually high QK score with a
matched source position from the same relative-position stratum.  The analysis
then asks which query and key directions distinguish those two sources.

## Scope and frozen inputs

- Model: `EleutherAI/pythia-70m-deduped`, final checkpoint `step143000`.
- Dataset: pinned `NeelNanda/pile-10k` revision
  `127bfedcd5047750df5ccf3a12979a47bfa0bafa`.
- Sequence length: 64 tokens, including a prepended BOS token.
- Discovery split: the 32 training documents in
  `activation_validation_pilot.npz`.
- Tuning split: its 32 held-out pilot documents.
- Confirmation split: 64 new deterministic documents (seed 1729), excluding
  every row used by the pilot splits.
- Target population: all 48 attention heads, with primary family tests on the
  six QK recurrent-edge sets saved in `subspace_family_audit.json`.

No token text, hand-written token class, known-head label, or activation
example is used to fit the event representation.  Text may be inspected only
after quantitative results and their controls are frozen.

## Event construction

For a head `h`, destination position `i`, and strict-past source `j`, let
`ell_ij^h` be the actual rotary-aware QK logit before the causal softmax.  In
the discovery split, estimate a mean and standard deviation independently for
each exact relative offset `delta = i - j`, then form

```text
z_ij^h = (ell_ij^h - mu_h,delta) / (sigma_h,delta + epsilon).
```

Those discovery normalizers are reused without refitting on tuning and
confirmation.  For each eligible destination (`i >= 8`), define `j+` as the
strict-past source with the largest residualized score.  Select `j-` from the
same document and the same offset bin as `j+`, excluding `j+`, whose score is
closest to that bin's median; break ties by smaller source index.

Offset bins are `[1,2]`, `[3,4]`, `[5,8]`, `[9,16]`, `[17,32]`, and `[33,63]`.
BOS sources and self-attention are retained only as separate diagnostics.
Heads need at least 1,000 discovery events to enter the primary analysis.

## Conditional subspace estimator

Fit in pre-RoPE head coordinates, but choose events with actual post-RoPE
logits.  For matched events, compute

```text
Delta C = mean((q - q_bar)^T (k_plus - k_minus)).
```

Shrink the centered query and key covariances by

```text
Sigma_lambda = (1 - lambda) Sigma + lambda trace(Sigma) / 64 I,
```

with tuning choices `lambda` in `{1e-4, 1e-3, 1e-2, 1e-1}`.  Whiten and take
an SVD:

```text
Sigma_q^(-1/2) Delta C Sigma_k^(-1/2) = U S V^T.
```

Candidate ranks are 1, 2, 4, 8, and 16.  The existing rank-4, rank-8, and
rank-16 recurrent sets remain the primary family-test targets.  A per-head
descriptive rank is the smallest rank within one document-bootstrap standard
error of the best tuning margin.

## Cross-head comparison and gauge safety

Raw head-coordinate singular vectors are not compared across heads.  The
change of coordinates `q -> qG`, `k -> kG^(-T)` preserves QK logits, so this
would be gauge-dependent.

Instead map learned coefficient bases back into the normalized residual basis,
folding in the layer-normalization gain:

```text
R_Q = P_center diag(gamma_layer) W_Q A
R_K = P_center diag(gamma_layer) W_K B,
```

where `A = Sigma_q^(-1/2) U_r` and `B = Sigma_k^(-1/2) V_r`.  QR-orthogonalize
the mapped bases and compare their projectors.  Report a sensitivity analysis
without the layer-normalization gain.  A synthetic change-of-basis test is
mandatory.

## Confirmation metrics and controls

On confirmation data report matched-source logit margin, conditional-feature
contribution, fraction of centered margin explained, squared correlation with
the full margin, AUC for positive versus matched-negative sources, and 1,000
document-bootstrap confidence intervals.

Use equal-rank unconditional PCA, QK-weight SVD, Haar subspaces, and shuffled
triplets as subspace controls.  For family recurrence, compare against exact
ordered layer-pair controls twice: once with all possible pairs and once with
the union of all 26 recurrent QK edges excluded.  Include position-only and
token-identity diagnostic fingerprints, but do not use either to fit the
representation.

For each of the six preregistered recurrent sets, test conditional relation
fingerprints, mapped query/key projectors, and feature-attributed logit
fingerprints.  Use 199 randomizations while developing and 9,999 for the final
frozen run.  Apply Benjamini--Hochberg correction separately within each target
family.

## Success criterion

The result is a scientific advance only if learned conditional factors beat
shuffled and Haar controls on fresh confirmation data; at least one
preregistered edge set is closer than exact layer-pair controls at BH-adjusted
`q <= 0.05`; the effect has the same direction on discovery, tuning, and
confirmation; and it survives excluding all 26 recurrent edges from the null.

Failure modes are informative: an offset-only result is positional, a
token-identity-only result is lexical, a confirmation failure is overfitting,
and no low-rank advantage suggests diffuse or nonlinear routing at this scale.
