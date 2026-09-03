# QK Reducing-Subspace Protocol

Status: frozen before implementation and real-data fitting  
Date: 2026-08-26  
Primary model: EleutherAI/pythia-70m-deduped  
Primary view: processed QK operators

## Scientific question

Can one head's QK trajectory be organized by stable, automatically discovered
read/write compartments, without assuming that the transformations inside the
compartments are rank one, diagonal, copy-like, or shared with other heads?

For one head at checkpoint t, let A_t map residual key/source space to residual
query/destination space. The primary two-block model learns output and input
projectors P and Q and predicts

    A_t ~= P A_t Q + (I-P) A_t (I-Q).

Its cross leakage is

    ||P A_t (I-Q)||_F^2 + ||(I-P) A_t Q||_F^2.

Equivalently, an exact reducing pair obeys P A_t = A_t Q for every checkpoint.
The transformations inside both diagonal blocks remain completely unrestricted.

## Why the trajectory is essential

One isolated matrix has a trivial singular-vector decomposition into one-
dimensional blocks. Requiring the same projectors to predict checkpoints that
were not used to fit them makes the claim nontrivial.

The primary forward split is:

- training: step0, step64, step512, step1000, step4000;
- validation: step16000;
- confirmation: step64000, step143000.

A late-training sensitivity split is:

- training: step1000, step4000, step16000;
- validation: step64000;
- confirmation: step143000.

All support bases, projectors, ranks, and resolution choices are learned without
the confirmation checkpoints.

## Normalization and active support

The primary analysis normalizes every complete operator to unit Frobenius norm;
raw norms are stored separately. This asks about shape rather than training-time
gain.

The large exact or near kernel is not eligible to become a compartment. Training
operators first define active output and input supports from

    G_out = sum_t A_t A_t^T
    G_in  = sum_t A_t^T A_t.

Only their leading eigenspaces are analyzed. The exploratory support dimensions
are 32, 64, and 96. The frozen primary support dimension is 64.

## Two-block estimator

Inside a training-only active support, write the projected operators as B_t.
For fixed ranks p and q, maximize

    sum_t ||P B_t Q||_F^2 + ||(I-P) B_t (I-Q)||_F^2.

Alternating eigenspace updates are exact conditional maximizers:

    P <- top-p eigenspace of 2 sum_t B_t Q B_t^T - sum_t B_t B_t^T
    Q <- top-q eigenspace of 2 sum_t B_t^T P B_t - sum_t B_t^T B_t.

Use a deterministic energy initialization, a mean-operator initialization, and
multiple seeded random starts. Retain the highest training objective.

The exploratory rank pairs are:

- support 32: (8,8), (8,16), (16,8), (16,16);
- support 64: (16,16), (16,32), (32,16), (32,32);
- support 96: (24,24), (24,48), (48,24), (48,48).

The frozen primary comparison is support 64 with (p,q)=(32,32).

## Held-out metrics

For every validation or confirmation operator, report:

1. active-support energy divided by complete-operator energy;
2. block-diagonal energy divided by complete-operator energy;
3. block-diagonal concentration inside the active support;
4. cross-leakage fraction inside the active support;
5. retained core-coordinate fraction

       [p q + (d-p)(d-q)] / d^2;

6. excess concentration over that coordinate fraction;
7. gain over random projectors with identical ranks and supports.

Resolution is selected globally from mean validation performance, not separately
for each head. Confirmation checkpoints are evaluated once using the frozen
selection.

## Stability

Fit the frozen primary configuration (support 64, ranks 32 and 32) on two
disjoint subsets of the development-to-mature training checkpoints and compare
ambient projectors with normalized trace overlap. Stability is deliberately
restricted to this fixed configuration: a 96-dimensional support cannot be
identified from a one-checkpoint partition because each QK operator has rank at
most 64. Because equal-size blocks can exchange labels, use the best legal
direct or complement alignment. Report population distributions rather than
requiring every head to be stable. The three-checkpoint late split is too short
for a strong disjoint stability test and is therefore not used for that claim.

## Null hierarchy

Every null is regenerated and fitted end to end.

### 1. Independent spectrum-Haar null

Preserve every checkpoint's singular values and independently Haar-randomize
left and right singular frames. This is a deliberately weak sanity null.

### 2. Within-layer side-trajectory pairing null

For each layer, keep one head's left singular-frame and spectrum trajectory but
replace its right singular-frame trajectory with another head's trajectory,
using one constant derangement across checkpoints. This preserves marginal side
development while breaking trained left/right pairing.

### 3. Smooth singular-frame trajectory null

Preserve:

- every checkpoint singular spectrum;
- every exact adjacent-checkpoint left-frame overlap matrix;
- every exact adjacent-checkpoint right-frame overlap matrix.

Start each null trajectory from Haar frames. If F_(t-1) is the null frame and
C_t is the real adjacent overlap matrix, generate

    F_t = F_(t-1) C_t + N_t sqrt(I - C_t^T C_t),

where N_t is a random orthonormal frame in the complement of F_(t-1). This
matches local spectral and orientation drift while destroying persistent ambient
organization not implied by one-step smoothness.

The primary real/null comparison uses 19 repetitions, giving a minimum finite-
null upper-tail p-value of 0.05.

## Synthetic acceptance tests

Before real fitting, the estimator must:

1. recover known fixed input/output blocks containing independently varying
   dense transformations;
2. reject a dense random matrix family at the same dimensions;
3. preserve scores and rotate ambient projectors correctly under independent
   orthogonal input/output basis changes;
4. avoid selecting inactive kernel directions as meaningful blocks;
5. return deterministic results for a fixed seed;
6. validate shapes, ranks, finiteness, and nonempty inputs.

## Decision gate

Evidence for within-head compartments requires all of:

- held-out concentration above dimension-matched random projectors;
- an excess over the smooth trajectory null;
- positive confirmation after global validation selection;
- stability above matched nulls for at least a reproducible population subset;
- no dependence on a single isolated support dimension or rank pair.

If the real population does not exceed the smooth null, the conclusion is not
that QK is unstructured. It is that fixed intrinsic compartments inside isolated
head trajectories are not supported at this resolution; typed architectural
connections remain the stronger primitive.

## Conditional follow-up

Only if the gate is positive:

1. extract the two unrestricted local transformations at the final checkpoint;
2. compare normalized singular profiles, common-rank cores, unmatched energy,
   and effective ranks across heads;
3. compare hard clusters, local neighborhoods, sparse dictionaries, continuous
   PCA/manifold coordinates, and bespoke residuals by held-out description
   length;
4. attach architectural producer/consumer context before assigning a functional
   name.
