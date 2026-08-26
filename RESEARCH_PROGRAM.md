# Research program

## Central objective

Build an increasingly concrete, falsifiable map of reusable computations encoded in transformer attention weights, beginning from weight-only structure and only later grounding discoveries in tokens, activations, architectural paths, and causal interventions.

The program does **not** assume in advance that heads belong to discrete functional classes or that useful structure must be rank one, fixed in the residual basis, globally reused, or cleanly interpretable by hand.

## Operational target

For an OV operator `A_h`, the broad target is a decomposition of the form

\[
A_h \approx \sum_j U_{hj} C_{hj} V_{hj}^{\top} + E_h,
\]

where each candidate compartment has a read subspace `V`, write subspace `U`, an internal transformation `C`, and an unexplained residual `E`.

For QK, the analogous components describe relations between query and key subspaces and therefore routing conditions rather than transported residual content.

A compartment is not merely an arbitrary matrix factorization. It must earn its existence through economical description and independent evidence.

## Two independent axes of structure

Keep **internal simplicity** and **cross-head reuse** separate.

A component may be:

1. internally simple and reused: a reusable motif;
2. internally simple but not reused: a bespoke clean mechanism;
3. weakly structured but statistically shared: a shared tendency that may or may not be mechanistically interesting;
4. neither: residual/unstructured under the tested languages.

Reuse is evidence, not a requirement for local structure.

## Why arbitrary basis simplicity is insufficient

For a single isolated matrix, SVD can always choose bases that diagonalize the map. Therefore simplicity observed only after completely free independent choices of read and write bases is not automatically meaningful.

One-off structure needs additional support, such as:

- a genuine spectral or multiscale boundary;
- stability under perturbation/bootstrap/optimization changes;
- a shorter code than matched random matrices under a declared coding language;
- clean separation from the rest of the operator;
- recurrence across independently fitted heads/layers/models;
- architectural producer/consumer anchoring;
- later, coherent activation/token behavior and causal effects.

Gauge freedom and identifiability must be analyzed explicitly before interpreting transformation cores.

## Primary selection principle: MDL / rate-distortion

The intended first-stage discovery principle is minimum description length / rate-distortion rather than a fixed number or size of compartments.

A model pays for:

- the number of compartments;
- compartment dimensions;
- read and write subspaces;
- transformation parameters or motif identities;
- deviations from shared prototypes;
- residual error at declared precision/distortion.

The comparison should expose the full complexity-versus-reconstruction curve rather than rely on one arbitrary continuous-weight coding constant.

A successful multidimensional decomposition should beat strong alternatives at equal description length, including appropriate versions of:

- truncated SVD;
- independent rank-one encodings;
- dense PCA/shared low-dimensional structure across heads;
- whole-matrix dictionaries;
- hard clustering;
- layer-only structure;
- fully bespoke per-head encodings.

Held-out performance must be evaluated on complete heads. If test-head-specific subspaces or coefficients are fitted, their description cost must be charged. Shared libraries selected during discovery remain frozen at test time.

## Null philosophy

Every positive structural claim must be compared with nulls that preserve increasingly strong generic properties that could otherwise explain the effect. The full discovery/selection procedure must be rerun on null data when the real procedure includes discovery.

Relevant null families include, depending on the experiment:

- rank/norm-matched random operators;
- exact-spectrum Haar rotations;
- within-layer read/write side re-pairings;
- smooth synthetic trajectories preserving spectra and realistic subspace motion;
- shuffled module-to-head assignments;
- layer-matched baselines;
- matched-flexibility controls with equal test-time degrees of freedom.

A visually strong or non-random structure is not itself evidence for the proposed mechanism.

## Validation hierarchy

Discovery, validation, and confirmation/test units must remain separate. Model class and relevant hyperparameters are selected without using the final test result. Entire-layer holdout is a harder generalization test where appropriate.

Positive claims should require some combination of:

- held-out compression/reconstruction advantage at equal complexity;
- advantage over relevant matched nulls;
- recurrence or independent replication where reuse is claimed;
- bootstrap/optimization stability;
- meaningful complete-operator variance or description-length contribution;
- replication in another model family;
- later architectural/activation/causal grounding.

Negative results are first-class outcomes and should state precisely which formulation was rejected and which broader alternatives remain open.

## Current decomposition program

The next central weight-only program is OV-first because OV directly matches the read-transform-write picture.

A candidate accounting is

\[
A_h = A_h^{\mathrm{reusable}} + A_h^{\mathrm{bespoke\ simple}} + A_h^{\mathrm{generic\ low\ rank}} + E_h.
\]

Across the population, also separate globally reusable, layer-local, and head-specific structure without misleadingly summing overlapping explained-variance quantities. Sequential residual fitting may be used for a preregistered accounting, with order-sensitivity checks later.

The intended execution order is:

1. Validate MDL accounting on synthetic matrices with known multidimensional compartments.
2. Run an engineering pilot on a subset of heads to debug the method, not to make the scientific claim.
3. Freeze the method.
4. Run all 48 final Pythia-70M OV heads with held-out heads/layers.
5. Replicate on all 144 GPT-2 small OV heads.
6. Extend the successful language, if any, to QK routing compartments.
7. Fit joint QK-OV rules.
8. Only then prioritize token/activation grounding and causal module tests.

Twelve to twenty-four heads are sufficient for engineering, not for the primary scientific conclusion.

## Complementary routes

If compression-first compartments show promise, important independent checks include:

- architecturally anchored decompositions using upstream producer and downstream reader geometry;
- co-moving coordinates for training trajectories rather than fixed ambient projectors;
- joint tensor/block-term decompositions across heads;
- multiscale transformation signatures;
- simple operator-algebra relations and compositions;
- recursive residual analysis.

Agreement between independently motivated methods is stronger evidence than a single flexible factorization.

## Grounding stage

After discovery is frozen, connect weight-only modules to the model:

- identify token/residual/upstream outputs occupying read subspaces;
- identify downstream readers/unembedding directions receiving write subspaces;
- characterize QK feature relationships that alter attention scores;
- combine QK routing with OV transport;
- identify examples where modules are active;
- selectively ablate/replace modules and test predicted effects.

The long-term objective is not merely a reconstruction percentage. It is an auditable accounting of which parts of attention weights are economically describable, which transformations recur, which are bespoke but clean, how modules compose through the architecture, and which discoveries correspond to actual computations.
