# Current research state

_Last seeded from research history through checkpoint 0026 (`79e01eb579dbe4ba512c0074221dbfa49f546871`). Update this file when a result changes the current best understanding._

## What is established

### Attention-weight geometry is non-random

Earlier checkpoints found stable population geometry and recurrent structure in attention operators. Those observations motivate further decomposition but do not themselves establish discrete mechanistic compartments.

### Trained QK trajectories have strong temporal persistence

The checkpoint-0026 trajectory study showed that real QK operators across training are much more persistent/reducible than independently oriented exact-spectrum trajectories.

### Matched query/key trajectories contain genuine paired structure

Replacing one side of a head's query/key trajectory with another same-layer head substantially weakens the reducing-subspace fit. The actual Q/K sides therefore co-develop in a nontrivial way that is not explained by their separate marginal spectra alone.

## Important negative result

### Fixed orthogonal trajectory-stable QK compartments are not supported beyond smooth low-rank dynamics

Checkpoint 0026 tested fixed read/write projectors across eight Pythia-70M-deduped training checkpoints for all 48 QK heads. The transformations inside the fitted blocks were unrestricted.

At the frozen 64-dimensional active support with a 32/32 split, real trajectories looked highly structured against random projectors and weaker nulls. However the strong smooth singular-frame null preserved every checkpoint spectrum and exact adjacent-checkpoint left/right frame overlap matrices.

Against that null, real-minus-null confirmation gain was negative in both splits:

- primary/main: approximately -2.04 percentage points, finite-null p = 1.00;
- late sensitivity: approximately -0.75 percentage points, finite-null p = 1.00.

All 19 smooth-null population means exceeded the real confirmation mean in both splits.

Iteration-depth and multistart audits did not reverse the sign. The result is therefore not explained by the initial 60-iteration cap or an obvious local-start artifact.

**Interpretation:** low rank plus smooth movement of singular frames is sufficient to create apparently convincing fixed fitted compartments. The population-level claim of additional fixed, orthogonal, trajectory-stable ambient compartments is rejected in the tested formulation.

**This does not rule out:**

- co-moving subspaces;
- overlapping or oblique compartments;
- transformations recurring across heads rather than checkpoints;
- architecturally anchored producer-to-reader channels;
- input-conditional operations;
- multi-head paths and MLP-mediated structure.

L3H5 remains an exploratory follow-up candidate but does not survive a 48-head multiple-testing claim with only 19 null repetitions.

## Current priority

The next central program is **variable-dimensional OV compartment discovery using MDL/rate-distortion**, with internal simplicity and cross-head reuse treated as independent axes.

The engineering pilot may use a subset of heads, but the primary scientific test should use all 48 final Pythia-70M heads and then replicate on all 144 GPT-2 small heads.

The intended first question is broader than "is there a small shared vocabulary?":

> How much of each OV operator can be decomposed into clean, low-complexity local transformations, and among those transformations how much is globally reused, layer-local, or genuinely bespoke?

The method should compare equal-bit reconstruction/generalization against strong baselines and nulls, freeze shared libraries before untouched test heads, charge test-head fitted parameter costs, and report the rate-distortion/Pareto curve rather than a single privileged MDL constant.

## Near-term scientific risks to resolve before scaling

- Precisely define coding languages and continuous-parameter precision so comparisons are fair.
- Ensure free read/write basis fitting does not make "simple transformation" vacuous; bespoke modules require independent support beyond self-diagonalization.
- Ensure test-head fitting flexibility is matched in baselines and fully charged in description length.
- Separate local simplicity from reuse rather than allowing recurrence to become a hidden existence criterion.
- Define nulls that distinguish generic low rank, recurring spectra, read/write coupling, layer structure, and genuine recurrence.
- Validate synthetic recovery for known multidimensional compartments before trusting real-data decompositions.
- Keep the engineering pilot distinct from the frozen primary experiment.

## Research-record rule

Before proposing a new experiment, search the checkpoint history and `docs/HYPOTHESIS_LEDGER.md`. If a proposed experiment is equivalent to a previous one under a relabeling or weaker/stronger formulation, explain the difference explicitly before spending compute.

When a new experiment completes, record:

- hypothesis tested;
- exact implementation/commit;
- models/data/splits;
- nulls and baselines;
- selection and multiplicity rules;
- main numerical result;
- robustness/audits;
- interpretation;
- what it rules out;
- what it does not rule out;
- whether/when it should be repeated.
