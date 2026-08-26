# Hypothesis ledger

This ledger records the current scientific status of major hypotheses. It is not a chronology; later evidence should update earlier entries. Checkpoint documents remain the detailed evidence trail.

## H001 — Fixed orthogonal trajectory-stable QK compartments

**Status:** Rejected in the tested population-level formulation.

**Claim:** A head's QK trajectory contains fixed ambient read/write subspaces whose block structure exceeds what generic low-rank training dynamics would create.

**Evidence:** `docs/checkpoints/0026-qk-reducing-subspaces.md`, commit `79e01eb579dbe4ba512c0074221dbfa49f546871`.

**Critical discriminator:** Smooth singular-frame null preserving checkpoint spectra and adjacent frame-overlap matrices.

**Result:** Real confirmation performance is worse than the smooth-frame null in both primary and late splits; convergence/multistart audits do not reverse the sign.

**Do not repeat unless:** the proposed compartments materially change (for example co-moving, oblique/overlapping, architecturally anchored, or cross-head rather than fixed across checkpoints).

---

## H002 — Trained QK trajectories have non-random temporal persistence

**Status:** Supported.

Real trajectories are much more temporally persistent/reducible than independently oriented exact-spectrum trajectories. This is genuine structure but is not sufficient evidence for fixed compartments.

---

## H003 — Matched query/key trajectories contain paired reusable structure

**Status:** Supported as a population-level geometric finding; mechanistic interpretation remains open.

Actual Q/K side pairings fit together substantially better than within-layer side-mismatched trajectories, indicating nontrivial co-development beyond marginal spectra.

---

## H004 — Variable-dimensional OV operators contain internally simple multidimensional compartments

**Status:** Open; current priority.

**Claim:** At some rate-distortion scales, moderate multidimensional read-transform-write components provide a shorter held-out description than treating the head as one dense object or as independent singular/rank-one channels.

**Required evidence:** synthetic recovery, equal-bit baseline advantage on untouched heads, matched null advantage, stability, and meaningful complete-operator contribution.

**Important caveat:** free choice of read/write bases can trivialize internal transformation simplicity. One-off bespoke compartments require evidence beyond self-diagonalization.

---

## H005 — Some clean OV compartments are bespoke rather than reused

**Status:** Open.

Internal simplicity and cross-head reuse are independent. A component may survive as a bespoke structured module if it earns a shorter description and has additional evidence such as stability, spectral separation, matched-null advantage, or architectural grounding.

---

## H006 — A reusable vocabulary of OV transformation motifs exists across heads

**Status:** Open; lower prior confidence than H004/H005.

A positive result requires frozen-library generalization to held-out complete heads/layers with all test-head parameter costs charged. Similar spectra alone are not sufficient if independent read/write rotations erase stronger notions of shared transformation.

---

## H007 — Architecturally anchored producer-to-consumer geometry reveals modules missed by unconstrained weight-only factorization

**Status:** Open; high-value complementary route.

Candidate read/write directions may become meaningful when weighted by which upstream components can produce them and which downstream components can consume them. This can provide evidence for bespoke modules even without recurrence.

---

## H008 — QK structure is better represented in co-moving rather than fixed ambient coordinates

**Status:** Open; motivated directly by H001's smooth-frame result.

Any test must retain a smooth-trajectory null so that moving-coordinate structure is not merely a re-expression of ordinary smooth singular-frame motion.

---

## H009 — Joint tensor/block-term decomposition recovers compatible modules independently of per-head MDL discovery

**Status:** Exploratory independent-check hypothesis.

Agreement with a separately motivated joint decomposition would strengthen evidence that a discovered module family is not an artifact of one flexible fitting language.

## Maintenance rules

- Add links to checkpoints/results/commits whenever a status changes.
- Use `supported`, `rejected in tested formulation`, `open`, or `exploratory`; avoid absolute language when a broader hypothesis remains possible.
- Record the strongest alternative explanation that survived each experiment.
- If a new proposal resembles a rejected hypothesis, state exactly which assumption has changed before opening a new experiment.
