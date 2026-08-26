# Experiment index

This file is a compact anti-duplication index. Detailed evidence remains in `docs/checkpoints/`, machine-readable results, and Git history.

## Recording rule

Before starting a new scientific experiment:

1. Search this index.
2. Search `docs/HYPOTHESIS_LEDGER.md`.
3. Search all relevant files under `docs/checkpoints/` for equivalent objects, nulls, splits, metrics, and conclusions.
4. If a related experiment exists, state what assumption or discriminator is genuinely new.

After an experiment completes, add a compact entry here even if the result is negative or the implementation failed.

## Seeded entry

### E0026 — Fixed QK reducing subspaces across training

- **Status:** Complete; negative for the strong compartment claim.
- **Hypothesis:** H001.
- **Checkpoint:** `docs/checkpoints/0026-qk-reducing-subspaces.md`.
- **Commit:** `79e01eb579dbe4ba512c0074221dbfa49f546871`.
- **Population:** all 48 processed Pythia-70M-deduped QK heads, eight checkpoints.
- **Design:** fixed learned read/write reducing subspaces; primary 64-dimensional support split 32/32 plus multiresolution audit; forward-held-out validation and confirmation.
- **Nulls:** independent exact-spectrum Haar; within-layer side pairing; smooth singular-frame trajectories preserving checkpoint spectra and exact adjacent frame overlaps.
- **Key result:** real beats weaker nulls but loses to smooth-frame null at confirmation in both splits.
- **Audits:** synthetic recovery, gauge invariance, inactive-kernel handling, deterministic seeds, compact/dense equivalence, iteration-depth audit, multistart audit.
- **Interpretation:** apparent fixed compartments are explained by generic smooth low-rank frame motion; temporal persistence and matched Q/K co-development remain real.
- **Do not repeat unless:** testing a materially different hypothesis such as co-moving, oblique/overlapping, architecturally anchored, or cross-head reusable transformations.

## Historical backfill

The repository contains checkpoint documents `0001` through `0026`. This index was introduced after those experiments. A future bookkeeping task should summarize the earlier checkpoints into this file without changing their conclusions. Until that backfill is complete, agents must search the checkpoint directory directly before claiming an experiment is new.

## Entry template

```markdown
### E#### — Short title

- **Status:** planned / running / complete / failed engineering / inconclusive.
- **Hypothesis:** H###.
- **Frozen contract:** path.
- **Checkpoint:** path.
- **Commit:** SHA.
- **Population/data:** ...
- **Discovery/validation/test split:** ...
- **Estimator/model class:** ...
- **Baselines/nulls:** ...
- **Selection/multiplicity:** ...
- **Key result:** ...
- **Robustness/audits:** ...
- **Interpretation:** ...
- **Rules out:** ...
- **Does not rule out:** ...
- **Repeat only if:** ...
```
