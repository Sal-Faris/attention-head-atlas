---
name: Experiment Memory
description: Prevent duplicate research by checking prior hypotheses/checkpoints before work and recording failures/results afterward
---

# Experiment memory workflow

## Before proposing or implementing an experiment

1. Read `CURRENT_RESEARCH_STATE.md`.
2. Read `docs/HYPOTHESIS_LEDGER.md`.
3. Read `docs/experiments/EXPERIMENT_INDEX.md`.
4. Search `docs/checkpoints/` for the mathematical object, estimator, null family, metric, and conclusion being proposed.
5. If relevant, inspect the exact Git commit/result artifact rather than relying on memory.
6. Decide whether the proposal is genuinely new, a stronger/weaker version of an old test, or the same experiment in different language.

If it overlaps a rejected/completed experiment, state exactly which assumption has changed and why the prior result does not already answer the question.

## During execution

Keep implementation facts separate from scientific interpretation. Record deviations from the frozen contract. Do not erase failed attempts that reveal a reusable implementation or methodological lesson.

## After execution

Update or create an experiment-index entry containing:

- hypothesis ID;
- status, including failed/inconclusive results;
- frozen contract and checkpoint paths;
- exact commit;
- data/model population and held-out units;
- estimator/model class;
- baselines/nulls;
- selection/multiplicity procedure;
- main numerical result;
- robustness/audits;
- interpretation;
- what it rules out;
- what it does not rule out;
- explicit `repeat only if` condition.

Then update `CURRENT_RESEARCH_STATE.md` and `docs/HYPOTHESIS_LEDGER.md` only if the result changes the current best understanding.

## Anti-forgetting rule

Negative and failed experiments are first-class memory. Never omit a result merely because it was unexciting. The purpose of this record is partly to stop the lab from spending days rediscovering an old dead end.
