# Agent instructions

This repository is an AI/interpretability research project, not a generic software project. Scientific correctness takes priority over implementation convenience.

## Canonical project state

Before doing research work, read:

1. `RESEARCH_PROGRAM.md` for the enduring scientific objective and definitions.
2. `CURRENT_RESEARCH_STATE.md` for the current best understanding.
3. `docs/HYPOTHESIS_LEDGER.md` for supported, open, and rejected hypotheses.
4. Relevant files in `docs/checkpoints/` before proposing an experiment that may overlap previous work.

Do not infer current beliefs from an old checkpoint alone. Later checkpoints and the current-state documents supersede earlier interpretations.

## Engineering environment

The package requires Python 3.11+. The project uses pytest and Ruff.

Typical setup:

```powershell
python -m pip install -e ".[dev,analysis,models]"
```

Verification:

```powershell
python -m pytest
python -m ruff check .
```

There is currently no project type checker configured in `pyproject.toml`; do not claim type checking passed unless one was explicitly run.

## Scientific guardrails

- Never silently replace the research question with an easier proxy.
- If an implementation choice changes the estimand, null distribution, held-out protocol, encoding cost, model class, or interpretation, stop and escalate the choice to the research lead.
- Distinguish exploratory, validation, and confirmatory analyses. Do not tune on a held-out confirmation/test result and then continue to call it held out.
- Run the complete discovery/selection procedure on null data when the real procedure includes discovery or selection.
- Do not treat non-randomness as mechanistic evidence without a null that preserves plausible generic structure.
- Treat basis/gauge freedom and identifiability as first-class concerns for matrix and subspace decompositions.
- For continuous MDL/rate-distortion work, expose sensitivity to distortion/precision rather than presenting one arbitrary coding constant as canonical.
- If an optimizer repeatedly hits its iteration cap or otherwise fails a convergence criterion, audit iteration depth and initialization before interpreting a close result.
- Correct for multiplicity when making population or per-head discovery claims.
- State explicitly what a result rules out and what it does not rule out.

## Implementation rules

- Preserve existing numerical conventions and deterministic seeds unless a research contract explicitly changes them.
- Do not weaken or delete a test merely to make a change pass.
- Add synthetic recovery/invariance tests when implementing a new mathematical estimator where such tests are meaningful.
- Prefer auditable, exact or equivalence-tested implementations over opaque optimizations.
- Keep raw numerical artifacts separate from narrative interpretation.
- Before declaring work complete, inspect the diff and report exactly which tests and analyses were run.
- Never push automatically. A human decides when a branch is ready to push/merge.

## Delegation policy

Routine implementation should be delegated to cheaper workers when the scientific contract is already frozen. Escalate to stronger models when:

- two materially different implementation/debugging attempts fail;
- a mathematical ambiguity appears;
- the implementation exposes a previously unrecognized scientific choice;
- results contradict the expected invariants or the frozen interpretation table;
- a conclusion depends on subtle identifiability, statistical, MDL, or causal reasoning.

Use the project skills `research-contract`, `scientific-adversary`, and `experiment-memory` when relevant.
