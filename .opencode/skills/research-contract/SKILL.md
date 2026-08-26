---
name: Research Contract
description: Freeze a scientific experiment into an implementation-ready contract without silently changing the research question
---

# Research contract workflow

Use this before substantial implementation of a new scientific experiment.

## Required contract

Write or identify a frozen contract containing:

1. **Question and hypothesis** — the exact claim being discriminated.
2. **Mathematical object** — equations, domains, gauges/invariances, units, normalization.
3. **Operational definition** — what counts as the proposed structure and what does not.
4. **Identifiability analysis** — basis freedoms, degeneracies, trivial representations, equivalent parameterizations.
5. **Discovery/validation/test protocol** — what data may influence model selection and what remains untouched.
6. **Estimator/model class** — parameters, constraints, optimization objective, initialization, convergence criterion.
7. **Complexity accounting** — especially any test-time fitted parameters and continuous precision for MDL/rate-distortion.
8. **Baselines and null hierarchy** — progressively preserve generic properties that could explain a positive result.
9. **Statistics** — population unit, repetition unit, p-value construction, uncertainty, multiplicity.
10. **Correctness tests** — synthetic recovery, invariance/equivalence, determinism, known edge cases.
11. **Robustness gates** — optimizer depth, multistart, sensitivity analyses, alternative ordering where relevant.
12. **Decision table** — expected interpretations for positive, null-like, unstable, and contradictory outcomes.
13. **Required artifacts** — machine-readable results, figures, checkpoint narrative, exact commands/commit.

## Freeze rule

After the contract is frozen, implementation agents may make ordinary engineering choices only. Any change that affects the mathematical object, selection procedure, null, statistical test, encoding cost, or interpretation must return to the research lead before execution.

If results cause a change to the method, treat the revised method as exploratory unless a fresh untouched confirmation unit remains.
