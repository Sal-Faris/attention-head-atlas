---
name: Scientific Adversary
description: Stress-test an AI/interpretability experiment for hidden mathematical, statistical, MDL, optimization, and interpretive failure modes
---

# Scientific adversary

Use this skill when reviewing a proposed experiment or interpreting a result.

## 1. Identifiability and invariance

Ask:

- What transformations of the parameters leave the observable model unchanged?
- Can free basis choices make the claimed simplicity true for every matrix?
- Is the recovered object unique, identifiable only up to a gauge, or fundamentally non-identifiable?
- Are comparisons invariant under permitted reparameterizations?
- Is a proposed boundary intrinsic or created by the optimizer/coordinate system?

## 2. Selection and held-out logic

Ask:

- What information influenced the model class, hyperparameters, resolution, or stopping rule?
- Is the final confirmation/test unit still untouched?
- Are complete heads/layers held out when the claim is about generalization across heads/layers?
- Are test-time fitted parameters charged and matched in baselines?

## 3. Null adequacy

Construct a hierarchy preserving progressively stronger generic structure. A positive result against random orientation may still be explained by low rank, spectra, layer structure, smooth training motion, or fitting flexibility.

Run the complete selection/discovery procedure on null data when the real analysis includes selection.

## 4. Statistics

Check the true independent population unit, null repetition count, finite-null p-value resolution, confidence intervals, multiple comparisons, bootstrap scheme, and whether per-head findings are being promoted to population discoveries.

## 5. MDL/rate-distortion accounting

Check whether any parameters are transmitted for free; whether precision/distortion is declared; whether shared-library cost is amortized fairly; whether bespoke and shared encodings use comparable languages; and whether the conclusion holds across a rate-distortion curve rather than one convenient constant.

## 6. Optimization

If the result depends on fitting, ask whether convergence was reached, whether real and null fits receive identical budgets, whether close differences survive more iterations/multistart, and whether numerical shortcuts were equivalence-tested.

## 7. Mechanistic interpretation

Separate these claims:

1. structure exists;
2. structure exceeds simple random geometry;
3. structure exceeds stronger matched generic structure;
4. structure is stable/reusable;
5. structure corresponds to a computation;
6. the computation is causally relevant.

Do not skip levels.

## Output format

For each objection return:

- severity: blocker / major / minor;
- exact failure mode;
- why it could change the conclusion;
- whether the current design already controls it;
- the smallest discriminating fix/test.

Prefer a decisive falsification test over adding many weak analyses.
