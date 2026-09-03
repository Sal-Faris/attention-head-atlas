# Checkpoint 0012: unsupervised contextual gating of QK channels

## Question

Do the compact rank-4 QK channels act uniformly on all inputs, or do
automatically discovered contextual input states select different channel
components?  This is the first direct test of the conditional-routing idea:

```text
query input class / key input class  ->  different QK channel mixture.
```

## Method

For each of the 17 heads incident to a recurrent rank-4 key-side family:

1. Fit its rank-4 QK channel on discovery events only.
2. Separately cluster the discovery residual inputs reaching the query and
   selected-key positions.  The procedure is PCA whitening followed by a
   diagonal Gaussian mixture; BIC chooses the number of mixture components
   from 1 through 8.
3. Assign held-out confirmation residual inputs to those discovery-fitted
   classes.
4. Let the channel component with largest absolute contribution be the
   event's dominant component.  Test its mutual information with input class
   on held-out events.
5. Build the null by shuffling dominant-component labels *inside each exact
   relative-offset bin*.  This controls the strongest obvious positional
   confound.  Apply Benjamini--Hochberg across all 34 query/key tests.

The full analysis uses up to eight components.  A preplanned coarser
sensitivity run repeats it with at most four components.

## Results

At the default resolution, 31 of 34 query/key input-class associations have
BH-adjusted q < 0.05.  With the coarser 3--4 class mixtures, 27 of 34 remain
significant.  Thus the association is not solely a fragile fine partition.

The effects are heterogeneous.  The largest stable excess mutual information
is 0.102 for the key input of L3H7; L3H5 key (0.074), L5H1 key (0.067), and
L2H0 query (0.052) are also prominent.  Many smaller effects are statistically
reliable because the confirmation set contains roughly 3,500 matched events
per head, so magnitude matters as well as p-values.

Representative held-out inputs give the result a plausible mechanistic shape:

- **L3H7 key:** a broad class whose representatives are newlines in diverse
  text/code contexts is 94.9% dominated by one channel component; a smaller
  class containing document-opening text shifts substantial mass to a second
  component (33.1%).
- **L2H0 query:** an ordinary-prose class and smaller punctuation/URL/markup
  classes have visibly different dominant-component distributions.  The latter
  include commas, question marks, slashes, brackets, and code-like spacing.

These representative strings were not used for clustering or testing.  They
are only a post hoc display of residual-state classes found from activations.

## Interpretation

This is evidence that the validated QK channels are **conditionally gated by
contextual residual inputs**.  It directly supports the possibility raised in
the research discussion: the same head can use different low-rank routing
components for different kinds of input state, rather than having one global
role.

However, it is not yet evidence for a neat finite set of universal token
types.  BIC often reaches the current eight-class ceiling, and some mixture
components fitted on discovery have negligible mass on confirmation.  The
better current picture is a continuous residual-state geometry for which a
coarse partition exposes stable conditional changes.  Formatting, document
boundaries, and token-shape features are visible examples; semantic classes
remain unestablished.

The next high-value step is to factor the *joint query-class × key-class →
channel-component* contingency tensor, using held-out tests and a stronger
control that preserves both offset and document.  That can distinguish simple
one-sided gating from genuinely conditional query--key rules.

## Reproduction

```powershell
python scripts/test_qk_channel_input_classes.py
python scripts/test_qk_channel_input_classes.py --max-classes 4 --output results/pythia-70m-deduped/qk_channel_input_classes_k4_sensitivity.json
python scripts/profile_qk_input_classes.py
```

Primary compact outputs:

- `results/pythia-70m-deduped/qk_channel_input_classes_v1.json`
- `results/pythia-70m-deduped/qk_channel_input_classes_k4_sensitivity.json`
- `results/pythia-70m-deduped/qk_channel_input_class_profiles_v1.json`
