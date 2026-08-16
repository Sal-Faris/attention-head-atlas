# Research Protocol v2: From Head Clusters to Operator Motifs

Status: **frozen after exploratory GPT-2-small geometry and before the full
published-family benchmark**  
Version: 0.2  
Date: 2026-08-16

This protocol supersedes the *direction* of the initial analysis contract but
does not replace its historical record. The initial contract correctly treated
discrete clusters, continuous geometry, reusable atoms, and
activation-conditional structure as competing hypotheses. Exploratory results
now justify changing the primary emphasis from global clustering to local
functional retrieval and compositional motifs.

## Revised research question

To what extent can unsupervised, prompt-independent geometry of attention-head
QK and OV operators:

1. recover independently established mechanistic families;
2. reveal previously unknown local operator families;
3. distinguish discrete types from continuous factors and sparse mixtures; and
4. predict functional and causal measurements beyond layer and singular
   spectrum?

The project does not assume that an architectural head is an atomic mechanism.
A "periodic table" is one possible outcome, not the premise.

## Current exploratory evidence

The following observations were made on GPT-2 small and are development
evidence, not confirmation:

- QK and OV normalized-Frobenius populations strongly reject independently
  rotated, per-head spectrum-matched nulls.
- Average-linkage clustering has low partition silhouette and tends to split
  outliers from a large background population rather than produce balanced
  families.
- QK has a strong depth-related leading direction and lower population
  participation dimension than OV.
- OV is more diffuse in its leading coordinates and is weakly organized by
  layer.
- The independently documented copy-suppression heads L10H7 and L11H10 are
  mutual nearest neighbours in both QK and OV normalized-Frobenius geometry.
  Their OV pair lies in the closest 0.1263% of all pairs; their QK pair lies in
  the closest 3.3994%.

The last observation was noticed after inspecting the geometry. It is an
encouraging retrospective external check, not a preregistered test.

## Competing structural hypotheses

- **S0 - nuisance/null geometry:** apparent organization is explained by layer,
  rank, norm, singular spectrum, or generic low-rank concentration.
- **S1 - local functional neighbourhoods:** related mechanisms are locally
  close, but the full population has no exhaustive partition.
- **S2 - discrete families:** a stable, null-separated partition assigns most
  heads to a small number of types.
- **S3 - continuous factors/manifold:** heads vary along reproducible continuous
  coordinates without natural boundaries.
- **S4 - sparse motif mixtures:** heads are sparse combinations of recurring
  operator atoms.
- **S5 - context-conditioned structure:** static operators are insufficient;
  useful organization appears only under activation-weighted or feature-space
  representations.

S1-S5 are not failures of clustering. They are different scientific models.

## Unit of discovery

The initial unit remains one architectural attention head, with two paired
views:

\[
M_{QK}^{\ell,h}=W_Q^{\ell,h}(W_K^{\ell,h})^\top,
\qquad
M_{OV}^{\ell,h}=W_V^{\ell,h}W_O^{\ell,h}.
\]

The products remove the internal head-basis gauge. QK and OV are first analyzed
separately to identify which view carries a relation, then jointly because a
head's mechanism depends on both where it attends and what it writes.

Later discovery units may be operator atoms or cross-head attention features.
Results must not silently switch between these levels.

## Frozen baseline representations

1. Frobenius-normalized QK with normalized Frobenius distance.
2. Frobenius-normalized OV with normalized Frobenius distance.
3. Equal-total-variance joint QK-OV PCoA coordinates, using every positive
   coordinate rather than the 2D visualization.

Secondary representations are evaluated as alternatives, not tuned until they
match known labels:

- singular-spectrum distance;
- read/write subspace projector distance;
- vocabulary/logit-projected OV;
- token/position-projected QK;
- activation-weighted QK score and OV action distances.

## External functional benchmark

Published GPT-2-small mechanistic labels will be transcribed with an exact
source, scope, and confidence. Candidate sources include:

- the seven IOI circuit classes identified through causal intervention;
- negative/copy-suppression heads;
- induction and previous-token heads with published behavioral scores.

Published labels are incomplete. Therefore an unlabelled head is *unknown*, not
a negative example. Evaluation emphasizes retrieval among known positives and
uses separately defined controls.

The frozen baseline distances will not be tuned on these labels. Any learned
combination of views requires a development/validation split by whole mechanism
family, followed by confirmation on another model or checkpoint.

## Primary evaluation: local retrieval

For every known family with at least two members, compute:

- percentile rank of within-family pairs among all pairs;
- reciprocal rank of the nearest same-family neighbour;
- precision and recall among the top-k neighbours;
- mean average precision with unknown labels excluded from negative claims;
- enrichment relative to layer-matched and spectrum-matched controls.

Aggregate evidence uses family-balanced averages so large published classes do
not dominate. Significance is estimated by label permutations preserving class
sizes and, separately, by layer-stratified permutations.

Local retrieval is primary because related mechanisms may occupy islands inside
an otherwise continuous, overlapping population.

## Secondary evaluation: global structural models

The same frozen representations compare three model classes:

1. **Discrete:** average-linkage and k-medoids partitions.
2. **Continuous:** PCoA/PCA factors and local intrinsic dimension.
3. **Sparse mixture:** dictionary learning or sparse coding on losslessly
   retained population coordinates.

Model selection considers:

- matched-null-adjusted fit;
- held-out reconstruction;
- stability under reasonable perturbations;
- incremental prediction of external functional diagnostics;
- complexity or description length.

No cluster count is selected from a 2D visualization. UMAP, if used, is display
only.

## Null and nuisance hierarchy

Every structural statistic is compared with the strongest feasible control:

1. label permutation preserving family sizes;
2. layer-stratified label permutation;
3. per-head singular-value-matched random singular directions;
4. feature/projection subsampling or numerical perturbation;
5. random-initialization and cross-seed models when available.

Nuisance predictors include layer, normalized depth, Frobenius norm, effective
rank, spectral norm, and leading singular-energy fractions. Functional
prediction must beat a nuisance-only model.

## Decision gates

- **Pursue discrete taxonomy** only if partitions are stable, substantially beat
  matched nulls, avoid singleton peeling, and retrieve external families.
- **Pursue continuous atlas** if leading factors beat nulls and predict
  diagnostics while partition stability remains weak.
- **Pursue motif mixtures** if sparse atoms reconstruct held-out heads better
  than clusters at comparable complexity and their coefficients predict
  mechanisms.
- **Pivot to activation-weighted geometry** if static representations fail to
  retrieve known families beyond nuisance variables.
- **Call a motif mechanistic** only after targeted activation evidence and a
  causal intervention behave as predicted.

## Confirmation standard

GPT-2-small is now an explored development population. Strong claims require at
least one of:

- a prespecified test on GPT-2 Medium or another architecture;
- recurrence across untouched training checkpoints or random seeds;
- prospective prediction of a previously unstudied head relation followed by
  causal validation.

## Near-term sequence

1. Compare real PCoA eigenspectra with repeated spectrum-matched null
   populations.
2. Transcribe and verify the external GPT-2-small functional benchmark.
3. Evaluate frozen QK, OV, and joint-view local retrieval.
4. Quantify layer and spectral nuisance effects in the complete geometry.
5. Compare global partition and continuous-factor evidence against nulls.
6. Attempt sparse motif learning only if the first five steps support reusable
   population structure.
7. Validate the strongest static relations using token/logit projections and
   real activations.

## Interpretation discipline

- A close pair is a candidate relation, not a shared function.
- A published same-family pair recovered post hoc is validation evidence, but
  not confirmation.
- A cluster is not a mechanism.
- A dictionary atom is not unique merely because optimization converged.
- A static composition edge is potential information flow, not causal use.
- Names are hypotheses assigned only after unsupervised discovery is frozen.

## Primary literature anchors

- Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021):
  https://transformer-circuits.pub/2021/framework/
- Wang et al., *Interpretability in the Wild: a Circuit for Indirect Object
  Identification in GPT-2 small* (2022): https://arxiv.org/abs/2211.00593
- Olsson et al., *In-context Learning and Induction Heads* (2022):
  https://arxiv.org/abs/2209.11895
- McDougall et al., *Copy Suppression: Comprehensively Understanding an
  Attention Head* (2023): https://arxiv.org/abs/2310.04625
- Anthropic, *Progress on Attention* (2025; preliminary research update):
  https://transformer-circuits.pub/2025/attention-update/

