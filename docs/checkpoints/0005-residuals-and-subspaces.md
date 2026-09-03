# Checkpoint 0005: residual geometry and factor subspaces

## Question

After the first sparse-operator dictionaries left most centered energy
unexplained, this checkpoint asks two deliberately separate questions:

1. Do dictionary residuals retain population structure beyond an isotropic
   direction null?
2. Do reusable structures become clearer when complete QK/OV operators are
   separated into their leading query/key or read/write subspaces?

Neither analysis uses functional head labels.

## Methods

### Dictionary-residual null

For each compact (8 atoms) and selected optimal (32 atoms) QK, OV, and joint
dictionary, the exact PCoA-coordinate residual is

\[
r_i = (x_i - \bar{x}) - c_i D.
\]

Residual directions are compared with 20 independent null populations. Each
null preserves every residual vector's norm while replacing its direction with
an independent isotropic direction in the same coordinate space. This is the
appropriate null for the dictionary's exact PCoA geometry; it is not described
as an operator singular-spectrum null.

The audit separately examines all discovery checkpoints and the final
checkpoint. The latter removes the repeated-head trajectories that can make a
population appear structured even when different final heads do not share
residual directions.

### Exact factor subspaces

Each stored operator has the exact skinny factorization

\[
M = L R^T.
\]

QR decomposition of the factors followed by an SVD of the resulting 64 by 64
core produces the exact thin SVD without materializing a 512 by 512 matrix.
Leading ranks 4, 8, 16, 32, and 64 are compared with normalized chordal
projector distance. The two sides mean query/key for QK and value-read/output-
write for OV.

Final-checkpoint structure is compared with 20 populations of independent
Haar-random subspaces at matched residual width and rank. Layer association is
tested with 999 PERMANOVA permutations. Adjacent-checkpoint matching asks
whether each subspace identifies the same layer/head trajectory.

## Results

### Residual structure is mainly longitudinal

The 32-atom dictionaries capture 53.6% of final QK centered energy, 48.4% of
final OV energy, and 51.2% of final joint energy. The compact dictionaries
capture only 20.4%, 16.8%, and 17.5%, respectively.

Across all discovery checkpoints, residual nearest neighbours remain much
closer than isotropic nulls. For compact QK, OV, and joint dictionaries, the
real/null mean-nearest-distance ratios are 0.77, 0.72, and 0.73. This signal
weakens but remains present for the 32-atom dictionaries.

At the final checkpoint, however, QK and joint residual nearest-neighbour
distances are indistinguishable from or slightly larger than the isotropic
null. OV compact residuals retain a small local-neighbour excess, but the
optimal OV residual does not. Final residual participation dimensions are also
at or above the null expectation.

The parsimonious interpretation is that much of the residual structure across
the full checkpoint population is persistent head identity through training,
not an omitted family of shared final-head matrix atoms.

### Factor subspaces are non-random

Every tested final query, key, read, and write subspace has lower participation
dimension and closer nearest neighbours than its matched Haar-random null at
the resolution of 20 repetitions (one-sided p = 1/21).

At rank 16, which captures 57% of mean QK operator energy and 54% of mean OV
energy, participation dimensions are:

| View and side | Real | Haar null |
| --- | ---: | ---: |
| QK query | 42.3 | 47.0 |
| QK key | 42.9 | 47.0 |
| OV read | 44.7 | 47.0 |
| OV write | 43.3 | 47.0 |

OV's rank-4 write side is especially concentrated: participation dimension
38.9 versus 47.0 under the null, with mean nearest-neighbour distance 0.86
versus 0.99.

Query and key geometry are strongly coupled: their pairwise-distance Spearman
correlation rises from 0.62 at rank 4 to 0.86 at rank 64. OV read and write
geometry are substantially less coupled, with correlations from 0.25 to 0.37.
This is the first direct evidence in the pilot that OV read and write primitives
may recombine more independently than complete OV matrices recur.

Layer explains 12.6% to 19.7% of final subspace geometry across the tested
views and ranks (permutation p = 0.001). Adjacent-checkpoint subspaces identify
their own head trajectories with mean accuracy between 97.6% and 100%.

## Interpretation and next gate

The results argue against spending the next phase only on larger whole-matrix
dictionaries. Their residual structure is predominantly longitudinal, while
the two-sided subspace representation exposes non-random cross-head geometry.

The most economical next analysis is therefore a read/write family and
co-occurrence model, beginning with low-rank OV write subspaces. A claimed
reusable primitive must still pass bootstrap stability, cross-layer reuse, and
functional or causal validation. QK should initially be treated as paired
query/key geometry rather than as freely recombined sides.

## Reproduction

```powershell
python scripts/audit_dictionary_residual_null.py
python scripts/analyze_factor_subspaces.py
```

Primary outputs:

- `results/pythia-70m-deduped/dictionary_residual_null.json`
- `results/pythia-70m-deduped/dictionary_residual_null.png`
- `results/pythia-70m-deduped/factor_subspace_atlas.json`
- `results/pythia-70m-deduped/factor_subspace_atlas.png`
