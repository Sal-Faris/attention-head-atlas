# Checkpoint 0006: stable factor-subspace neighborhoods

## Question

Checkpoint 0005 established that query, key, read, and write subspaces are
non-random. This checkpoint asks whether that structure is best described as
stable discrete families, persistent local neighborhoods, or a layer-driven
artifact. It also tests whether the two sides of QK and OV pair non-randomly
after preserving their layer structure.

No activation data or functional head labels are used.

## Fixed analysis

The audit uses ranks 4, 8, and 16 and the final three stored checkpoints
(`step16000`, `step64000`, and `step143000`). For each side separately:

- average-linkage cuts from 2 through 10 clusters are considered;
- the final-checkpoint cut is selected only by maximum silhouette;
- 200 repetitions remove 20% of heads and compare the induced cut with the
  full-population cut using adjusted Rand index (ARI);
- temporal cluster and three-nearest-neighbour recurrence are compared with
  499 head-identity shuffles performed independently within each layer;
- cluster silhouette is compared with 49 populations of independent
  Haar-random subspaces at matched width and rank.

Cross-layer nearest-neighbour recurrence forbids neighbours from the same
layer. Query/key and read/write coupling is measured by distance Spearman
correlation, neighbour overlap, and adjusted mutual information (AMI). Their
nulls permute one side within each layer, preserving layer effects while
breaking the actual pairing of the two sides inside a head.

## Results

### The population is structured but not cleanly separated

Every selected silhouette exceeds every matched Haar realization
(one-sided p = 1/50), but the absolute silhouettes are only 0.009 to 0.051.
The selected cuts usually contain 7 to 10 groups and several singletons.
Most QK cuts contain one background group of 20 to 37 heads plus small
satellites. OV cuts are more distributed, although their separation remains
weak.

The strongest silhouette is the rank-4 OV write view at 0.051. This is still
far below what would justify treating the cut as a clean taxonomy. The
subsampling ARIs of 0.65 to 0.98 show that the weak partitions are not simply
destroyed by removing a few items, but stability does not make the gaps large.

The appropriate current description is therefore **stable local structure and
specialist islands in a mostly continuous population**, rather than a periodic
table of sharply separated head species.

### Specific neighborhoods persist beyond layer effects

Cross-layer three-nearest-neighbour overlap across the final three checkpoints
ranges from 0.365 to 0.573. The corresponding within-layer-shuffle expectations
range from 0.120 to 0.196; all twelve view/rank comparisons have one-sided
p = 1/500.

Thus the signal is not only a smooth difference between layers. Particular
heads in different layers repeatedly remain near one another. Many of the
strongest recurrent pairs occur in adjacent layers, so developmental depth
gradients remain a plausible part of the explanation and should be retained as
a covariate during functional tests.

Hard-cluster recurrence is less uniform. Query/key rank-4 cuts and OV read
cuts recur above their layer-matched nulls, whereas rank-16 QK query and OV
write cuts do not. This is another reason to trust neighborhoods more than a
single global cut.

### QK sides are coupled; OV sides retain more recombination freedom

After within-layer shuffling, all three pairing metrics remain significant at
all ranks (one-sided p = 1/500). The magnitudes differ substantially:

| View | Rank | Distance Spearman (null) | Neighbour overlap (null) | Cluster AMI (null) |
| --- | ---: | ---: | ---: | ---: |
| QK | 4 | 0.619 (0.450) | 0.514 (0.226) | 0.672 (0.470) |
| QK | 8 | 0.751 (0.501) | 0.681 (0.257) | 0.662 (0.452) |
| QK | 16 | 0.780 (0.508) | 0.674 (0.282) | 0.659 (0.436) |
| OV | 4 | 0.249 (0.111) | 0.444 (0.228) | 0.496 (0.342) |
| OV | 8 | 0.303 (0.124) | 0.472 (0.230) | 0.412 (0.277) |
| OV | 16 | 0.292 (0.100) | 0.486 (0.201) | 0.469 (0.328) |

QK should therefore be modeled as coupled query/key structure. OV read and
write are not independent, but their weaker dependence leaves substantially
more room for recombination.

## Decision

Do not promote the selected cluster IDs to biological-style head types. Use
continuous subspace-neighbourhood and co-occurrence fingerprints as the next
representation, while retaining small satellite groups as candidate
specialists.

The next economical gate is a small activation-weighted validation pilot on
the recurrent cross-layer pairs recorded in the JSON output. It should test
whether subspace proximity predicts attention-pattern or output effects beyond
layer, head norm, and operator-spectrum baselines. Only a positive held-out
result would justify causal tests or replication on more models.

## Reproduction

```powershell
python scripts/audit_subspace_families.py
```

Primary outputs:

- `results/pythia-70m-deduped/subspace_family_audit.json`
- `results/pythia-70m-deduped/subspace_family_audit.png`
