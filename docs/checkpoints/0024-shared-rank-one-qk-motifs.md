# Checkpoint 0024: QK heads contain reusable local channels, not a unique atom table

The preceding held-out compression experiment showed that QK heads share more
input/output support geometry than OV heads.  This checkpoint tests the next,
more specific part of the research hypothesis:

> Can a QK matrix be represented as a combination of compact, reusable
> subspace-to-subspace actions learned without behavior labels?

Within 64-dimensional query and key supports learned only from training heads,
the model fits a CP decomposition

\[
M_h \approx \sum_{k=1}^{K} a_{hk}\,u_kv_k^\top.
\]

Each `u_k v_k^T` is a rank-one conditional-scoring channel: one residual
direction contributes a query feature, another contributes a key feature, and
their signed product contributes to the attention logit.  The directions and
coefficients are learned entirely from weights.  Complete heads are held out;
in a harder second split, complete alternating layers are held out.

## Held-out result

| Split | 4 motifs | 8 motifs | 16 motifs | 32 motifs | 32-motif spectrum-Haar null | 32-motif side-pairing null |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unseen heads | 0.40% | 0.89% | 1.55% | **2.72%** | 0.012% | 0.41% |
| Unseen layers | 0.13% | 0.25% | 0.50% | **0.88%** | 0.012% | 0.16% |

All percentages are variance recovered from the complete, unit-Frobenius QK
operator, not merely from its projected core.  Every one of nine repetitions
of both nulls is below the observation (finite-null upper-tail p-value 0.10,
the minimum possible with this deliberately inexpensive run).

The 32 rank-one motifs use 4,096 shared dictionary parameters.  One arbitrary
64-by-64 PCA matrix atom also uses 4,096 parameters and recovers 2.25% for
unseen heads and 0.76% for unseen layers.  The structured motifs therefore
beat the equally sized unrestricted dictionary in both splits.  Unrestricted
PCA with 32 full-matrix atoms performs better (4.33% and 1.41%) but uses
131,072 dictionary parameters.  The rank-one structure buys substantial
compression rather than winning through greater capacity.

This parameter comparison concerns the shared dictionaries.  A 32-motif
encoding also stores 32 coefficients per head, whereas one PCA atom stores one;
including the 24 training-head coefficients gives 4,864 versus 4,120 fitted
numbers.  The structured model remains close in total size and beats the
smaller PCA model, but the comparison is not an exact minimum-description-
length proof.

The main values are stable across independent optimization seeds.  With seeds
23, 71, and 131, 32 motifs recover 2.72--2.78% in the unseen-head split and
0.856--0.880% in the unseen-layer split.

## Are the individual atoms real?

Only partly.  After optimal permutation matching between three independent
full-population fits, 12 of 32 reference atoms have matrix cosine at least 0.8
in each comparison.  Median matched cosine is 0.63--0.68.  Thus a stable core
exists, but the decomposition as a complete list of 32 named atoms is not
identifiable.

At a deliberately simple activity threshold (at least 25% of a motif's
maximum absolute coefficient), the median motif is appreciably used by 7.5 of
48 heads and appears in two of six layers.  The median participation ratio is
12.1 effective heads.  Some motifs are broad; others are layer-local.  This
agrees with the much smaller, but still positive, transfer to unseen layers.

## Interpretation

This is direct evidence for one important portion of the compartmentalization
hypothesis: QK matrices contain reusable low-dimensional query-key actions,
and multiple such actions combine inside a head.  The evidence is not merely
whole-head similarity, ordinary rank, singular spectrum, or layer membership.

It does **not** establish a canonical periodic table.  Most complete-operator
variance remains unexplained, unrestricted matrix atoms remain more accurate
when allowed far more parameters, and many individual CP atoms rotate between
equivalent solutions.  The defensible object is currently a reusable family
or span of local channels, with a smaller stable subset of individual motifs.

The next scientific step should be behavioral validation of only the stable
12-or-so motifs: test whether their query/key directions select coherent input
classes and predict held-out attention-score contributions.  That is now more
valuable than increasing the dictionary size, because it distinguishes real
conditional mechanisms from statistical compression.
