# Exploratory Motif Stability Audit

Date: 2026-08-16  
Parent checkpoint: `checkpoint-0002`

Checkpoint 0002 found that sparse mixtures reconstruct held-out operators much
better than hard cluster centroids. This follow-up asks whether the learned
dictionary atoms themselves are reproducible.

## Method

- Fit 16- and 32-atom dictionaries to exact PCoA coordinates of QK, OV, and
  equal-weight joint normalized-Frobenius geometry.
- Run five explicitly randomized initial dictionaries and coefficient matrices.
- Separately fit five dictionaries to 80% bootstrap samples of heads.
- Remove atom order and sign symmetries with optimal Hungarian matching, then
  report mean absolute matched cosine.
- Compare with independently sampled random dictionaries of the same shape.

The explicit random coefficient initialization matters: scikit-learn ignores a
provided `dict_init` unless `code_init` is also provided. The audit tests both.

## Results

| View | Atoms | Random-start stability | Bootstrap stability | Random baseline |
| --- | ---: | ---: | ---: | ---: |
| QK | 16 | 0.847 | 0.385 | 0.159 |
| QK | 32 | 0.800 | 0.465 | 0.181 |
| OV | 16 | 0.795 | 0.292 | 0.155 |
| OV | 32 | 0.764 | 0.362 | 0.181 |
| Joint | 16 | 0.843 | 0.300 | 0.156 |
| Joint | 32 | 0.738 | 0.382 | 0.179 |

## Interpretation

The optimization repeatedly finds substantially aligned atoms from unrelated
starts, and bootstrap dictionaries agree far more than random dictionaries.
However, sample stability is only moderate. The current evidence supports a
reproducible motif-rich subspace or vocabulary, not a unique, sharply defined
list of atoms. Stable *families of atoms* may be a more appropriate object than
individual atom identities.

This remains exploratory. Confirmation requires another model or checkpoint,
and semantic or causal tests of matched atoms.
