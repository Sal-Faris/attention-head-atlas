# Checkpoint 0013: joint conditional QK rules

## Result

Checkpoint 0012 showed that unsupervised query/key residual-input classes are
associated with a QK channel's dominant component.  This checkpoint asks the
stronger question: does each side retain information after the other side is
already known?

Using the coarse discovery-only 1--4 class mixtures, test both

```text
I(component; query class | key class)
I(component; key class | query class).
```

The held-out null shuffles component labels only within the same document,
exact relative-offset bin, and conditioned-side class.  It therefore preserves
document/style, position, and the entire opposite-side class association.

With 999 permutations and BH correction across 34 tests, **23 of 34**
conditional effects remain significant (mean excess conditional mutual
information 0.00885).  Examples include additional query information given
key class for L4H7, L5H3, L2H0, and L3H5; and additional key information given
query class for L1H3, L3H4, L5H4, and L3H0.

## Interpretation

This is the strongest evidence so far for the research program's central
conditional-routing hypothesis.  A useful current unit is not a whole head
or a universal atom, but a soft rule of the form

```text
(query-side residual state, key-side residual state, relative position)
    -> mixture of low-rank QK routing components.
```

The effects remain continuous and modest; this does not establish sharp token
categories or semantic names.  It does establish that pair-dependent structure
exists beyond the obvious one-sided, document, and positional alternatives.

## Reproduction

```powershell
python scripts/test_qk_joint_input_rules.py --permutations 999
```
