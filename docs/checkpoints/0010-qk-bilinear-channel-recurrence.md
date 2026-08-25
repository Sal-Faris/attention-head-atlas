# Checkpoint 0010: recurrence of joint QK channels

## Question

Checkpoint 0009 found that a single head's matched QK preference margins are
compactly described by rank-2--4 *joint* query--key channels.  This checkpoint
asks a deliberately narrower follow-up question: do the rank-4 channels of
heads already known to have recurrent QK **side** geometry also resemble one
another after they are mapped into the full residual-to-residual routing map?

This is not a new clustering sweep.  It tests an existing cross-checkpoint
family relation in a richer representation.

## Method

For each of the 48 heads at Pythia-70M-deduped step 143000, refit the rank-4
bilinear margin model with the ridge selected in checkpoint 0009:

```text
margin ~= q^T U V^T (k_positive - k_neutral).
```

For every selected event, map the learned head-space channel through the
observed destination and source RoPE rotations, then average those maps over
the discovery events.  Finally map it through the head's processed Q and K
factors.  The resulting 512-by-512 matrix is a position-mixture estimate of
the channel's residual-space routing kernel.

Compare normalized Frobenius distances between those kernels on the 12
previously recurrent rank-4 query-side edges and the 12 recurrent key-side
edges.  Each observed mean is compared to 1,999 draws which preserve the
ordered layer pair of every edge.  The stricter null also removes the union of
all 18 known rank-4 family edges from the candidate pool.

The old family relations and the new channels share model weights, so this is
not a fully independent replication.  Its purpose is representation-level
triangulation: a recurrence detected from side subspaces should remain visible
when evaluated as a joint conditional routing kernel if the same underlying
structure is involved.

## Results

| Earlier family | Observed distance / exact-layer null | p | With all known family edges excluded | p |
| --- | ---: | ---: | ---: | ---: |
| rank-4 query side | 0.9957 | 0.0125 | 0.9955 | 0.0180 |
| rank-4 key side | 0.9885 | 0.0005 | 0.9880 | 0.0005 |

The key-side families remain closer as **joint routing kernels** than matched
ordered-layer-pair controls, with a 1.2% reduction in mean normalized
distance.  The query-side correspondence is weaker: about a 0.45% reduction,
but still below its exact-layer null.  Excluding every previously known family
edge barely changes either result.

## Interpretation

This modestly strengthens the case that the recurrent key-side families are
not an artifact of looking at one factor in isolation.  Their learned
query--key channels also implement slightly more similar full routing maps.

It is still a small effect.  This does **not** establish a discrete atlas of
universal QK atoms, semantic input classes, or causal circuits.  The evidence
is better described as a weak, continuous, reusable channel geometry, with
the key side more conserved than the query side in these families.

The most informative next test is to retain position as part of the object
instead of averaging RoPE-conditioned kernels across events.  That can tell
us whether the residual recurrence comes from a genuinely reusable routing
channel or merely from a similar mixture of position-specific channels.

## Reproduction

```powershell
python scripts/audit_bilinear_channel_recurrence.py --iterations 200 --permutations 1999
```

Primary compact output:

- `results/pythia-70m-deduped/qk_bilinear_channel_recurrence_v1.json`
