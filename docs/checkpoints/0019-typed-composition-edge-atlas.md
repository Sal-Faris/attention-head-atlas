# Checkpoint 0019: typed writer-to-reader edge identities persist through training

The previous head and side analyses show that static geometric recurrence does
not automatically produce a transferable isolated function.  This checkpoint
changes the primitive object: an earlier OV head's **write directions** composed
with a later component's **read directions**.

For every ordered pair of layers and every source/target head pair, three
scale-free static edges are calculated:

\[
E_{a\to b}^{R} =
\frac{\lVert W_O^a W_R^b\rVert_F}
{\lVert W_O^a\rVert_F\lVert W_R^b\rVert_F},
\qquad R\in\{Q,K,V\}.
\]

Here `Q` says that head \(a\)'s write space reaches what head \(b\) asks for,
`K` says it reaches how a source position advertises itself to \(b\), and `V`
says it reaches what \(b\) can retrieve.  The factors include the same static
layer-norm processing used elsewhere in the atlas.

For each exact ordered layer pair, the 64 source-head/target-head edge
strengths at step 16k are rank-correlated with the corresponding strengths at
step 143k.  The control independently permutes source and target head identity
within each fixed layer pair, retaining that pair's full strength distribution.

| Edge type | Mean within-layer-pair early-to-final Spearman | Head-identity null mean | Upper-tail p-value |
| --- | ---: | ---: | ---: |
| OV writer → Q reader | 0.416 | -0.003 | 0.001 |
| OV writer → K reader | 0.769 | 0.000 | 0.001 |
| OV writer → V reader | 0.739 | -0.002 | 0.001 |

This is the strongest stable structure observed so far.  In particular, K and
V edge identity is highly persistent even after conditioning on the same
source/target layer pair.  The strongest final static examples include
L2H6→L3H3/Q, L2H7→L3H6/K, and L4H7→L5H4/V.

**Interpretation.** These are not yet interpreted *circuits*: a large static
edge means a possible write-to-read route, not that it is used on all prompts
or causally decisive.  It does show that typed inter-component relations are
more stable objects than a whole-head atom.  The next correct validation is to
measure whether the strongest static edges predict held-out activation-mediated
influence better than exact layer-pair-matched edges, before examining their
semantics or applying interventions.
