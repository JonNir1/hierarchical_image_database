# rsatoolbox MDS backend evaluation

Investigation into whether `rsatoolbox`'s MDS embedding step
(`rsatoolbox.util.weighted_mds`) could replace the R `smacof::mds()` bridge used in
[`SpAM_Simulations/core/multi_dimensional_scaling.py`](../../SpAM_Simulations/core/multi_dimensional_scaling.py).
This is a *separate* question from whether `rsatoolbox` is useful for building the
behavioral RDMs in the first place ([`analysis/rdms/behavioral.py`](../rdms/behavioral.py)) --
that part of `rsatoolbox` (`rdm.combine.from_partials()`/`rescale()`) has no bugs found
against it and remains in active use.

## Decision

**Keep the R backend. Do not swap.** `multi_dimensional_scaling.py` is unchanged.

## Why

Two independent, real bugs were found in `rsatoolbox.util.weighted_mds`, both filed
upstream with minimal reproductions and proposed fixes:

- **[rsagroup/rsatoolbox#494](https://github.com/rsagroup/rsatoolbox/issues/494)** --
  the stress/convergence check ignores the weight matrix entirely: it sums over
  *every* pair unconditionally, including zero-weight (missing) ones, even though the
  embedding update step correctly excludes them. This breaks SMACOF's core
  monotone-convergence guarantee whenever any pair is unweighted -- which is the norm
  for this project's data (SpAM sessions cover only a subset of image pairs), not a
  corner case. Once triggered, the result is independent of `eps`: tightening it from
  `1e-6` to `1e-12` changes nothing. Minimal repro: 5 points, one missing pair.

- **[rsagroup/rsatoolbox#495](https://github.com/rsagroup/rsatoolbox/issues/495)** --
  the weighted update's `V` matrix is built via a Python double loop over every pair,
  each iteration doing a full `(n, n)` outer product: O(n⁴) overall. It didn't finish
  `n=500` within 2+ minutes in testing; this project's real data is `n=725`. Separate
  from the correctness bug above -- pure performance. `V` is exactly the weight
  matrix's graph Laplacian (`diag(weight.sum(axis=1)) - weight`), computable in O(n²)
  with no loop; `mds_rsatoolbox_vectorized.py` here is that fix, verified bit-identical
  to the original at `n=60` (the largest size the original still completes in
  reasonable time).

On dense, fully-observed synthetic data the two backends (R vs. rsatoolbox) agree
almost exactly (Procrustes disparity ~5e-5). On any data with missing pairs --
including this project's real behavioral RDMs, and even `SpAM_Simulations`' own
plain-mean + binary-mask scheme, not just rsatoolbox's `evidence` weighting --
agreement collapses (Procrustes disparity 0.5-0.99 depending on scale; confdist
correlation as low as 0.036 at the real `n=725` scale). The bug's impact *grows*
with `n`, not shrinks.

## What's in this directory

- **`mds_rsatoolbox_vectorized.py`** -- the O(n²) fix for issue #495 (graph-Laplacian
  identity in place of the double loop), verified bit-identical to rsatoolbox's own
  `smacof()` at `n=60`. This is what makes running rsatoolbox's algorithm at this
  project's real `n=725` scale possible at all; used by the other two scripts here.
  Kept in case rsatoolbox's upstream fix (or this project's own future use of it)
  makes the backend swap worth revisiting.
- **`mds_rsatoolbox_compare.py`** -- point comparison tool: runs the R backend and
  rsatoolbox's *real* (unmodified, slow) `smacof()` side by side on the same input, for
  small-to-moderate `n` where the original is still tractable. Useful for re-checking
  agreement against a future rsatoolbox release without needing the vectorized
  workaround.
- **`mds_deep_dive.py`** -- the full study: a 2x2 sweep (aggregation method `mean`/
  `evidence` x MDS backend R/rsatoolbox-vectorized) across `ndim ∈ {2,3,5,10}` and both
  SHINE variants, on the real `behav_{pre,post}_{evidence,mean}` RDMs at full `n=725`
  scale, plus the systematic minimal-repro search that found the examples used in the
  two filed issues.

## Results from the 2x2 deep-dive (`mds_deep_dive.py`, full real data, both variants, `ndim ∈ {2,3,5,10}`)

- **Backend effect** (same aggregation, R vs. rsatoolbox-vectorized): DIFFERENT in
  every cell (8/8) -- Procrustes disparity 0.97-0.99, confdist Pearson r 0.036-0.11.
  This is bug #494's real-world severity, not a subtle numerical discrepancy.
- **Aggregation effect, R backend** (`mean` vs. `evidence`, both via R): DIFFERENT --
  disparity 0.19-0.25, r 0.82-0.91. **This is a real, independent finding for the
  actual behavioral analysis**, unrelated to either rsatoolbox bug: the two
  aggregation methods produce noticeably different *converged* embeddings, more so
  than their RDM-level correlation (ρ=0.90, computed in `analysis/rdms/validate_rdms.py`)
  alone would suggest.
- **Aggregation effect, rsatoolbox backend** (`mean` vs. `evidence`, both via
  rsatoolbox): looks "similar" by the numbers (disparity ~0.005-0.013, r ~0.96-0.98) --
  but this is not meaningful. Both conditions are independently broken by bug #494,
  converging in a handful of iterations from the same starting point, so of course
  they land close to each other. Do not read this as "aggregation doesn't matter" --
  it's an artifact of both arms failing the same way.
- **Minimal-repro search**: the eps-sensitivity signature (bug #494) appeared in
  **16/16** tested cells (both variants x both aggregation methods x all 4 dims) at
  real scale -- not rare. Shrinking a real-data subset found it reproduces down to
  `n=4`; investigating why revealed the real `evidence`-combined distances are tiny
  (~1e-3 to 1e-6), comparable to rsatoolbox's hardcoded `1e-5` zero-distance floor.
  Confirmed on synthetic data: small-magnitude coordinates *and* at least one missing
  pair together trigger it; neither alone does. Minimal working example: `n=5`, 2D,
  exactly one missing pair (used as the repro in issue #494).

## If revisiting this later

Re-run `mds_rsatoolbox_compare.py` against a newer `rsatoolbox` release first, at
small `n`, to check whether #494/#495 have been fixed upstream before considering
`mds_rsatoolbox_vectorized.py` or a backend swap again.
