# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.

Should the R `smacof::mds()` bridge in `SpAM_Simulations/core/multi_dimensional_scaling.py` be replaced by
`rsatoolbox.util.weighted_mds`? Decision: **keep the R backend** (long form in [`REPORT.md`](REPORT.md)).
The scripts here are command-line tools with printed output, not notebooks, so each section cites the script.
This review re-ran the two small ones on 2026-11-24 (R 4.5 with `smacof` through `rpy2`); the full
real-data sweep was not re-run.

- **Caveat, not re-run:** `mds_deep_dive.py` needs the RDMs `behav_{pre,post}_{mean,evidence}`, and
  `analysis/results/rdms/` holds only `behav_{pre,post,joint}_simple` (plus joint MDS). The 2x2 results in
  `REPORT.md` (disparity 0.97 to 0.99 across backends, 0.19 to 0.25 across aggregations) are not re-verified here.
- **Caveat, R start is unseeded:** with `precalc_init=False` R's random start differs between runs, so
  iteration counts and the last digits of the agreement statistics vary (disparity 5.1e-5 in one run, 9.5e-5 in the next).

## Is the vectorized `weighted_mds` fix equivalent?

Script: [`mds_rsatoolbox_vectorized.py`](mds_rsatoolbox_vectorized.py) (commit `26d10e8e`)

- **Equivalent:** at n = 60 the vectorized and original solvers agree exactly (53 iterations, stress 6384.4873593045, maximum configuration difference 0). The O(n^2) graph-Laplacian form is the same computation as the O(n^4) double loop.

## Do R and rsatoolbox agree on clean data?

Script: [`mds_rsatoolbox_compare.py`](mds_rsatoolbox_compare.py) (commit `26d10e8e`)

- **Synthetic, n = 30, true dimension 3, no missing pairs:** Procrustes disparity 4.6e-5 to 9.5e-5 and distance-vector correlation 0.99998, verdict EQUIVALENT, with and without a shared starting configuration. Matches `REPORT.md` (about 5e-5).

## Do they agree on real behavioral data with missing pairs?

Script: [`mds_rsatoolbox_compare.py`](mds_rsatoolbox_compare.py) (commit `26d10e8e`)

- **No:** on the 60 best-covered images of `behav_pre_simple` and `behav_post_simple` (5 dimensions) the verdict is DIFFERENT in all four runs, with disparity 0.15 to 0.62 and distance-vector correlation 0.53 to 0.92.
- **Stress:** R reports about 0.30 (it hit the 300-iteration cap in three of four runs); rsatoolbox reports about 0.0024 after 10 to 220 iterations, consistent with the weight-blind stopping rule in issue #494.
- **Caveat:** this is the `simple` aggregation at n = 60. `REPORT.md`'s figures are for `mean` and `evidence` at n = 725, so the two are consistent in direction, not comparable in size.
