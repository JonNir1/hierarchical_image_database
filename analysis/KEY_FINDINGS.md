# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.
Retained subjects: N=79 pre-SHINE, N=76 post-SHINE. Tests are two-sided Mann-Whitney U on per-subject values unless stated.

## Pre- vs post-SHINE cohort comparison

Three notebooks, one question each. Effect signs: positive means the pre-SHINE cohort tends to be lower.

- **Caveat, equivalence tests and the permutation test:** there is no pre-defined smallest effect size of interest. Every equivalence bound is the smallest bound with 80% power at these sample sizes (simulated, `analysis/utils/equivalence.py`), rounded up, so a bound says what the sample can resolve, not that the difference is negligible. Equivalence is TOST in its 90%-CI form. The permutation and equivalence tests on RDM agreement assume subjects are exchangeable across cohorts apart from SHINE.

### Do the cohorts behave the same?

Notebook: [`behavioral/1_demographics_and_behavior.ipynb`](behavioral/1_demographics_and_behavior.ipynb) (commit `b6747fc`)

- **Coverage:** 64.6% (pre) and 63.1% (post) of the 262,450 image pairs were rated at least once, at about 1.9 ratings per rated pair.
- **Trial time and moves:** duration (86.9 s pre, 84.2 s post, p=0.76) and moves (27.1 vs 27.3, p=0.25) do not differ. Moves and duration correlate across subjects in pre (rho=0.31, p=0.006) but not in post (rho=0.14, p=0.23).
- **Demographics and retention:** age (median 29 in both, p=0.71), sex (Fisher p=0.87) and retained rate among production v4+ participants (79/101 vs 76/94, Fisher p=0.72) do not differ.
- **Equivalence, behaviour and sample:** age, trial duration and moves are equivalent within +/-0.30 rank-biserial (90% CIs: age [-0.12, 0.19], duration [-0.18, 0.13], moves [-0.04, 0.26]). Sex is equivalent within +/-0.25 (CI [-0.11, 0.16]) and retention within +/-0.20 (CI [-0.12, 0.07]); at these group sizes that only rules out differences of about 20 to 25 percentage points.
- **QC flags:** the one significant cohort difference. Pre has more main-trial QC flags per subject (mean 1.1% vs 0.4%, p=0.014, r=-0.14).

### Is the data equally good?

Notebook: [`behavioral/2_data_quality.ipynb`](behavioral/2_data_quality.ipynb) (commit `5d1be86`)

- **Within-subject reliability:** median Spearman r is 0.276 (pre) and 0.291 (post); the cohorts do not differ (p=0.71, r=0.03).
- **Reliability over repeat index:** every retained subject has 4 verbatim repeats, at fixed session positions (about 16%, 28%, 64% and 84% of the trials), so repeat index and session position are the same variable. Mean reliability is flat in both cohorts (0.30 to 0.33). Per-subject slopes do not differ from 0 (p=0.95 pre, p=0.31 post) or between cohorts (p=0.42, r=0.075).
- **Distance from the random-placement null (primary metric: Wasserstein):** both cohorts sit far from the null (pooled Wasserstein 0.027 pre, 0.028 post). They do not differ from each other on per-subject Wasserstein (p=0.44) or KS distance (p=0.48).
- **Equivalence, data quality:** median reliability, reliability slope, Wasserstein distance and KS distance from the null are equivalent within +/-0.30 rank-biserial (90% CIs: reliability [-0.12, 0.19], slope [-0.08, 0.23], Wasserstein [-0.08, 0.23], KS [-0.08, 0.22]).

### Do they give the same similarity structure?

Notebook: [`behavioral/3_geometry.ipynb`](behavioral/3_geometry.ipynb) (commit `5d1be86`)

- **Structural agreement, pre vs post:** on the raw RDMs (107,040 jointly rated pairs) Kendall tau is 0.15 and Spearman rho 0.22. After SMACOF embedding, agreement is much higher (d=2: tau 0.50, rho 0.69). Rank correlation dips to about 0.40 at d=4-6 and recovers to 0.47/0.65 at d=15, while Procrustes m^2 worsens overall (0.31 at d=2 to 0.56 at d=15).
- **Label permutation:** the raw pre-post agreement (tau 0.148, rho 0.220) is no lower than for shuffled cohort labels (null mean 0.148 and 0.220, one-sided p=0.44 for both, 500 permutations). There is no detectable cohort effect on the similarity structure.
- **Equivalence of agreement:** observed minus the exchangeable expectation is -0.0002 for tau (90% CI [-0.005, 0.005], bound +/-0.01) and -0.0004 for rho (CI [-0.007, 0.007], bound +/-0.02), so the cohorts agree as much as identical cohorts would, to within about 0.01 to 0.02.
- **Stress:** stress-1 falls from about 0.45 (d=2) to about 0.36 (d=15) in both cohorts, with no sharp elbow. All 28 fits converged at `max_iters=1000`.
- **Caveat, eigenvalue scree:** the classical-MDS scree is not valid at this coverage. Mean-imputation gives an "effective rank" of 354 (pre) and 353 (post) out of 725, the artifact documented in `SpAM_Simulations/empirical/gt_construction.py`. Do not use it to choose dimensionality.

## Choice of RDM aggregation method

Notebook: [`behavioral/aggregation_comparison.ipynb`](behavioral/aggregation_comparison.ipynb) (commit `8a2b898`)

- **Noise-ceiling reliability:** "simple" > "mean" > "evidence" in both cohorts. Split-half Kendall tau (simple / mean / evidence): 0.132 / 0.129 / 0.110 (pre), 0.140 / 0.139 / 0.119 (post). Evidence vs mean: p<1e-12, |effect size| >= 0.95 in every cell.
- **Full-sample noise ceiling:** the Spearman-Brown-projected ceiling for "simple" is about 0.33 (pre) and 0.35 (post).
- **Out-of-sample generalizability:** OOS Kendall tau against the retained-cohort RDM again ranks "simple" first (pre 0.122, post 0.100). The post-SHINE OOS pool is small (18 subjects, 15.7% pair coverage), so that number is lower-confidence. The OOS RDMs are much noisier than the retained ones (split-half Spearman about 0.14 pre and 0.06 post, vs about 0.20 for retained), so raw OOS correlations (0.18 pre, 0.15 post) sit below the retained pre-vs-post value (0.22) because of lower reliability, not lower agreement. Restricting OOS to pilot subjects (excluding v4 screened-out) does not change this, and small-pool reliability estimates (post OOS) are too unstable to use for disattenuation.
- **Caveat, MDS generalizability:** the MDS leg labels all 24 cells "DIFFERENT" (Procrustes disparity 0.67-0.98 at d=10), and 13 of 48 fits hit the 500-iteration cap. Treat it as weak evidence.
- **Verdict:** all three lines of evidence favour "simple".
