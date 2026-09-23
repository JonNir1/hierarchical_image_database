# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.
Retained subjects: N=79 pre-SHINE, N=76 post-SHINE. Tests are two-sided Mann-Whitney U on per-subject values unless stated.

## Pre- vs post-SHINE cohort comparison

Notebook: [`behavioral/cohort_comparison.ipynb`](behavioral/cohort_comparison.ipynb) (commit `304ac40`)

- **Coverage:** 64.6% (pre) and 63.1% (post) of the 262,450 image pairs were rated at least once, at about 1.9 ratings per rated pair.
- **Within-subject reliability:** median Spearman r is 0.276 (pre) and 0.291 (post); the cohorts do not differ (p=0.71, r=0.03).
- **Distance from the random-placement null (primary metric: Wasserstein):** both cohorts sit far from the null (pooled Wasserstein 0.027 pre, 0.028 post). They do not differ from each other on per-subject Wasserstein (p=0.44) or KS distance (p=0.48).
- **Trial time and moves:** duration (86.9 s pre, 84.2 s post, p=0.76) and moves (27.1 vs 27.3, p=0.25) do not differ. Moves and duration correlate across subjects in pre (rho=0.31, p=0.006) but not in post (rho=0.14, p=0.23).
- **Demographics and retention:** age (median 29 in both, p=0.71), sex (Fisher p=0.87) and retained rate among production v4+ participants (79/101 vs 76/94, Fisher p=0.72) do not differ.
- **QC flags:** the one significant cohort difference. Pre has more main-trial QC flags per subject (mean 1.1% vs 0.4%, p=0.014, r=-0.14).
- **Structural agreement, pre vs post:** on the raw RDMs (107,040 jointly rated pairs) Kendall tau is 0.15 and Spearman rho 0.22. After SMACOF embedding, agreement is much higher (d=2: tau 0.50, rho 0.69). Rank correlation dips to about 0.40 at d=4-6 and recovers to 0.47/0.65 at d=15, while Procrustes m^2 worsens overall (0.31 at d=2 to 0.56 at d=15).
- **Stress:** stress-1 falls from about 0.45 (d=2) to about 0.36 (d=15) in both cohorts, with no sharp elbow. All 28 fits converged at `max_iters=1000`.
- **Caveat, eigenvalue scree:** the classical-MDS scree is not valid at this coverage. Mean-imputation gives an "effective rank" of 354 (pre) and 353 (post) out of 725, the artifact documented in `SpAM_Simulations/empirical/gt_construction.py`. Do not use it to choose dimensionality.

## Choice of RDM aggregation method

Notebook: [`behavioral/aggregation_comparison.ipynb`](behavioral/aggregation_comparison.ipynb) (commit `8a2b898`)

- **Noise-ceiling reliability:** "simple" > "mean" > "evidence" in both cohorts. Split-half Kendall tau (simple / mean / evidence): 0.132 / 0.129 / 0.110 (pre), 0.140 / 0.139 / 0.119 (post). Evidence vs mean: p<1e-12, |effect size| >= 0.95 in every cell.
- **Full-sample noise ceiling:** the Spearman-Brown-projected ceiling for "simple" is about 0.33 (pre) and 0.35 (post).
- **Out-of-sample generalizability:** OOS Kendall tau against the retained-cohort RDM again ranks "simple" first (pre 0.122, post 0.100). The post-SHINE OOS pool is small (18 subjects, 15.7% pair coverage), so that number is lower-confidence.
- **Caveat, MDS generalizability:** the MDS leg labels all 24 cells "DIFFERENT" (Procrustes disparity 0.67-0.98 at d=10), and 13 of 48 fits hit the 500-iteration cap. Treat it as weak evidence.
- **Verdict:** all three lines of evidence favour "simple".
