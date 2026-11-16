# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.
Retained subjects: N=79 pre-SHINE, N=76 post-SHINE. Tests are two-sided Mann-Whitney U on per-subject values unless stated.

## Pre- vs post-SHINE cohort comparison

Five notebooks, one question each. Effect signs: positive means the pre-SHINE cohort tends to be lower.

- **Caveat, equivalence tests and the permutation test:** there is no pre-defined smallest effect size of interest. Every equivalence bound is the smallest bound with 80% power at these sample sizes (simulated, `../utils/equivalence.py`), rounded up, so a bound says what the sample can resolve, not that the difference is negligible (notebooks 4 and 5 use the closed-form bound described in their sections for combined-RDM differences). Equivalence is TOST in its 90%-CI form. The permutation and equivalence tests on RDM agreement assume subjects are exchangeable across cohorts apart from SHINE.

### Do the cohorts behave the same?

Notebook: [`1_demographics_and_behavior.ipynb`](1_demographics_and_behavior.ipynb) (commit `5aef7c0`)

- **Coverage:** 64.6% (pre) and 63.1% (post) of the 262,450 image pairs were rated at least once, at about 1.9 ratings per rated pair.
- **Trial time and moves:** duration (86.9 s pre, 84.2 s post, p=0.76) and moves (27.1 vs 27.3, p=0.25) do not differ. Moves and duration correlate across subjects in pre (rho=0.31, p=0.006) but not in post (rho=0.14, p=0.23).
- **Demographics and retention:** age (median 29 in both, p=0.71), sex (Fisher p=0.87) and retained rate among production v4+ participants (79/101 vs 76/94, Fisher p=0.72) do not differ.
- **Equivalence, behaviour and sample:** age, trial duration and moves are equivalent within +/-0.30 rank-biserial (90% CIs: age [-0.12, 0.19], duration [-0.18, 0.13], moves [-0.04, 0.26]). Sex is equivalent within +/-0.25 (CI [-0.11, 0.16]) and retention within +/-0.20 (CI [-0.12, 0.07]); at these group sizes that only rules out differences of about 20 to 25 percentage points.
- **QC flags:** the one significant cohort difference. Pre has more main-trial QC flags per subject (mean 1.1% vs 0.4%, p=0.014, r=-0.14).

### Is the data equally good?

Notebook: [`2_data_quality.ipynb`](2_data_quality.ipynb) (commit `856daa5`)

- **Within-subject reliability:** median Spearman r is 0.276 (pre) and 0.291 (post); the cohorts do not differ (p=0.71, r=0.03).
- **Reliability over repeat index:** every retained subject has 4 verbatim repeats, at fixed session positions (about 16%, 28%, 64% and 84% of the trials), so repeat index and session position are the same variable. Mean reliability is flat in both cohorts (0.30 to 0.33). Per-subject slopes do not differ from 0 (p=0.95 pre, p=0.31 post) or between cohorts (p=0.42, r=0.075).
- **Distance from the random-placement null (primary metric: Wasserstein):** both cohorts sit far from the null (pooled Wasserstein 0.027 pre, 0.028 post). They do not differ from each other on per-subject Wasserstein (p=0.44) or KS distance (p=0.48).
- **Equivalence, data quality:** median reliability, reliability slope, Wasserstein distance and KS distance from the null are equivalent within +/-0.30 rank-biserial (90% CIs: reliability [-0.12, 0.19], slope [-0.08, 0.23], Wasserstein [-0.08, 0.23], KS [-0.08, 0.22]).

### Do they give the same similarity structure?

Notebook: [`3_geometry.ipynb`](3_geometry.ipynb) (commit `3a719ff`)

- **Structural agreement, pre vs post:** on the raw RDMs (107,040 jointly rated pairs) Kendall tau is 0.15 and Spearman rho 0.22. After SMACOF embedding, agreement is much higher (d=2: tau 0.50, rho 0.69). Rank correlation dips to about 0.40 at d=4-6 and recovers to 0.47/0.65 at d=15, while Procrustes m^2 worsens overall (0.31 at d=2 to 0.56 at d=15).
- **Label permutation:** the raw pre-post agreement (tau 0.148, rho 0.220) is no lower than for shuffled cohort labels (null mean 0.148 and 0.220, one-sided p=0.43 for both, 250 permutations). There is no detectable cohort effect on the similarity structure.
- **Equivalence of agreement:** observed minus the exchangeable expectation is -0.0003 for tau (90% CI [-0.005, 0.004], bound +/-0.01) and -0.0004 for rho (CI [-0.007, 0.006], bound +/-0.02), so the cohorts agree as much as identical cohorts would, to within about 0.01 to 0.02.
- **Stress:** stress-1 falls from about 0.45 (d=2) to about 0.36 (d=15) in both cohorts, with no sharp elbow. All 28 fits converged at `max_iters=1000`.

### Are they equally similar to the Kiani hierarchy?

Notebook: [`4_semantic_hierarchy.ipynb`](4_semantic_hierarchy.ipynb) (commit `8ad69be`)

- **Agreement with the Kiani tree (725 images):** Kendall tau-a (primary) is 0.209 pre and 0.210 post, with 95% participant-bootstrap CIs [0.187, 0.213] and [0.187, 0.216]; Spearman rho (secondary) is 0.318 and 0.320. Both image-relabeling permutation p-values are at the floor for 1,000 permutations (0.001).
- **Per participant:** mean tau-a against the Kiani RDM is 0.186 pre and 0.189 post, above the same participants' lower noise bounds (0.131 and 0.139; Wilcoxon p < 0.001 in both). The Kiani RDM takes few integer values, which is why tau-a is primary.
- **Equivalence, per-participant fit:** rank-biserial effect 0.002, 90% CI [-0.155, 0.156], equivalent within +/-0.30 (Mann-Whitney p=0.99).
- **Equivalence, combined-RDM agreement:** pre minus post tau-a is -0.0005 (90% CI [-0.018, 0.016], bound +/-0.03) on each cohort's own observed pairs and 0.004 ([-0.015, 0.019], +/-0.03) on the 107,040 pairs both cohorts observed; rho gives -0.002 ([-0.028, 0.023], +/-0.04) and 0.004 ([-0.023, 0.027], +/-0.04). The data alone exclude differences larger than about 0.02 in tau-a and 0.03 in rho.
- **Tree agreement (average linkage, descriptive for pre vs post):** Baker's gamma against the exact directory-tree dendrogram is 0.98, 0.80 and 0.81 for pre and 0.68, 0.67 and 0.69 for post at MDS d = 3, 9 and 15, all above the image-shuffle null (p=0.001). Pre is higher than post at every dimension, but this is not tested (a bootstrap would need an MDS refit per draw). Gamma is dominated by coarse structure, since most pairs merge near the root.
- **HDBSCAN (descriptive):** at d = 9 and min_cluster_size 10 the clusters match the mid-level Kiani categories best (ARI 0.94 pre, 0.99 post), against about 0.5 at the domain and basic levels; 12 to 13% of images are labelled noise.
- **Multiple comparisons:** Benjamini-Hochberg within each cohort over the tau-a permutation and the three gamma tests leaves every p_fdr at 0.001.
- **Caveat, bound rule for combined-RDM differences:** these bounds are (z(0.95) + z(0.80)) x SE rounded up to 0.01. Each one-sided test has 80% power at the bound's edge but both must pass, so a true zero difference is declared equivalent about 60% of the time (calibrated in notebook 5). They are therefore tighter than the simulated rule used for the per-subject tests above and in notebooks 1 and 2.

### Are they equally similar to earlier ratings and recordings?

Notebook: [`5_ground_truth.ipynb`](5_ground_truth.ipynb) (commit `fa85272`)

- **Overlap:** the external 92-image data (the rsatoolbox demo copy of Kriegeskorte et al. 2008b and Mur et al. 2013) has no image IDs, so images were matched by content and confirmed by eye: 42 shared images (861 pairs; pre observed 536, post 534, both 334). Categories: natural 19, artificial 10, nonhuman face 6, human body part 5, nonhuman body part 2. There are no human faces, so everything below concerns objects, animals and body parts only.
- **5a, Mur 2013 ratings (16 raters):** tau-a is 0.258 pre and 0.270 post (bootstrap CIs [0.203, 0.293] and [0.204, 0.300]); rho is 0.382 and 0.404; permutation p = 0.0002 (the floor for 5,000). Each individual rater is fitted at 0.204 (pre) and 0.236 (post) on average, below the raters' leave-one-out ceiling of 0.356.
- **5a equivalence:** pre minus post tau-a is -0.011 (90% CI [-0.063, 0.048], bound +/-0.09) on each cohort's own pairs and -0.028 ([-0.102, 0.040], +/-0.11) on jointly observed pairs; Spearman is -0.023 ([-0.093, 0.063], +/-0.12). Post is numerically higher, but the difference is not distinguishable from zero.
- **5a trees (descriptive):** Baker's gamma of each cohort's average-linkage tree against the Mur tree is 0.61, 0.77 and 0.61 for pre and 0.72, 0.61 and 0.61 for post at d = 3, 9 and 15 (all p = 0.0002). The Kiani tree itself agrees with the Mur tree at 0.57.
- **5b, Kriegeskorte et al. 2008b:** human IT (fMRI, 8 RDMs) tau-a is 0.128 pre and 0.150 post, and monkey IT (recordings, one pooled RDM) 0.157 and 0.173; all p = 0.0002. The human-IT values are comparable to the human-IT leave-one-out ceiling (0.129; mean fit per RDM 0.066 pre and 0.072 post). The monkey RDM has no ceiling.
- **5b equivalence:** pre minus post tau-a is -0.022 ([-0.058, 0.041], +/-0.08) for human IT and -0.016 ([-0.053, 0.042], +/-0.08) for monkey IT on own pairs, and -0.005 ([-0.054, 0.060], +/-0.09) and 0.003 ([-0.063, 0.067], +/-0.10) on jointly observed pairs. All 12 tests (3 targets, 2 statistics, 2 pair sets) are equivalent.
- **Calibration of the bound rule:** on 10 random halves of the pre cohort, where the true difference is zero, 37 of 60 tests (62%) declared equivalence, as expected for this rule (about 60%: 80% power per one-sided test, both must pass).
- **Multiple comparisons:** within each cohort, the three permutation tests and three gamma tests all have p_fdr = 0.0002.
- **Caveat:** 42 images with no faces limit generalisation, and each bound says what the design can resolve, not that a difference is negligible. The 8 human-IT RDMs are 4 subjects x 2 sessions, so they are not independent, and HDBSCAN on 42 images is exploratory only (see the notebook).
