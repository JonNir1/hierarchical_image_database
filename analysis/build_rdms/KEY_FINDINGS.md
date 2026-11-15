# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.
Retained subjects: N=79 pre-SHINE, N=76 post-SHINE.

## Choice of RDM aggregation method

Notebook: [`behavior/aggregation_comparison.ipynb`](behavior/aggregation_comparison.ipynb) (commit `8a2b898`)

- **Noise-ceiling reliability:** "simple" > "mean" > "evidence" in both cohorts. Split-half Kendall tau (simple / mean / evidence): 0.132 / 0.129 / 0.110 (pre), 0.140 / 0.139 / 0.119 (post). Evidence vs mean: p<1e-12, |effect size| >= 0.95 in every cell.
- **Full-sample noise ceiling:** the Spearman-Brown-projected ceiling for "simple" is about 0.33 (pre) and 0.35 (post).
- **Out-of-sample generalizability:** OOS Kendall tau against the retained-cohort RDM again ranks "simple" first (pre 0.122, post 0.100). The post-SHINE OOS pool is small (18 subjects, 15.7% pair coverage), so that number is lower-confidence. The OOS RDMs are much noisier than the retained ones (split-half Spearman about 0.14 pre and 0.06 post, vs about 0.20 for retained), so raw OOS correlations (0.18 pre, 0.15 post) sit below the retained pre-vs-post value (0.22) because of lower reliability, not lower agreement. Restricting OOS to pilot subjects (excluding v4 screened-out) does not change this, and small-pool reliability estimates (post OOS) are too unstable to use for disattenuation.
- **Caveat, MDS generalizability:** the MDS leg labels all 24 cells "DIFFERENT" (Procrustes disparity 0.67-0.98 at d=10), and 13 of 48 fits hit the 500-iteration cap. Treat it as weak evidence.
- **Verdict:** all three lines of evidence favour "simple".
