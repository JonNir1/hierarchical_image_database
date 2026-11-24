# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.
Retained subjects: N=79 pre-SHINE, N=76 post-SHINE.

## Choice of RDM aggregation method

Notebook: [`behavior/aggregation_comparison.ipynb`](behavior/aggregation_comparison.ipynb) (commit `8a2b898`)

- **Noise-ceiling reliability:** "simple" > "mean" > "evidence" in both cohorts. Split-half Kendall tau (simple / mean / evidence): 0.132 / 0.129 / 0.110 (pre), 0.140 / 0.139 / 0.119 (post). Evidence vs mean: p<1e-12, |effect size| >= 0.95 in every cell.
- **Lower and upper bounds ("simple"):** leave-one-subject-out (lower) and split-half (upper) means. Kendall tau 0.131 / 0.132 (pre) and 0.139 / 0.140 (post); Spearman rho 0.194 / 0.197 (pre) and 0.207 / 0.209 (post). The two bounds nearly coincide in both cohorts.
- **Full-sample noise ceiling:** the Spearman-Brown projection of the split-half bound (about 0.33 pre, 0.35 post) overstates what full-size samples reach on these sparse RDMs: extra subjects mostly add covered pairs, not repeated ratings per pair, so agreement grows slowly with group size. Two groups of about 77 subjects agree at Spearman 0.22, not 0.33 (label-permutation null in [`../s1_pre_vs_post_shine/3_geometry.ipynb`](../s1_pre_vs_post_shine/3_geometry.ipynb)), and two groups of 38 at about 0.20 (same notebook, section 1.5). Use these empirical values as the ceiling at those group sizes.
- **Out-of-sample generalizability:** OOS Kendall tau against the retained-cohort RDM again ranks "simple" first (pre 0.122, post 0.100). The post-SHINE OOS pool is small (18 subjects, 15.7% pair coverage), so that number is lower-confidence. The OOS RDMs are much noisier than the retained ones (split-half Spearman about 0.14 pre and 0.06 post, vs about 0.20 for retained), so raw OOS correlations (0.18 pre, 0.15 post) sit below the retained pre-vs-post value (0.22) because of lower reliability, not lower agreement. Restricting OOS to pilot subjects (excluding v4 screened-out) does not change this, and small-pool reliability estimates (post OOS) are too unstable to use for disattenuation.
- **Caveat, MDS generalizability:** the MDS leg labels all 24 cells "DIFFERENT" (Procrustes disparity 0.67-0.98 at d=10), and 13 of 48 fits hit the 500-iteration cap. Treat it as weak evidence.
- **Verdict:** all three lines of evidence favour "simple".

## Why `D_sem_wn` is not built from ImageNet predictions

Notebook: [`semantic/imagenet_vs_path.ipynb`](semantic/imagenet_vs_path.ipynb) (backend `semantic/imagenet_vs_path.py`; ResNet-50 on the pre-SHINE images)

- **Closest match:** the ResNet top-1 class supplies the closest WordNet match to a path-derived concept for 422 of 717 scorable images (59%), through the filename slot for most (549 of 717 overall). The median minimum distance is 3, and only 117 of 725 images (16%) reach distance 0, nearly all through the filename slot.
- **Agreement with the curated synset:** the path synset that the grid selects equals the curated one for 60.6% of non-human images (animals 84.2%, handmade 57.8%, natural 45.2%) and for 3.7% of the 164 human images. The grid picks the sense closest to the model, so these are upper bounds on automatic path parsing.
- **Why it fails on humans:** ImageNet-1K has no person classes, so the grid lands on `female` or `male` where the curated synset is `woman` or `man` (49 and 37 images, WordNet distance 5).
- **Nature of the disagreements:** of 376, 37.2% differ only in word sense (same lemma), and the median WordNet distance between the two synsets is 9.
- **Decision:** `D_sem_wn` uses direct label-to-synset assignment (`semantic/assign_wordnet_synsets.py`).
