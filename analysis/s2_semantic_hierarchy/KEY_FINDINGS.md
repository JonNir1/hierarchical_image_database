# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.

How similar are our RDMs to the a priori (Kiani-Mur directory) hierarchy? RDMs: `sem_km` (the tree, the
reference), `sem_wn`, `clip_pre` / `clip_post`, `sens_pre` / `sens_post` (pixels), `behav_joint` (SpAM
behavior, both SHINE cohorts pooled: N = 155, 79 pre + 76 post; 13.1% of pairs unobserved) and two SMACOF
embeddings of it, `behav_joint_mds_low` (d = 5) and `behav_joint_mds_high` (d = 20). Leaf depths 4 and 5 are
pooled into one tier throughout. Kendall tau-a is primary against `sem_km` (9 distinct values), Spearman
secondary. CIs are 95% image-bootstrap percentile intervals (1,000 draws) unless they are participant-level
(`0b`), which are delete-one participant jackknife CIs centred on the estimate; tests against chance
are image-relabeling permutations (1,000, so p >= 0.001).

- **Caveat, the embeddings:** both dimensionalities were chosen from the behavioral data alone (notebook 1),
  never from agreement with the tree. Embedding smooths the ratings and fills the unobserved pairs, so the
  embeddings' larger agreement with the tree is a property of a smoothed model, not of raw behavior; the
  raw RDM is always reported alongside.

## The shape of the tree

Notebook: [`0a_tree_shape.ipynb`](0a_tree_shape.ipynb) (commit `8e5bb00b`)

- **Structure:** 2 domains, 4 mid-level and 16 basic-level categories; basic-level categories hold 6 to 102 images.
- **Ragged:** leaves sit at depths 3 (336 images), 4 (316) and 5 (73), so KM distance is not a function of LCA depth alone (each LCA depth maps to two to five KM values).
- **Pooled bins:** LCA depth {0, 1, 2, 3, >=4} (5,106 pairs in the deepest bin) and branch-depth strata 3x3, 3x>=4, >=4x>=4 (56,280, 130,704, 75,466 pairs).
- **Caveat, strata are confounded with domain:** every animate image is at depth >= 4 and every depth-3 image is inanimate, so the 3x3 stratum is inanimate-only.
- **Pooling hides WordNet's primate reversal** at LCA depth 5 (WordNet routes Old to New World monkeys through the primate root); it is a WordNet bookkeeping quirk, noted and not analysed.

## The joint behavioral RDM on its own

Notebook: [`0b_joint_behavior.ipynb`](0b_joint_behavior.ipynb) (commit `d70fadaa`)

- **Missing data are spread evenly:** 86.9% of pairs observed overall; every pooled LCA bin, branch-depth stratum and category pair (HF, HB, AF, AB, NO, HO) is within about one percentage point (86.0% to 87.6%), at 2.8 ratings per observed pair.
- **Test-retest:** median within-participant Spearman r 0.289 (IQR 0.227 to 0.399), none below 0, as in s1.
- **Nili et al. (2014) bounds on a participant's fit:** the KM tree fits single participants at mean tau-a 0.187, above the lower bound (0.146) for 87.1% of them. The upper bound (0.513) is not usable: each participant is a median 43.0% of every group entry they rated (1/N would be 0.6%).
- **Group RDM reliability grows slowly with group size:** Spearman 0.172 (10 per group), 0.195 (20), 0.201 (38), 0.221 (77), far below what Spearman-Brown predicts (0.335 from 0.201 at 38). So the full-sample ceiling on a noise-free model's Spearman is bracketed between 0.470 (sqrt 0.221) and 0.601 (sqrt of the Spearman-Brown 0.361); the raw RDM's 0.351 with the tree is 58% to 75% of it.
- **Placements are not random:** Wasserstein distance to random placement 0.0275 pooled (per participant median 0.041); more close and far placements than chance (SD 0.202 vs 0.175, mean almost equal).
- **Agreement with the Mur et al. (2013) ratings (42 images):** tau-a 0.280 [0.227, 0.333], Spearman 0.418, p = 0.0002; above either cohort alone (0.258, 0.270 in s1), closer to the raters' consensus than to single raters (0.239 per rater; raters' leave-one-out 0.355).
- **Decomposition (NNLS, pixels and CLIP as the mean of their pre and post RDMs):** full-model R^2 0.210 [0.192, 0.228]; betas KM 0.270 [0.247, 0.293], CLIP 0.204, WordNet 0.111, pixels 0.055. Hierarchically, pixels add 0.018, CLIP 0.103, the KM tree a further 0.078 and WordNet 0.011. The ordering KM > CLIP > WordNet > pixels holds in all 155 leave-one-out refits.
- **Caveat, participant-level CIs:** they are delete-one participant jackknife CIs, centred on the estimate. A participant bootstrap was used first, but a resample has about 63% distinct participants and behaves like a smaller group on these sparse RDMs, so its draws were attenuated and its CIs sat below the estimates (KM beta 0.270 vs bootstrap mean 0.245). The jackknife's bias estimates are large for the same reason (KM beta -0.059, R^2 -0.091): these statistics grow with the number of participants, so the CIs describe a 155-participant RDM, not an infinitely large sample.

## Which behavioral embeddings, and does embedding move behavior toward the tree?

Notebook: [`1_mds_dimensionality.ipynb`](1_mds_dimensionality.ipynb) (commit `54cbfafc`; coverage and the noise ceiling are in `0b`)

- **Held-out prediction:** an embedded half predicts the other half's raw ratings better than the raw half does (0.221) from d = 2 on (0.329), rising slowly to 0.362 at d = 30 with no interior optimum (5 splits). Hence two embeddings: low d = 5 (stress elbow, kneedle; d = 4 if only d = 1 to 15 is searched) and high d = 20 (smallest d capturing 95% of the held-out gain).
- **Embedding moves behavior toward the tree:** Spearman with `sem_km` 0.351 raw, 0.558 (d = 5), 0.625 (d = 20), peaking at d = 2 (0.648); tau-a 0.231, 0.373, 0.421. On the raw RDM's observed pairs and on all pairs the values differ by at most 0.002, so the gain is smoothing, not the filled-in pairs.
- **Robust across d:** for every d from 3 to 30 the depth-gradient rank slope lies between 0.63 and 0.70 (raw 0.39), and the nested separation peaks at the mid-level.
- **The raw RDM's ceiling does not bound the embeddings:** their Spearman with the tree (0.558, 0.625) exceeds the bracket's lower end (0.470), and the high-d one its upper end (0.601).
- **Caveat:** the five half splits share participants, so the held-out SEs understate sampling variability; robustness CIs hold each MDS fit fixed.

## How much rank structure does each RDM share with the tree?

Notebook: [`2_rank_agreement.ipynb`](2_rank_agreement.ipynb) (commit `9432acdd`; ceiling `157a0ef6`)

- **Agreement with `sem_km` (tau-a [CI]; Spearman):** CLIP pre 0.233 [0.210, 0.260]; 0.348. Raw behavior 0.231 [0.221, 0.243]; 0.351. CLIP post 0.207; 0.310. WordNet 0.189 [0.155, 0.222]; 0.286. Pixels pre 0.058, post 0.034 (0.090, 0.054). Embeddings: 0.372 [0.354, 0.391] (d = 5) and 0.421 [0.401, 0.439] (d = 20); 0.557 and 0.625. All permutation p = 0.001.
- **The old "CLIP tracks the tree better than WordNet" is not supported:** tau-a difference +0.045 [-0.001, 0.092], p = 0.054, p_fdr = 0.072 (BH over four tested differences).
- **Raw behavior equals CLIP pre** (tau-a -0.003 [-0.025, 0.018], p_fdr = 0.80); both embeddings exceed it (+0.138 [0.112, 0.164] and +0.187 [0.161, 0.214], p_fdr = 0.002).
- **Ceiling:** raw behavior reaches 58% to 75% of the bracket 0.470 to 0.601 (from `0b`); the embeddings reach 0.93 to 1.19 (d = 5) and 1.04 to 1.33 (d = 20) of it, which shows the bound does not apply to smoothed models.
- **Families share little with each other:** WordNet vs CLIP tau-a 0.03, WordNet vs raw behavior 0.10, CLIP vs raw behavior 0.20; pre vs post 0.76 (CLIP) and 0.53 (pixels).
- **Missing pairs do not distort the comparison:** restricted to behavior's 228,001 observed pairs, every RDM's agreement changes by at most 0.002.

## Does distance grow with hierarchical separation?

Notebook: [`3_depth_gradient.ipynb`](3_depth_gradient.ipynb) (commit `5fd686ef`)

- **Rank slope** (minus Spearman of distance vs LCA bin; [CI]): `sem_km` 0.907, CLIP pre 0.434 [0.399, 0.469], CLIP post 0.398 [0.366, 0.432], raw behavior 0.393 [0.376, 0.410], WordNet 0.279 [0.245, 0.317], pixels 0.107 and 0.086; embeddings 0.625 [0.602, 0.648] (d = 5) and 0.686 [0.662, 0.709] (d = 20).
- **Behavior and CLIP have the same slope but different shapes:** behavior falls further by the deepest bin (root/deepest ratio 4.03 vs 2.13 and 2.00 for CLIP), CLIP flattens at the deep end.
- **Monotone falling** holds for `sem_km`, CLIP, raw behavior and both embeddings; it fails for WordNet (bin 1 is 1.012) and pixels.
- **Not just the domain split:** in the inanimate-only 3x3 stratum raw behavior still falls 1.00, 0.83, 0.54.
- **Caveat:** the deepest bins are small (5,106 pairs; 200 in one stratum cell), and strata cannot be compared on height (each is normalised to its own first bin).

## Which levels of the tree does each RDM recover? (H1b)

Notebook: [`4_level_by_level.ipynb`](4_level_by_level.ipynb) (commit `31fc11d4`)

- **Global contrast (Cohen's d, levels 1 to 3):** d rises with depth for every data-derived RDM (CLIP pre 0.80, 0.96, 1.80; raw behavior 0.61, 1.02, 1.58), the opposite of H1b.
- **Nested contrast (siblings only) removes part of that rise:** CLIP pre 0.80, 0.73, 1.36; raw behavior 0.61, 0.94, 0.93; WordNet 0.44, 0.90, 0.79; embeddings 1.16, 2.06, 1.53 (d = 5) and 1.24, 2.07, 1.66 (d = 20); `sem_km` 2.80, 2.57, 1.36.
- **H1b's ordering holds only for `sem_km`** (d1 - d2 = 0.24 [0.06, 0.40], d2 - d3 = 1.20 [1.02, 1.38]). CLIP and pixels peak at the basic level (CLIP pre d2 - d3 = -0.63 [-0.78, -0.47]); behavior and WordNet peak at the mid-level (raw behavior d1 - d2 = -0.33 [-0.40, -0.26]; embeddings -0.90 and -0.83; WordNet -0.46). The embeddings do place the mid-level above the basic level (d2 - d3 = 0.53 [0.28, 0.79] and 0.42 [0.18, 0.65]).
- **Weakest category:** `inanimate/handmade/other`, a heterogeneous catch-all, for CLIP and behavior.
- **Caveat:** d is not strictly comparable across levels (the level-1 "within" group contains all finer structure, which deflates top-level d), so the d ordering is a weak test of H1b as worded.

## Is the agreement semantic, or inherited from pixels?

Notebook: [`5_semantic_vs_lowlevel.ipynb`](5_semantic_vs_lowlevel.ipynb) (commit `772caa7a`)

- **Pixels do not explain the agreement:** partialling out `sens_pre` keeps 96% to 100% of every RDM's Spearman with the tree (CLIP pre 0.348 to 0.338, raw behavior 0.351 to 0.344, embeddings 0.557 to 0.553 and 0.625 to 0.623; WordNet 0.286 to 0.294).
- **CLIP explains a fifth to a tenth of behavior's agreement:** controlling for CLIP pre leaves 79% of raw behavior's (0.351 to 0.276; CIs do not overlap), 85% (d = 5) and 89% (d = 20) of the embeddings'; adding pixels changes nothing.
- **CLIP and WordNet are different semantic surfaces:** R^2 for the tree 12.1% and 8.2% alone, 19.4% jointly, 20.3% if independent.
- **Behavior absorbs CLIP's contribution but not WordNet's** (commonality analysis, target `sem_km`, 500 draws). With the high-d embedding, full R^2 0.434 [0.407, 0.465]: unique to behavior 0.239 [0.208, 0.273], WordNet 0.040 [0.025, 0.058], CLIP 0.001 [0.000, 0.004], pixels 0.001; shared 0.153, of which CLIP with behavior 0.100. With the low-d embedding, full R^2 0.356: unique to behavior 0.162, WordNet 0.035, CLIP 0.011.
- **Caveat:** unique shares depend on the predictor set, and the CLIP and behavior overlap is symmetric; only monotone pixel contributions are removed.

## Does each RDM's own clustering reproduce the tree?

Notebook: [`6_tree_recovery.ipynb`](6_tree_recovery.ipynb) (commit `e15529e0`)

- **Baker's gamma against the exact tree** (average linkage; image-shuffle null 95% range about +/-0.01, all p = 0.001): `sem_km` 0.979, behavior embeddings 0.804 (d = 20) and 0.680 (d = 5), CLIP pre 0.329, WordNet 0.306, CLIP post 0.277, pixels 0.115 and 0.049. Behavioral embeddings at d = 3, 9, 15 give 0.680, 0.805, 0.806. Raw behavior cannot be clustered (unobserved pairs).
- **Not a matter of tree-likeness alone:** cophenetic correlations are 0.65 to 0.72 for the CLIP, WordNet and pixel RDMs alike (pixels fit a hierarchy, just not this one); 0.673 and 0.773 for the embeddings.
- **Flat cuts vs the KM levels (ARI at 2, 4, 16 clusters):** the behavioral embeddings recover the mid-level almost exactly (0.948 at d = 5, 0.956 at d = 20) and the basic level well (0.630, 0.625), the domain only partly (0.391, a 77/23 split). The other RDMs are about 0 at 2 and 4 clusters (average linkage peels off small groups) and 0.07 to 0.30 at the basic level.
- **Outlier-robust cut** (clusters of at least 5 images): basic-level ARI 0.643 (both embeddings), CLIP pre 0.558, CLIP post 0.466, WordNet 0.321, pixels 0.07 to 0.10 (reference 0.697).
- **HDBSCAN (density-based; descriptive; ARI over the images each RDM's clustering keeps):** behavior's 5 to 9 dense groups (5% to 16% noise) match the mid-level categories (ARI 0.895 low-d, 0.883 high-d at min_cluster_size 10; 0.87 to 0.94 across settings), as the average-linkage cut does.
- **Expected ceiling and floor, only partly:** `sem_km` (expected ceiling) is clustered into leaf-sized groups with 0.1% noise, so it is high at the basic level (0.691 at 10, 0.908 at 20) but low at coarser levels (0.087, 0.232), where ARI penalises the finer partition. The pixel RDMs (expected floor) leave 70% to 77% of the images as noise (at 10), so they are a floor in forming almost no dense groups; their ARIs cover only 23% to 30% of the images.
- **CLIP is unstable:** its dense groups match the basic level over about half the images (`clip_pre` 0.811 at 10, 52% noise), but `clip_post` collapses to 2 clusters at 10 and has 20 at 5.
- **Caveat:** point estimates only (no image bootstrap); results depend on average linkage and the minimum cluster size, and HDBSCAN on min_cluster_size; gamma is dominated by coarse structure.
