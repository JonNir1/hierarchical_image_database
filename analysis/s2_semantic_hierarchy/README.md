# `s2_semantic_hierarchy/`

How similar are our RDMs to the **a priori hierarchy**, the curated Kiani-Mur directory tree under
`images/<variant>/`? One numbered notebook per sub-question, each with its own backend module and tests;
findings are summarised in `KEY_FINDINGS.md`.

**RDMs compared:** `sem_km` (the tree itself, the reference), `sem_wn` (WordNet), `clip_pre` / `clip_post`
(CLIP ViT-B/32), `sens_pre` / `sens_post` (pixels), `behav_joint` (the SpAM behavioral RDM with both SHINE
cohorts pooled, `D_behav_joint_simple`), and two SMACOF embeddings of it, `behav_joint_mds_low` and
`behav_joint_mds_high`, whose dimensionalities notebook 1 selects from the behavioral data alone (stress elbow;
95% of the held-out prediction gain). The per-cohort behavioral RDMs are deliberately left out: s2 does not depend on cohort,
and SHINE questions belong to `analysis/s1_pre_vs_post_shine/`.

**The deep tree is pooled:** leaf depths 4 and 5 (both below the basic level) form one tier, so LCA depth
is binned {0, 1, 2, 3, >=4} and pairs fall into three branch-depth strata (3x3, 3x>=4, >=4x>=4).

```
s2_semantic_hierarchy/
  common.py                  RDM set and loading, pooled tree bins and strata, NaN-aware rank agreement
  0a_tree_shape              observational: the tree itself, raggedness, pooled bins   (tree_shape.py)
  0b_joint_behavior          observational: the joint behavioral RDM, missing data, noise model
                             (test-retest, Nili and group-size ceilings), external references,
                             semantic + sensory decomposition                           (joint_behavior.py)
  1_mds_dimensionality       which embeddings of behav_joint to carry forward, chosen
                             without the tree; does embedding move it toward the tree?  (mds_dimensionality.py)
  2_rank_agreement           how much rank structure each RDM shares with the tree    (rank_agreement.py)
  3_depth_gradient           does distance grow with hierarchical separation?        (depth_gradient.py)
  4_level_by_level           which levels each RDM recovers (H1b)                     (level_separation.py)
  5_semantic_vs_lowlevel     is the agreement semantic or pixel-driven?               (semantic_vs_lowlevel.py)
  6_tree_recovery            does each RDM's own clustering reproduce the tree?       (tree_recovery.py)
  KEY_FINDINGS.md
  __tests__/                 tests for the modules above
```

Figures share one design language (the RDM section of `analysis/utils/plotting.py`): hue = RDM family, marker symbol =
variant (pre circle, post diamond, joint star, variant-free square), open marker = MDS embedding (lighter
for the low-d one).

**Run** each notebook from this directory with the RDMs built
(`python -m analysis.s0_build_rdms.build_all --validate` from the repo root), in order: notebook 1 writes
`D_behav_joint_mds_low.npy` and `D_behav_joint_mds_high.npy`, which notebooks 2 to 6 load. Bootstraps, permutation nulls and MDS fits are cached
under `analysis/results/semantic_hierarchy/` (gitignored). That `D_sem_km` is exactly the tree is checked in
`s0_build_rdms/validate_rdms.py`, not here.
