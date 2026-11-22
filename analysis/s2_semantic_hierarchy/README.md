# `s2_semantic_hierarchy/`

How the **reference RDMs** (WordNet, CLIP, pixels) relate to the curated directory-tree
hierarchy (the Kiani-Mur tree under `images/<variant>/`). No human perceptual data is involved:
this is a comparison of stimulus-derived RDMs with each other and with the tree, run before
committing to a `D_sem` source for RQ3. (How the *cohorts'* behavior relates to the tree is
`analysis/s1_pre_vs_post_shine/4_semantic_hierarchy.ipynb`.)

```
s2_semantic_hierarchy/
  rdm_vs_hierarchy.py      pairwise rank agreement, depth profile, partial correlation,
                           incremental R^2, per-level separation, and their figures
  rdm_vs_hierarchy.ipynb   front end: tree shape, correlation matrix, depth profile,
                           shuffle null + pixel control, CLIP vs WordNet, level by level
  __tests__/               tests for the module above
```

Status: moved here from `s0_build_rdms/` unchanged apart from paths; to be revisited (which
variant, pre or post, is canonical here; whether the level-by-level view stays). That `D_sem_km`
is exactly the tree is checked in `s0_build_rdms/validate_rdms.py`, not here.
