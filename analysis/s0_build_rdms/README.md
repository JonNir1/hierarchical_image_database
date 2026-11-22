# `s0_build_rdms/`

Code that **builds** the RDMs the analysis consumes, plus the methodological checks that
validate them (which aggregation to use, why WordNet is built by direct assignment, that
`D_sem_km` is exactly the directory tree). One subfolder per RDM type, each holding the
builder(s) for that type; shared infrastructure stays at this level. Cohort comparisons
(pre vs post SHINE) live in `analysis/s1_pre_vs_post_shine/` and the reference-RDM-vs-hierarchy
diagnostics in `analysis/s2_semantic_hierarchy/`, not here.

```
s0_build_rdms/
  common.py              paths, manifest loading + guards, save/load RDMs and
                         embeddings, the order_hash alignment contract
  build_all.py           CLI to build any or all reference RDMs (--validate runs the checks after)
  validate_rdms.py       invariant checks over whatever has been built, including that
                         D_sem_km equals the directory tree under an independent LCA
  KEY_FINDINGS.md        findings about the builders (aggregation choice, ImageNet vs path)
  __tests__/             all tests for this package, flat (see note below)

  behavior/   spam_ratings.py     SpAM trials -> 725x725 RDM (3 aggregations)
              mds.py              MDS embedding layer over those RDMs
              noise_ceiling.py    leave-one-out / split-half reliability bounds
              generalizability.py agreement with a held-out-subject RDM
              aggregation_comparison.ipynb   picks the aggregation method
              cache/              its CSV outputs (gitignored: participant ids)

  semantic/   kiani.py                   Kiani-Mur directory-tree path length
              wordnet.py                 WordNet shortest path
              assign_wordnet_synsets.py  the reference concept assignment it reads
              clip.py                    CLIP ViT-B/32 cosine (visual-semantic)
              imagenet_vs_path.py        ResNet-50 vs path-derived concepts: why WordNet is
                                         built by direct assignment, not from ImageNet classes
              imagenet_vs_path.ipynb     its table and plots
              imagenet_class_index.json  its data file (class index -> wnid)

  sensory/    pixels.py           pixel Euclidean, pre and post SHINE
```

`clip.py` sits under `semantic/` rather than `sensory/` because D_CLIP acts as a
semantic-side predictor in the H3 decomposition. `sensory/` is specifically the
pixel-level matrix that SHINE manipulates, which is the contrast H3 is built on.

Tests are kept flat in `__tests__/` rather than mirrored per subfolder: one
`conftest.py`, one pytest root, and `test_<module>.py` already maps obviously to
its module.
