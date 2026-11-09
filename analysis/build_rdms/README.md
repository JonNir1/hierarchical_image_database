# `build_rdms/`

Code that **builds** the RDMs the analysis consumes. One subfolder per RDM type,
each holding the builder(s) for that type; shared infrastructure stays at this level.

```
build_rdms/
  common.py              paths, manifest loading + guards, save/load RDMs and
                         embeddings, the order_hash alignment contract
  build_all.py           CLI to build any or all reference RDMs
  validate_rdms.py       invariant checks over whatever has been built
  __tests__/             all tests for this package, flat (see note below)

  behavior/   behavioral.py       SpAM trials -> 725x725 RDM (3 aggregations)
              mds.py              MDS embedding layer over those RDMs
              noise_ceiling.py    leave-one-out / split-half reliability bounds
              generalizability.py agreement with a held-out-subject RDM
              aggregation_comparison.ipynb   picks the aggregation method

  semantic/   semantic_km.py      Kiani-Mur directory-tree path length
              semantic_wn_dir.py  WordNet shortest path
              assign_wn_synsets.py  the reference concept assignment it reads
              clip.py             CLIP ViT-B/32 cosine (visual-semantic)

  sensory/    sensory.py          pixel Euclidean, pre and post SHINE
```

`clip.py` sits under `semantic/` rather than `sensory/` because D_CLIP acts as a
semantic-side predictor in the H3 decomposition. `sensory/` is specifically the
pixel-level matrix that SHINE manipulates, which is the contrast H3 is built on.

Tests are kept flat in `__tests__/` rather than mirrored per subfolder: one
`conftest.py`, one pytest root, and `test_<module>.py` already maps obviously to
its module.

## TODO: these are not builders and should move

The following are inference / hypothesis-testing or diagnostics. They live here
only for historical reasons and should move once there is a home for them
(`analysis/inference/`, or per-question folders). Left in place deliberately, so
the move is one decision rather than several:

| file | what it is |
|---|---|
| `run_h1.py` | driver for Q1 analysis 1, both cohorts |
| `model_evaluation.py` | model vs behavioral RDM, Q1 analysis 1a |
| `hierarchical_clustering.py` | perceptual vs KM-tree agreement, Q1 analysis 1b |
| `hierarchy_comparison.py` | reference-RDM diagnostics against the path hierarchy |
| `imagenet_vs_path.py` | ImageNet vs path-derived concept distance (+ its `imagenet_class_index.json`) |
| `h1_inference.ipynb` | plots Q1 analysis 1 results |
| `compare_to_hierarchy.ipynb` | reference RDMs vs the curated hierarchy |

`imagenet_class_index.json` is `imagenet_vs_path.py`'s own data file, loaded via
`Path(__file__).parent`, so the two move together.
