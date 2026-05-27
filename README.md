# hierarchical_image_database
Building a database of real-world objects in a neutral background that can be used for neuroscientific studies, along with a concept hierarchy for the objects and similarity scores for image pairs

## Research Questions

1. **Hierarchy validation** — does the curated 725-image set exhibit hierarchical perceptual structure that resembles the Kiani-Mur (2007, 2013) hierarchy?
2. **SHINE effect** — does SHINE-color preprocessing (required for EEG compatibility) preserve that perceptual structure or distort it?
3. **Decomposition shift** — when perceptual distance is modelled as a weighted combination of semantic and sensory distances, does SHINE selectively reduce the sensory weight while leaving the semantic weight intact?

## Analysis Plan (summary)

- Collect SpAM data via Prolific/Pavlovia for two between-subject cohorts (N=75 each; one pre-SHINE, one post-SHINE).
- Build population RDMs per cohort via unweighted metric MDS (R `smacof` via `rpy2`; binary pair weights — 1 if observed, 0 otherwise).
- **RQ1a (any hierarchy?)** — normalised Gromov δ-hyperbolicity; compared against semantic (tree-like) and sensory (Euclidean) baselines.
- **RQ1b (matches Kiani-Mur?)** — Spearman/Mantel ρ between perceptual RDM and a Kiani-Mur categorical-distance RDM; stratified by tree level.
- **RQ2a (SHINE preserves vs perturbs)** — Spearman ρ(pre, post) tested against (i) an image-label shuffle null for shared structure and (ii) a cross-subject shuffle null for SHINE-specific perturbation.
- **RQ2b** — level-stratified Spearman ρ(pre, post); level × condition interaction.
- **RQ2c** — bootstrap CI on [ρ(pre, KM) − ρ(post, KM)].
- **RQ3 (decomposition)** — fit `D_perc = α·D_sem + β·D_sens` per cohort; bootstrap CIs on (α_pre − α_post) and (β_pre − β_post), under both Kiani-Mur and WordNet `D_sem`.
- Multiple comparisons via FDR within each RQ family.
- Exploratory: hyperbolic-space replication; item-level Procrustes; alternate sensory metrics.

**Planning documents** (in [`docs/`](./docs/)):
- [`docs/OSF_PRE_REG__draft.md`](./docs/OSF_PRE_REG__draft.md) — full pre-registration draft (hypotheses, design, sampling, analysis plan).
- [`docs/WORKFLOW.md`](./docs/WORKFLOW.md) — operational checklist for executing the analysis plan; not part of the OSF submission.

Full pre-registration on OSF: *[link, populated after submission]*.

## Dataset Layout

The 725-image curated dataset lives at `<repo>/images/`, with one subdirectory per
SHINE variant and a per-image manifest:

```
images/
  README.md                                      # dataset documentation (outward-facing)
  manifest.csv                                   # per-image source attribution + KM labels
  pre_shine/  <cat1>/<cat2>/.../<name>NN.png     # original images
  post_shine/ <cat1>/<cat2>/.../<name>NN.png     # SHINE-color processed (same filenames as pre_shine)
```

Image files are tracked on both `main` (GitHub) and `pavlovia_deploy` (Pavlovia gitlab).
The hierarchy is encoded in the directory tree, mirrored across the two variants so
per-image pre/post pairing is a filename match.

See [`images/README.md`](./images/README.md) for dataset provenance, curation criteria,
and per-source attribution. The CSV manifest at `images/manifest.csv` is the
machine-readable index used by analysis sub-modules.

`SpAM_Task` reads the active variant via `stimuli_paths.main_root` +
`shine.shine_variant` in `task_config.json`.

## Source Datasets

The directory [`source_datasets/`](./source_datasets/) groups the original image sets
from which our 725-image set was curated:

```
source_datasets/
  Grootswagers 2019/   .gitkeep         # Grootswagers et al., 2019, NeuroImage
  Kiani 2007/          .gitkeep         # Kiani et al., 2007 (with Kriegeskorte 2008 / Mur 2013 PDFs)
  Kiani_ImageSet/      .gitkeep
  PiCS 2025/           .gitkeep         # PiCS dataset, Robbins et al. 2025
  face place/          .gitkeep         # Face-Place, Righi et al. 2012
  huji_flowers/        .gitkeep
```

**Source images themselves are NOT redistributed** — they belong to their respective
owners under their own licenses. Only the directory structure + `.gitkeep` stubs are
tracked. Re-obtain the originals from each source's official channel; the curation
mapping is in `images/manifest.csv`.

## Repository Layout

```
index.html                 ← Pavlovia entry point; loads scripts from SpAM_Task/
SpAM_Task/                 ← task code (js/, jspsych/, assets/, task_config.json, ...)
images/                    ← canonical 725-image curated dataset (tracked on both branches)
source_datasets/           ← original datasets (structure tracked; image files NOT redistributed)
docs/                      ← planning documents (pre-reg draft + workflow checklist)
SpAM_Simulations/          ← MDS simulations (developer-only; not on Pavlovia)
visualize_dataset/         ← dataset visualization tools (developer-only; not on Pavlovia)
analysis/                  ← future: analysis notebooks (developer-only; not on Pavlovia)
```

The `main` and `pavlovia_deploy` branches share an identical structure for the task and
dataset. `pavlovia_deploy` differs only in a stricter `.gitignore` that excludes the
developer-only directories (`SpAM_Simulations/`, `visualize_dataset/`, etc.) so they
don't reach Pavlovia.
