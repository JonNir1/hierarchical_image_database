# Workflow

Operational checklist for executing the analysis plan registered in
[`docs/OSF_PRE_REG__draft.md`](./OSF_PRE_REG__draft.md). This file is **not** part
of the OSF submission — it's the implementation-side companion for tracking what
has been done and what is pending.

## Conventions
- `code: <path>` = where the implementation lives (or should live)
- `out: <artifact>` = expected output, stored in `analysis/results/`
- Iteration counts (same everywhere unless noted):
  - bootstrap = 5k subject-resamples
  - permutation = 10k shuffles
  - Gromov-δ subsample = 10⁵ 4-tuples
- **Embedders** (CLIP now; VGG-16 and SGPT planned, see Phase 0) persist their raw
  `(725, embedding_dim)` feature matrix as `E_<name>.npy` via `common.save_embeddings()`,
  separately from the derived `D_<name>.npy` condensed-distance RDM. Embeddings are
  the non-lossy artifact (re-deriving a distance metric doesn't require re-running
  the encoder); distances are computed from them via the shared, embedder-agnostic
  `common.euclidean_distances()` / `common.cosine_distances()` helpers, with
  `save_result=True` persisting the RDM in the same call. `analysis/rdms/sensory.py`
  (raw pixels, not a learned embedder) uses `euclidean_distances()` for the same
  reason but does not persist its pixel matrix — at 175×175×3 it would dwarf the RDM.

---

## Phase 0 — Pre-launch prep (no data yet)
- [ ] Resize images. Images should already be resized to 175×175 px, but need to make sure.
      `code: image_processing/run_resize.py`.
      input: `images/pre_shine/`, output: `images/pre_shine/`.
- [ ] SHINE images. `code: image_processing/run_shine.py`.
      Operations: `lumMatch` + `histMatch`. **Must preserve filenames and directory
      structure** so per-image pre/post correspondence is a filename match.
      input: `images/pre_shine/`, output: `images/post_shine/`
- [ ] Dataset manifest. `code: image_processing/build_dataset_manifest.py`.
      Build a manifest CSV with one row per image, including source attribution,
      Kiani-Mur labels, WordNet labels, and a per-image **text description**
      (~1 sentence; consumed by SGPT in the exploratory semantic RDM step below).
      Output: `analysis/results/dataset_manifest.csv`. (Distinct from the task's `stimuli_manifest.json`.)
- [ ] Sensory RDMs. `code: analysis/rdms/sensory.py`.
      flatten images, calculate pairwise Euclidean. No luminance
      renorm (SHINE already handles it). Out: `D_sens_pre.npy`, `D_sens_post.npy`.
- [ ] Semantic RDM (Kiani-Mur). `code: analysis/rdms/semantic_km.py`.
      Per pair: sum of mismatch indicators across the 3 KM tree levels
      (animate/inanimate, mid-level, basic). Out: `D_sem_km.npy`.
- [ ] Semantic RDM (WordNet). `code: analysis/rdms/semantic_wn_dir.py`.
      Synsets assigned directly from image category labels (filename stem + directory,
      with manual overrides for polysemous cases; see `images/manifest.csv` column
      `wn_synset_name`). Pairwise WordNet shortest-path distance. Out: `D_sem_wn.npy`.
      **`wn_synset_name` is the reference concept assignment for the dataset.** Any
      later analysis needing a per-image concept reads that column; do not re-derive
      concepts from the path or from a classifier. Automatic derivation disagrees with
      it for 376 of 717 resolvable images (median 9 WordNet edges) and drops to 3.7%
      agreement on the human branch. Evidence: `analysis/rdms/imagenet_vs_path.py`.
- [ ] Visual-semantic RDM (CLIP). `code: analysis/rdms/clip.py`.
      Load OpenAI pretrained CLIP **ViT-B/32** (via `open_clip`), using the
      `ViT-B-32-quickgelu` config — **not** plain `ViT-B-32`, which pairs OpenAI's
      QuickGELU-trained weights with standard-GELU activations and silently shifts
      the RDM (Spearman ρ ≈ 0.94 vs the correct config). See the module docstring
      in `clip.py` and the implementation note in the pre-reg's Indices section.
      Encode each of the 725 images via the image encoder, take the **output-layer**
      embedding (Shoham et al. 2024 spec). Embeddings are persisted first, then
      collapsed to pairwise **cosine** distance via the shared `cosine_distances()`
      helper (see "Embedders" convention below). Run separately for pre-SHINE and
      post-SHINE images.
      Out: `E_clip_pre.npy`, `E_clip_post.npy` (raw embeddings),
      `D_clip_pre.npy`, `D_clip_post.npy` (condensed RDM).
- [ ] (Exploratory) High-level visual RDM (VGG-16). `code: analysis/rdms/visual_vgg.py`.
      Load ImageNet-pretrained VGG-16 from `torchvision`. Forward-pass each image and
      extract **FC7 penultimate-layer** activations (Shoham et al. 2024 spec).
      Pairwise **cosine** distance via `cosine_distances()`. Run pre and post SHINE.
      Out: `E_vgg_pre.npy`, `E_vgg_post.npy`, `D_vgg_pre.npy`, `D_vgg_post.npy`.
- [ ] (Exploratory) SGPT-based semantic RDM. `code: analysis/rdms/semantic_sgpt.py`.
      Requires per-image text descriptions in the dataset manifest (Shoham et al.
      used first-paragraph Wikipedia / dictionary definitions). Encode each description
      with **SGPT-1.3B-msmarco-mean-tokens** (bi-encoder), take output layer.
      Pairwise **cosine** distance via `cosine_distances()`. Variant-agnostic — one
      RDM, used for both cohorts.
      Out: `E_sem_sgpt.npy`, `D_sem_sgpt.npy`.
- [ ] Reference-RDM diagnostics vs the curated hierarchy.
      `code: analysis/rdms/hierarchy_comparison.py` + `analysis/rdms/compare_to_hierarchy.ipynb`.
      Confirms `D_sem_km` reproduces the directory tree exactly (independent
      reimplementation of the LCA computation, so the check is not a tautology),
      then asks how far each non-KM RDM tracks that tree. Two distinct questions,
      deliberately plotted as two separate figures: pairwise rank agreement
      (`plot_correlation_matrix`) and distance-vs-hierarchical-depth
      (`plot_depth_profile`). Inference uses image-label shuffling, never analytic
      p-values, since the 262,450 condensed entries come from only 725 images.
      Not a gate on anything; run it after the reference RDMs are built and before
      committing to a `D_sem` source for RQ3.
- [ ] SHINE manipulation check. `code: analysis/rdms/manip_check.ipynb`.
      Verify `D_sens_post` has much lower variance in luminance + color-histogram
      moments than `D_sens_pre`. If not, abort and re-run SHINE.
- [ ] OSF. Submit pre-reg to OSF, get timestamp.
- [ ] Code-freeze the `SpAM_Task/` at OSF submission timestamp (git tag `osf-freeze-v1`).
- [ ] Prolific + Pavlovia setup. Single Pavlovia/Prolific study; cohort assignment is
      automatic (PID hash). No blocklist required.
      Configure `task_config.json`: `deployment.mode = "pilot"` for pilot run,
      then `"production"` for full data collection. Set real `prolific_completion_url`.
      Set Pavlovia status to RUNNING.

## Phase 1 — Data collection & QC

### Phase 1a — Pilot (mode: `"pilot"`, ~15-20 participants, mostly pre-SHINE)
- [ ] Launch pilot study with `deployment.mode = "pilot"` in `task_config.json`.
      `task.js` assigns pilot participants the pre-SHINE images.

> **The collected pilot data does not match that rule.** Of the 47 loadable pilot
> subjects, **41 are `pre` and 6 are `post`** (all v3.06) — some pilot sessions ran
> under a build that did not pin the variant. So `cohort == "pilot"` is **not** a
> pre-SHINE filter, and anything variant-sensitive must filter on `shine_variant`
> explicitly (`load_pilot_subjects(..., variants=("pre",))`). Ground-truth
> construction does; noise-model fitting deliberately does not, since it estimates
> a property of subjects rather than of the images.
- [ ] Download data from Pavlovia. Inspect QC pass-rates, trial RT distributions,
      image loading, and CSV format. Fix any issues before full launch.
- [ ] After pilot sign-off: set `deployment.mode = "production"` in `task_config.json`,
      commit, and push to Pavlovia. Exclude pilot PIDs from Prolific before relaunching.

### Phase 1b — Production (mode: `"production"`, both cohorts concurrently)
- [ ] Launch production study. Both cohorts recruited simultaneously; assignment is
      automatic from PID hash (even → pre-SHINE, odd → post-SHINE).
- [ ] Monitor cohort sizes periodically by downloading Pavlovia data and counting
      retained subjects per `shine_variant` value. Continue until both cohorts reach
      N=75 retained subjects with strongly connected dissimilarity graphs.
      Check connectivity in batches of 5 subjects after reaching N=75 per cohort.
- [ ] Final data export after both cohorts close. Snapshot raw session CSVs to
      `analysis/data/pre_shine_raw/` and `analysis/data/post_shine_raw/` (split by
      `shine_variant` column). Verify row counts match Pavlovia session counts.
- [ ] Report quality control statistics: mean & SD of reaction time, number of moves,
      QC thresholds & pass-rates, etc. `code: analysis/pipeline/qc_report.py`
- [ ] Calculate within-cohort split-half reliability as a diagnostic.
      Report in final paper, but does not affect stopping.
      `code: analysis/pipeline/reliability_cohort.py`
- [ ] Calculate within-subject reliability index for each subject (Spearman ρ on the
      50 repeated images), with and without 10th percentile exclusion.
      `code: analysis/pipeline/reliability_subject.py`
- [ ] Checkpoint :: Compare data collection against pre-reg. Document any deviations
      (e.g., recruitment shortfalls, technical issues, exclusion-rate surprises) as
      an OSF amendment alongside the registered version.

## Phase 2 — Per-cohort RDM construction
- [ ] Population RDM via unweighted metric MDS. `code: analysis/pipeline/mds.py`.
      Out: `D_perc_pre.npy`, `D_perc_post.npy`.
- [ ] Noise ceiling. `code: analysis/pipeline/noise_ceiling.py`.
      LOSO Spearman per cohort (Nili 2014). Out: `nc_pre.json`, `nc_post.json`.

## Phase 3 — Confirmatory + planned-secondary analyses
*Tags: `[CONFIRM]` = pre-registered confirmatory test, FDR-corrected within each RQ family;
`[SECONDARY]` = pre-registered robustness/replication, reported but not the primary inference;
`[EXPLORATORY]` = not part of the pre-registered confirmatory plan.*
*FDR correction (BH) applies within each RQ family across all `[CONFIRM]` items only.*

- [ ] `[CONFIRM]` RQ1a — Gromov δ on `D_perc_pre`. Compare to δ on `D_sem_km`, `D_sens_pre`
      via bootstrap CIs. `code: analysis/pipeline/rq1a_gromov.py`.
- [ ] `[CONFIRM]` RQ1b — Spearman ρ(D_perc_pre, D_sem_km), full + level-stratified.
      10k label-shuffle perms. `code: analysis/pipeline/rq1b_kiani_mur.py`.
- [ ] `[EXPLORATORY]` RQ1b-replication / H1c — Spearman ρ(D_perc_pre, D_sem_wn), full + level-stratified.
      10k label-shuffle perms. `code: analysis/pipeline/rq1b_wordnet.py`.
- [ ] `[CONFIRM]` RQ2a — Spearman ρ(D_perc_pre, D_perc_post) vs image-label shuffle null
      AND vs cross-subject shuffle null. Decode 2×2 verdict per main draft.
      `code: analysis/pipeline/rq2a_shine_effect.py`.
- [ ] `[EXPLORATORY]` RQ2c / H2c — bootstrap CI of
      ρ(pre, D_sem_km) − ρ(post, D_sem_km).
      `code: analysis/pipeline/rq2c_km_drift.py`.
- [ ] `[EXPLORATORY]` RQ2d / H2d — level-stratified ρ(pre, post),
      level × condition test. `code: analysis/pipeline/rq2d_level.py`.
- [ ] `[EXPLORATORY]` RQ2c+d WordNet replication — repeat RQ2c and RQ2d with WordNet D_sem.
      `code: analysis/pipeline/rq2c_wordnet.py` and `rq2d_wordnet.py`.
- [ ] `[CONFIRM]` RQ3 — 3-predictor NNLS fit per cohort:
      `D_perc = α·D_sem_km + β·D_clip + γ·D_sens + ε`.
      Bootstrap CIs (5k subject resamples) on (α, β, γ) and the pre−post differences.
      Confirmatory predictions on α (semantic preserved) and γ (sensory reduced).
      β (CLIP / visual-semantic) reported descriptively, no directional claim.
      Additional reporting per Shoham et al. 2024 protocol:
        • Hierarchical MLR: ΔR² for D_sens → +D_clip → +D_sem_km
        • Partial correlations for each predictor controlling for the other two
        • Fisher-z transform + FDR across the three predictors
      `code: analysis/pipeline/rq3_decomp.py`.
- [ ] `[SECONDARY]` RQ3 / WN — same with D_sem = D_sem_wn (supplementary
      robustness check; reported alongside KM but not required to support H3).
      `code: analysis/pipeline/rq3_decomp_wn.py`.
- [ ] `[EXPLORATORY]` RQ3 / SGPT — same with D_sem = D_sem_sgpt (Shoham et al.
      2024 exact specification). `code: analysis/pipeline/rq3_decomp_sgpt.py`.
- [ ] `[EXPLORATORY]` RQ3 / 4-predictor — add D_VGG (FC7 cosine) as a 4th predictor:
      `D_perc = α·D_sem_km + β·D_clip + γ·D_vgg + δ·D_pix + ε`.
      Tests whether SHINE selectively reduces δ (pixel) while preserving γ (VGG).
      Reported descriptively only. `code: analysis/pipeline/rq3_decomp_4pred.py`.

## Phase 4 — Exploratory (Optional)
- [ ] S1 hyperbolic embedding (Poincaré / HyPoE replication of Phases 2-3).
- [ ] S2 per-item Procrustes displacement + top-5 NN Jaccard.
- [ ] S3 alt sensory metrics: Gabor-bank, corneal/V1.

## Phase 5 — Write-up
- [ ] Dataset deliverable: condensed RDMs, MDS embeddings, hierarchy labels,
      analysis code → OSF/Zenodo with DOI.
- [ ] Manuscript draft. Compare against the pre-reg point-by-point; any
      deviation goes into the Discussion as an explicit deviation log.
