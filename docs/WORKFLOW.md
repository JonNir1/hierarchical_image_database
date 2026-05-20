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

---

## Phase 0 — Pre-launch prep (no data yet)
- [ ] Resize images. Images should already be resized to 100×100 px, but need to make sure.
      `code: image_processing/run_resize.py`.
      input: `images/pre_shine/`, output: `images/pre_shine/`.
- [ ] SHINE images. `code: image_processing/run_shine.py`.
      Operations: `lumMatch` + `histMatch`. **Must preserve filenames and directory
      structure** so per-image pre/post correspondence is a filename match.
      input: `images/pre_shine/`, output: `images/post_shine/`
- [ ] Dataset manifest. `code: image_processing/build_dataset_manifest.py`.
      Build a manifest CSV with one row per image, including source attribution and Kiani-Mur & WordNet labels.
      Output: `analysis/results/dataset_manifest.csv`. (Distinct from the task's `stimuli_manifest.json`.)
- [ ] Sensory RDMs. `code: analysis/rdms/sensory.py`.
      flatten images, calculate pairwise Euclidean. No luminance
      renorm (SHINE already handles it). Out: `D_sens_pre.npy`, `D_sens_post.npy`.
- [ ] Semantic RDM (Kiani-Mur). `code: analysis/rdms/semantic_km.py`.
      Per pair: sum of mismatch indicators across the 3 KM tree levels
      (animate/inanimate, mid-level, basic). Out: `D_sem_km.npy`.
- [ ] Semantic RDM (WordNet). `code: analysis/rdms/semantic_wn.py`.
      Top-1 ImageNet (ResNet-50 / CLIP) classification of each image → WordNet synset →
      shortest-path distance in the WordNet hypernym graph. Out: `D_sem_wn.npy`.
- [ ] SHINE manipulation check. `code: analysis/rdms/manip_check.ipynb`.
      Verify `D_sens_post` has much lower variance in luminance + color-histogram
      moments than `D_sens_pre`. If not, abort and re-run SHINE.
- [ ] OSF. Submit pre-reg to OSF, get timestamp.
- [ ] Code-freeze the `SpAM_Task/` at OSF submission timestamp (git tag `osf-freeze-v1`).
- [ ] Prolific + Pavlovia setup. Two studies (pre-SHINE first, post-SHINE second).
      Configure `task_config.json`: `debug=false`, real `prolific_completion_url`.
      Set Pavlovia status to RUNNING. Configure blocklist on the post-SHINE study.

## Phase 1 — Data collection & QC
- [ ] Launch pre-SHINE study (N=75 retained). Monitor QC pass-rate via Pavlovia
      data exports.
- [ ] Connectivity check at N=75: confirm dissimilarity graph is strongly connected.
      If not, add subjects in batches of 5 until it is.
- [ ] Final data export from Pavlovia after the cohort closes. Snapshot raw session
      CSVs to `analysis/data/pre_shine_raw/`. Verify row counts match Pavlovia session counts.
- [ ] Report quality control statistics: mean & SD of reaction time, number of moves, QC thresholds & pass-rates, etc.
      `code: analysis/pipeline/qc_report.py`
- [ ] Calculate within-cohort split-half reliability as a diagnostic.
      Report in final paper, but does not affect stopping.
      `code: analysis/pipeline/reliability_cohort.py`
- [ ] Calculate within-subject reliability index for each subject (Spearman ρ on the 50 repeated images),
      with and without 10th percentile exclusion. Report distribution in final paper.
      `code: analysis/pipeline/reliability_subject.py`
- [ ] Launch post-SHINE study (with pre-SHINE PIDs as Prolific blocklist). Snapshot raw
      to `analysis/data/post_shine_raw/`. Repeat all previous Phase 1 steps for the post-SHINE cohort.
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
- [ ] `[CONFIRM]` RQ3 / KM — NNLS fit per cohort with D_sem = D_sem_km. Bootstrap CIs on α, β, diffs.
      `code: analysis/pipeline/rq3_decomp_km.py`.
- [ ] `[SECONDARY]` RQ3 / WN — same as above with D_sem = D_sem_wn (supplementary
      robustness check; reported alongside KM but not required to support H3).
      `code: analysis/pipeline/rq3_decomp_wn.py`.

## Phase 4 — Exploratory (Optional)
- [ ] S1 hyperbolic embedding (Poincaré / HyPoE replication of Phases 2-3).
- [ ] S2 per-item Procrustes displacement + top-5 NN Jaccard.
- [ ] S3 alt sensory metrics: Gabor-bank, corneal/V1.

## Phase 5 — Write-up
- [ ] Dataset deliverable: condensed RDMs, MDS embeddings, hierarchy labels,
      analysis code → OSF/Zenodo with DOI.
- [ ] Manuscript draft. Compare against the pre-reg point-by-point; any
      deviation goes into the Discussion as an explicit deviation log.
