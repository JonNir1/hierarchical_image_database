# Key findings

Short, accurate bullets, each pointing to the notebook (and commit) that produced it.

What did the SpAM pilot show about task versions v1 to v4.01? Data: `data/` as of 2026-11-24, 280 participants
and 5,252 trials, task versions v1.0, v2.0, v3.0, v3.06, v4.0, v4.01. Comparisons are pairwise Mann-Whitney U on
per-subject aggregates (one value per subject), Bonferroni-corrected across the 15 version pairs, with
rank-biserial r as the effect size (positive r: the first-named version has the smaller values). Numbers below
come from a sandbox re-run of the notebook on the current data. The notebook's stored outputs predate it
(earlier data, an older print format) and were not overwritten: the new run also stores 10 figures and grows
the file from 0.1 MB to 5.6 MB.

- **Caveat, version confound:** each version changed the task (time floor, repeats, screening), so a version
  difference is a bundle of design changes, not one manipulation. Cohorts are also unequal (v4.01 has 154 participants, v2 has 10).
- **Caveat, frozen QC thresholds:** the QC section tests the thresholds that were in `SpAM_Task/task_config.json`
  when the analysis was written. The config has since changed (see the QC section), so re-running it does not test today's config.

## Who took part, and who was kept?

Notebook: [`analysis.ipynb`](analysis.ipynb) (commit `5562e37f`)

- **Status:** full data 203, revoked consent 34, screened out 18, false_positive 15, missing data 10.
- **N by version (total / analytic):** v1 16 / 16, v2 10 / 10, v3 12 / 11, v3.06 14 / 11, v4.0 41 / 34, v4.01 154 / 121.
- **Screening, v4.0 and later:** 2 of 41 screened out in v4.0 (4.9%) and 16 of 154 in v4.01 (10.4%).
- **Caveat:** `false_positive` is a status the parser produces now and the notebook's text does not explain; participants with that status are neither full data nor screened out.

## Do versions differ in time and effort?

Notebook: [`analysis.ipynb`](analysis.ipynb) (commit `5562e37f`)

- **Trial duration:** median 86.5 s (v1), 65.4 (v2), 64.9 (v3), 68.6 (v3.06), 70.4 (v4), 72.8 (v4.01). v1 has trials as short as 0.94 s; from v2 the minimum is about 60 s (the time floor).
- **Duration, tested:** only v2 vs v4.01 survives correction (p corrected = 0.036, r = 0.58).
- **Moves per trial:** median 24 (v1), 27, 26.5, 23, 26, 24 (v4.01). No pair differs (smallest corrected p = 0.86).
- **Idle tail (fraction of the trial after the last move):** lower in v1 (median 0.042) than in v4 (0.091, p corrected = 0.0014, r = 0.68) and v4.01 (0.082, p corrected = 0.0003, r = 0.65). All other pairs are not significant.

## How far are arrangements from random placement?

Notebook: [`analysis.ipynb`](analysis.ipynb) (commit `5562e37f`)

- **Null:** 1,000 simulated trials of 20 dots, 190,000 distances, mean 0.369 and SD 0.175 (`analysis/utils/null_distances.py`, seed 42).
- **KS distance D from the null, per trial then averaged per subject:** median 0.268 (v1), 0.296 (v2), 0.262 (v3), 0.276 (v3.06), 0.234 (v4), 0.232 (v4.01).
- **Tested:** v4 has smaller D than v1, v2, v3 and v3.06 (p corrected 0.0072, 0.0027, 0.013, 0.0039; r from -0.60 to -0.78), and v4.01 has smaller D than v1, v2 and v3.06 (p corrected 0.0099, 0.0021, 0.0084). v3 vs v4.01 is borderline (p corrected = 0.052). v4 and v4.01 do not differ (p corrected = 1).
- **Caveat:** D is the maximum gap between two cumulative distributions, so it is sensitive to shape, not to how far the distances sit from the null.

## Is test-retest reliability different across versions?

Notebook: [`analysis.ipynb`](analysis.ipynb) (commit `5562e37f`)

- **Per-subject median Spearman r of verbatim repeats (v3 and later):** 0.206 (v3), 0.286 (v3.06), 0.261 (v4), 0.247 (v4.01).
- **Tested:** no pair differs (smallest corrected p = 0.69).
- **v1 vs v2 SNR** (sigma_d / mean |d1 - d2| over incidentally repeated pairs): medians 1.41 and 1.26, not significant (p = 0.44, r = -0.19). v3 and later are excluded because their reliability measure is different (see the notebook).

## Do the QC thresholds hold up?

Notebook: [`analysis.ipynb`](analysis.ipynb) (commit `5562e37f`)

- **Catch trials, frozen thresholds (cluster mean 0.15, cluster SD 0.10, distance to target 0.20):** no catch trial exceeds any of them in any version (largest cluster mean 0.142). The recorded `qc_flag` was true for all 20 v2 catch trials and none elsewhere (3% overall).
- **Main trials, frozen thresholds (pairwise SD 0.04, move ratio 0.75):** no trial falls below the SD floor (minimum 0.063); 4.9% of trials fall below the move-ratio floor (v1 11.9%, v2 1.0%, v3 9.1%, v3.06 13.2%, v4 4.9%, v4.01 3.8%).
- **Config drift:** `SpAM_Task/task_config.json` now has `min_pairwise_distance_sd` 0.1, `min_move_item_ratio` 0.65 and `location_tolerance` 0.15, and no catch-trial cluster checks. The tables above test the older values.
- **Caveat:** the `qc_flag` column records whichever thresholds were active when each session ran.
