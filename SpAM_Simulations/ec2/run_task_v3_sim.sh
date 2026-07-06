#!/usr/bin/env bash
#
# Provision an EC2 instance, run the full SpAM MDS sweep for the TASK-V3 simulation - the
# generative, coordinate-space observation model (per-subject PC "perspective" + a local 2-D
# arrangement projection per trial + canvas placement noise on the projected 2-D positions,
# replacing the additive-distance-noise model of v0.1/v2.3/v2.4) - and upload the results to S3.
# subjects_noise_scale is that placement jitter as a ratio of a trial's 2-D arrangement spread
# (post-projection), so within-subject test-retest is independent of perspective_dispersion.
#
# Task v3.0 (SpAM_Task/js/trial_generator.js) drops the frac_images_repeated cross-context lever:
# every image appears in exactly one distinct trial, and the only repetition is a verbatim
# whole-trial repeat (frac_trials_repeated, giving the within-subject test-retest reliability).
# This sweep therefore has NO frac_images_repeated, and adds two new levers:
#   * perspective_dispersion - between-subject disagreement (spread of each subject's per-PC
#     weights). 0 = everyone shares the ground-truth geometry (pure-noise model).
#   * use_isotropic (set per run below, not swept) - the ground-truth eigenvalue spectrum:
#     True  = isotropic, the conservative full-rank "every dimension equally important" bound;
#     False = geometric `decay` spectrum (+ optional hierarchical `n_clusters`), the realistic
#     case where low-variance dims rarely surface in any trial's 2-D slice. Run the sweep twice
#     (use_isotropic True and False) to bracket the required-N estimate.
#
# TWO FLAVORS (CALIBRATE):
#   * CALIBRATE=false (default) - synthetic ground truth (use_isotropic/decay) and SWEPT, guessed
#     noise/dispersion grids. This is the block below, unchanged.
#   * CALIBRATE=true            - fits the simulation to the real pilot, then sweeps the DESIGN grid:
#     it fetches the pilot from $S3_URI/data/pilot and builds the ground truth ONCE by weighted-SMACOF
#     of the pooled pilot (so decay/n_clusters/use_isotropic are moot). Then, for each per-subject
#     noise-heterogeneity df in $DF_LIST (default 5,8,10), it refits subjects_noise_scale +
#     perspective_dispersion to the v3.0 test-retest / between-subject agreement on that fixed GT and
#     sweeps num_subjects x trials_per_subject x images_per_trial at those FITTED (not swept) values.
#     Per-df outputs go to out/df<df>/ + mds_store/df<df>/, plus a combined out/plateau_by_df.csv
#     (required-N per ndim x design). Calibration artifacts (gt_pilot_coords.npy,
#     calibrated_params_df<df>.json, calibrate.log) go to $S3_URI/calibration/.
#     (The calibration is also callable programmatically: `SpAM_Simulations.pilot.calibrate_params_from_pilot`.)
#
# Sibling scripts: run_task_v2_4_sim.sh (additive-noise model + both repeat levers),
# run_task_v2_3_sim.sh, run_task_v0_1_sim.sh. All source the shared prepare_machine.sh (must be
# copied alongside - see README.md's "Running on EC2" cookbook).
#
# Target: Ubuntu 22.04/24.04 (apt + CRAN). On Amazon Linux swap the apt blocks for dnf.
#
# Prerequisites
#   * The commit/branch you want to run is PUSHED to the remote (this clones from it).
#   * The instance can reach the repo (public repo, or a PAT/deploy key in REPO_URL/ssh).
#   * S3 access: attach an IAM role with s3:PutObject on the bucket (preferred), or run
#     `aws configure` with the project IAM user's credentials before this script.
#   * For CALIBRATE=true only: stage the pilot under $S3_URI/data/pilot (PRIVATE), containing the
#     per-session + demographics CSVs and stimuli_manifest.json, e.g.:
#       aws s3 cp data/pilot/                     "$S3_URI/data/pilot/" --recursive
#       aws s3 cp SpAM_Task/stimuli_manifest.json "$S3_URI/data/pilot/"
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/task-v3-isotropic
#   export USE_ISOTROPIC=true            # (default flavor only) or false for the anisotropic spectrum
#   # export CALIBRATE=true              # opt into the pilot-calibrated flavor instead
#   # export DF_LIST=5,8,10              # (calibrate flavor) per-subject noise-heterogeneity df values
#   bash run_task_v3_sim.sh
#
# Default-flavor grid: 4 num_subjects x 3 trials_per_subject x 1 images_per_trial x 2 noise_scale x 1
# noise_df x 4 frac_trials_repeated x 3 perspective_dispersion = 288 combos x 5 reps x 9 target
# ndims = ~12960 MDS fits. Calibrate-flavor grid: 3 df x 4 num_subjects x 2 trials x 2 images = 48
# combos x 5 reps x 13 ndims (min_ndim=3) = ~3120 fits (itmax=1000, precalc_init=True). c7i.4xlarge
# (16 vCPU) fits both; the calibrate sweep is ~2.5-3 h. REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.
# REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/task-v3-isotropic}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
USE_ISOTROPIC="${USE_ISOTROPIC:-false}"   # ground-truth spectrum: true=isotropic bound, false=realistic
CALIBRATE="${CALIBRATE:-false}"           # true = fit to pilot data (see TWO FLAVORS above); ignores USE_ISOTROPIC
GT_METHOD="${GT_METHOD:-smacof}"          # (CALIBRATE only) 'smacof' (needs R, canonical) or 'classical' (no-R)
REPS="${REPS:-5}"                         # (CALIBRATE only) cohorts averaged per simulated fit point
# MDS worker processes. All vCPUs OOM'd a run (each worker holds its own R/smacof process), so
# default to 2/3 of them.
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# --------------------------------------------------------------------------- run the sweep
if [ "${CALIBRATE,,}" = "true" ] || [ "$CALIBRATE" = "1" ]; then
# ===== calibrated flavor: fetch pilot -> fit noise/perspective + pilot GT -> calibrated sweep =====
# Calibration reuses analysis/utils/parser.py (the canonical pilot loader), which prepare_machine's
# sparse checkout (SpAM_Simulations only) doesn't include - add it.
git sparse-checkout add analysis/utils

# Fetch the (gitignored, human-subjects) pilot data; delete it from the box on ANY exit.
PILOT_DIR="data/pilot"
trap 'rm -rf "$PILOT_DIR"' EXIT
echo ">> [calibrate] fetching pilot data from $S3_URI/data/pilot ..."
mkdir -p "$PILOT_DIR"
aws s3 sync "$S3_URI/data/pilot" "$PILOT_DIR/" --only-show-errors

# Fail fast if the pilot prefix was empty/misconfigured (before spending on the sweep).
shopt -s nullglob; _csvs=("$PILOT_DIR"/*.csv); shopt -u nullglob
if [ ${#_csvs[@]} -eq 0 ] || [ ! -f "$PILOT_DIR/stimuli_manifest.json" ]; then
  echo "!! [calibrate] no *.csv and/or stimuli_manifest.json under $S3_URI/data/pilot"
  echo "!! Stage the pilot first, e.g.:"
  echo "!!   aws s3 cp data/pilot/                     \"$S3_URI/data/pilot/\" --recursive"
  echo "!!   aws s3 cp SpAM_Task/stimuli_manifest.json \"$S3_URI/data/pilot/\""
  exit 1
fi
mkdir -p calibration

N_JOBS="$N_JOBS" GT_METHOD="$GT_METHOD" REPS="$REPS" DF_LIST="${DF_LIST:-5,8,10}" python - <<'PY' 2>&1 | tee calibration/calibrate.log
import os
import numpy as np
import pandas as pd
from SpAM_Simulations.config import TaskV3SimulationConfig, MDSSweepConfig
from SpAM_Simulations import pipeline, eval_helpers
from SpAM_Simulations.pilot import calibrate_params_from_pilot, load_pilot_subjects, build_gt_from_pilot

PILOT, MANIFEST = "data/pilot", "data/pilot/stimuli_manifest.json"
GT_METHOD = os.environ.get("GT_METHOD", "smacof")
REPS = int(os.environ.get("REPS", "10"))
N_JOBS = int(os.environ["N_JOBS"])
DF_LIST = [int(x) for x in os.environ.get("DF_LIST", "5,8,10").split(",")]

# A: build the pooled-pilot GT ONCE (deterministic; identical across noise_df), reused by every fit so
# the noise_df comparison isn't confounded by GT differences.
allsub = load_pilot_subjects(PILOT, MANIFEST)
coords, gt_info = build_gt_from_pilot(allsub, method=GT_METHOD)
np.save("calibration/gt_pilot_coords.npy", coords)
print(f"[gt] {gt_info['method']} embedding: N={coords.shape[0]}, n_dims={gt_info['n_dims']}, "
      f"observed {gt_info['observed_frac']:.1%} of pairs")

# B-D: for each per-subject noise-heterogeneity df, refit (noise, dispersion) to the pilot on that fixed
# GT, then sweep the DESIGN grid (num_subjects x trials_per_subject x images_per_trial). noise_df must
# match between calibration and sweep. Plateau is read per (ndim, trials_per_subject, images_per_trial).
sweep = MDSSweepConfig(min_ndim=3, max_iters=1000, convergence_tol=1e-6, precalc_init=True)
plateaus = []
for df in DF_LIST:
    print(f"\n===== noise_df={df} =====")
    _, fit, _ = calibrate_params_from_pilot(
        PILOT, MANIFEST, gt_method=GT_METHOD, reps=REPS, noise_df=df, gt_coords=coords,
        save_params=f"calibration/calibrated_params_df{df}.json",
    )
    cfg = TaskV3SimulationConfig(
        gt_embeddings=coords,
        num_subjects=[50, 75, 150, 300],
        trials_per_subject=[20, 25], images_per_trial=[20, 25],
        subjects_noise_scale=[fit["subjects_noise_scale"]], subjects_noise_df=[df],
        frac_trials_repeated=[0.15], perspective_dispersion=[fit["perspective_dispersion"]],
        reps=5, seed=42,
    )
    outd, stored = f"out/df{df}", f"mds_store/df{df}"
    os.makedirs(outd, exist_ok=True)
    sim = pipeline.generate_task_v3_simulation(cfg, verbose=True)
    pipeline.compute_coverage_table(sim).to_csv(f"{outd}/coverage.csv", index=False)
    pipeline.compute_stability_table(sim).to_csv(f"{outd}/stability.csv", index=False)
    store = pipeline.run_mds_sweep(sim, sweep, stored, parallel=True, n_jobs=N_JOBS, verbose=True)
    es = pipeline.compute_embedding_stability(store)
    es.to_csv(f"{outd}/embedding_stability.csv", index=False)
    pl = eval_helpers.plateau_num_subjects(
        es, group_by=("ndim", "trials_per_subject", "images_per_trial"))
    pl.insert(0, "subjects_noise_df", df)
    plateaus.append(pl)
    print(f"[df={df}] {len(store)} MDS results; plateau per (ndim, design):")
    print(pl.to_string(index=False))

pd.concat(plateaus, ignore_index=True).to_csv("out/plateau_by_df.csv", index=False)
print("\n[done] combined plateau table -> out/plateau_by_df.csv")
PY

echo ">> [calibrate] uploading calibration artifacts to $S3_URI/calibration/ ..."
aws s3 sync calibration/ "$S3_URI/calibration/" --only-show-errors   # gt_pilot_coords.npy + calibrated_params.json + calibrate.log
else
# ===== default flavor: synthetic GT + swept guessed noise/dispersion =====
N_JOBS="$N_JOBS" USE_ISOTROPIC="$USE_ISOTROPIC" python - <<'PY'
import os
from SpAM_Simulations.config import TaskV3SimulationConfig, MDSSweepConfig
from SpAM_Simulations import pipeline

use_isotropic = os.environ["USE_ISOTROPIC"].strip().lower() in ("1", "true", "yes")
cfg = TaskV3SimulationConfig(
    n_images=725, n_dims=10,
    num_subjects=[30, 50, 75, 250],
    trials_per_subject=[10, 15, 20],
    images_per_trial=[20],
    subjects_noise_scale=[0.6, 1.0],   # v3: canvas placement jitter as a ratio of a trial's 2-D spread
    subjects_noise_df=[1],
    frac_trials_repeated=[0.0, 0.1, 0.15, 0.2],
    perspective_dispersion=[0.0, 0.1, 0.25],
    use_isotropic=use_isotropic,         # set per run (see header); False also uses decay/n_clusters
    decay=0.7,
    n_clusters=None,
    reps=5, seed=42,
)
print(f"ground-truth spectrum: {'isotropic (conservative bound)' if use_isotropic else 'anisotropic (realistic)'}")
sim = pipeline.generate_task_v3_simulation(cfg, verbose=True)
pipeline.compute_coverage_table(sim).to_csv("out/coverage.csv", index=False)   # test-retest cols included automatically
pipeline.compute_stability_table(sim).to_csv("out/stability.csv", index=False)

sweep = MDSSweepConfig(min_ndim=2, max_iters=500, convergence_tol=1e-6, precalc_init=False)
store = pipeline.run_mds_sweep(
    sim, sweep, "mds_store",
    parallel=True, n_jobs=int(os.environ["N_JOBS"]), verbose=True,
)
pipeline.compute_embedding_stability(store).to_csv("out/embedding_stability.csv", index=False)
print(f"done: {len(store)} MDS results")
PY
fi

upload_and_finish
