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
#   * CALIBRATE=true            - fits the simulation to the real pilot, then sweeps a MULTIVARIATE grid:
#     it fetches the pilot from $S3_URI/data/pilot and builds the ground truth ONCE by weighted-SMACOF
#     of the pooled pilot (so decay/n_clusters/use_isotropic are moot). Perspective dispersion is FIXED
#     to $DISP (default 0.2, empirical). For each per-subject noise-heterogeneity df in $DF_LIST
#     (default 3,5) x target within-subject test-retest R in $TR_LIST (default 0.24,0.35,0.5,0.65) it
#     INVERTS subjects_noise_scale to hit R at reference images=20 (test-retest is perspective-invariant
#     and only weakly image-dependent, so one noise per (df,R) spans both images), then sweeps
#     num_subjects x trials_per_subject x images_per_trial. The grid is self-contained on THIS run's GT
#     (a re-uploaded pilot changes the GT, so results across runs are not comparable - point $S3_URI at
#     a fresh prefix per pilot version). Outputs: out/df<df>_tr<RR>/ + mds_store/df<df>_tr<RR>/,
#     out/plateau_by_df_tr.csv, calibration/noise_map.csv (target vs ACHIEVED R per df - analysis groups
#     by achieved R), and run.log (full run transcript + start/end timestamps, uploaded to $S3_URI).
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
#   # export S3_URI=.../task-v3-multivar # (calibrate) use a FRESH prefix to extend, not overwrite
#   # export DF_LIST=3,5                 # (calibrate) per-subject noise-heterogeneity df values
#   # export TR_LIST=0.24,0.35,0.5,0.65  # (calibrate) target within-subject test-retest R values
#   # export DISP=0.2                    # (calibrate) fixed perspective dispersion
#   bash run_task_v3_sim.sh
#
# Default-flavor grid: 4 num_subjects x 3 trials_per_subject x 1 images_per_trial x 2 noise_scale x 1
# noise_df x 4 frac_trials_repeated x 3 perspective_dispersion = 288 combos x 5 reps x 9 target
# ndims = ~12960 MDS fits. Calibrate-flavor grid (defaults): 2 df x 4 R x 5 num_subjects x 2 trials x
# 2 images = 160 leaf configs x 5 reps x 13 ndims (min_ndim=3) = ~10400 fits (itmax=1000,
# precalc_init=True) ~10 h on c7i.4xlarge. REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/task-v3-isotropic}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
USE_ISOTROPIC="${USE_ISOTROPIC:-false}"   # ground-truth spectrum: true=isotropic bound, false=realistic
CALIBRATE="${CALIBRATE:-false}"           # true = fit to pilot data (see TWO FLAVORS above); ignores USE_ISOTROPIC
GT_METHOD="${GT_METHOD:-smacof}"          # (CALIBRATE only) 'smacof' (needs R, canonical) or 'classical' (no-R)
# (CALIBRATE only) GT dimensionality. REQUIRED under CALIBRATE, no default: it used to be inferred
# from a mean-imputed eigenspectrum, which manufactures rank on this coverage and always returned
# its cap. Take it from gt/selection.json, produced by run_gt_construction.sh.
if [ "${CALIBRATE,,}" = "true" ] || [ "$CALIBRATE" = "1" ]; then
  N_DIMS="${N_DIMS:?set N_DIMS, e.g. from gt/selection.json produced by run_gt_construction.sh}"
  export N_DIMS
fi
REPS="${REPS:-5}"                         # (CALIBRATE only) cohorts averaged per simulated fit point
# MDS worker processes. All vCPUs OOM'd a run (each worker holds its own R/smacof process), so
# default to 2/3 of them.
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# --------------------------------------------------------------------------- logging
# Capture EVERYTHING (this script + the sourced prepare_machine.sh + the python blocks) to a log file
# on the box, independent of how the caller redirects stdout - so `run.log` is never empty and survives
# an SSH drop. On ANY exit: scrub the pilot data, stamp the end time, and push the log to S3.
# NB: the log MUST live OUTSIDE $WORKDIR - prepare_machine.sh does `rm -rf "$WORKDIR"` for a clean clone,
# which would unlink a log placed inside it out from under the running tee (leaving an empty run.log).
LOGFILE="${WORKDIR%/}.log"      # e.g. ~/spam_run.log  (sibling of $WORKDIR, not wiped by the clean-clone rm)
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1
_START_TS=$(date -u +%s)
echo ">> [start] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (S3_URI=$S3_URI, CALIBRATE=$CALIBRATE)"

_on_exit() {
  local rc=$?
  [ -n "${PILOT_DIR:-}" ] && rm -rf "$PILOT_DIR"   # human-subjects data never left on the box
  echo ">> [end] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (exit $rc, elapsed $(( $(date -u +%s) - _START_TS ))s)"
  if [ -n "${S3_URI:-}" ]; then
    sleep 1   # let the tee subprocess flush the final lines before we upload
    aws s3 cp "$LOGFILE" "$S3_URI/run.log" --only-show-errors || true
  fi
}
trap _on_exit EXIT

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# --------------------------------------------------------------------------- run the sweep
if [ "${CALIBRATE,,}" = "true" ] || [ "$CALIBRATE" = "1" ]; then
# ===== calibrated flavor: fetch pilot -> fit noise/perspective + pilot GT -> calibrated sweep =====
# Calibration reuses analysis/utils/parser.py (the canonical pilot loader), which prepare_machine's
# sparse checkout (SpAM_Simulations only) doesn't include - add it. `sparse-checkout add` alone can
# silently fail to materialise the tree on a --depth 1 cone clone, so verify and force a path checkout.
git sparse-checkout add analysis/utils || true
if [ ! -f analysis/utils/parser.py ]; then
  echo ">> [calibrate] sparse add didn't materialise analysis/utils; forcing a path checkout ..."
  git checkout "$GIT_REF" -- analysis/utils 2>/dev/null \
    || git checkout HEAD -- analysis/utils 2>/dev/null || true
fi
if [ ! -f analysis/utils/parser.py ]; then
  echo "!! [calibrate] analysis/utils/parser.py still missing (checked out ref: $GIT_REF)."
  echo "!! sparse-checkout list:"; git sparse-checkout list || true
  echo "!! tracked analysis parser paths:"; git ls-tree -r --name-only HEAD | grep -i "analysis/.*parser.py" || true
  exit 1
fi
echo ">> [calibrate] pilot parser present: analysis/utils/parser.py"

# Fetch the (gitignored, human-subjects) pilot data; the EXIT trap (_on_exit) deletes it on any exit.
PILOT_DIR="data/pilot"
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

N_JOBS="$N_JOBS" GT_METHOD="$GT_METHOD" REPS="$REPS" DF_LIST="${DF_LIST:-3,5}" \
  TR_LIST="${TR_LIST:-0.24,0.35,0.5,0.65}" DISP="${DISP:-0.2}" python - <<'PY' 2>&1 | tee calibration/calibrate.log
import os
import numpy as np
import pandas as pd
from SpAM_Simulations.core.config import TaskV3SimulationConfig, MDSSweepConfig
from SpAM_Simulations.core import pipeline
from SpAM_Simulations.notebooks import eval_helpers
from SpAM_Simulations.empirical.pilot import (
    load_pilot_subjects, build_gt_from_pilot, fit_noise_for_test_retest, _simulated_targets,
)

PILOT, MANIFEST = "data/pilot", "data/pilot/stimuli_manifest.json"
GT_METHOD = os.environ.get("GT_METHOD", "smacof")
N_JOBS = int(os.environ["N_JOBS"])
DF_LIST = [int(x) for x in os.environ["DF_LIST"].split(",")]
TR_LIST = [float(x) for x in os.environ["TR_LIST"].split(",")]
DISP = float(os.environ["DISP"])                 # perspective dispersion, FIXED (empirical 0.2)
INV_REPS = int(os.environ.get("REPS", "8"))      # cohorts averaged in the noise->test-retest inversion
FULL_N = [30, 50, 75, 150, 300]

# A: pooled-pilot GT, built ONCE (deterministic; shared by every cell so comparisons aren't confounded).
allsub = load_pilot_subjects(PILOT, MANIFEST)
print(f"[gt] loaded {len(allsub)} completed sessions; calculating GT embeddings "
      f"({GT_METHOD}) - this may take a few minutes ...", flush=True)   # SMACOF runs silently in R
# GT dimensionality is no longer inferred here. build_gt_from_pilot used to default n_dims from the
# eigenspectrum of a mean-imputed aggregate; on 63.6%-unobserved data that manufactures rank and
# simply returned its cap of 15. N_DIMS is required, and comes from a run_gt_construction.sh scan.
N_DIMS = int(os.environ["N_DIMS"])
coords, gt_info = build_gt_from_pilot(allsub, n_dims=N_DIMS, method=GT_METHOD)
np.save("calibration/gt_pilot_coords.npy", coords)
print(f"[gt] {gt_info['method']}: N={coords.shape[0]}, n_dims={gt_info['n_dims']}, "
      f"observed {gt_info['observed_frac']:.1%} of pairs")

# B: invert subjects_noise_scale for each (df, target test-retest) at REFERENCE images=20. test-retest
# is perspective-invariant (dispersion irrelevant here) and depends only weakly on images (~0.02), so
# one noise per (df, R) is applied across both images - matching the previous run's methodology. Targets
# are nominal; the achieved R (recorded) is what analysis should group by.
rows = []
for df in DF_LIST:
    for R in TR_LIST:
        noise, ach20 = fit_noise_for_test_retest(coords, R, noise_df=df, images_per_trial=20, reps=INV_REPS)
        ach25 = _simulated_targets(coords, noise, 0.0, 20, 20, 25, 0.15, INV_REPS, 0, 25, noise_df=df)[0]
        rows.append(dict(subjects_noise_df=df, target_test_retest=R, subjects_noise_scale=noise,
                         achieved_tr_img20=ach20, achieved_tr_img25=ach25))
        print(f"[invert] df={df} targetR={R:.2f} -> noise={noise:.2f} "
              f"(achieved {ach20:.3f}@img20, {ach25:.3f}@img25)")
noise_map = pd.DataFrame(rows)
noise_map.to_csv("calibration/noise_map.csv", index=False)
print("[save] noise map -> calibration/noise_map.csv")

# C: design sweep per (df, target R), dispersion FIXED = DISP, noise = the inverted value (across both
# images). Self-contained on THIS run's GT: every cell runs the full num_subjects grid (no skipping /
# merging with a prior run - a re-uploaded pilot changes the GT, so old results are not comparable).
# num_subjects x trials x images swept; itmax=1000, precalc_init=True (deterministic init).
sweep = MDSSweepConfig(min_ndim=3, max_iters=1000, convergence_tol=1e-6, precalc_init=True)
plateaus = []
for _, r in noise_map.iterrows():
    df = int(r.subjects_noise_df); R = float(r.target_test_retest); noise = float(r.subjects_noise_scale)
    Ns = FULL_N
    tag = f"df{df}_tr{int(round(R * 100)):02d}"
    print(f"\n===== {tag}: noise={noise:.2f}, disp={DISP}, num_subjects={Ns} =====")
    cfg = TaskV3SimulationConfig(
        gt_embeddings=coords, num_subjects=Ns,
        trials_per_subject=[20, 25], images_per_trial=[20, 25],
        subjects_noise_scale=[noise], subjects_noise_df=[df],
        frac_trials_repeated=[0.15], perspective_dispersion=[DISP],
        reps=5, seed=42,
    )
    outd, stored = f"out/{tag}", f"mds_store/{tag}"
    os.makedirs(outd, exist_ok=True)
    sim = pipeline.generate_task_v3_simulation(cfg, verbose=True)
    pipeline.compute_coverage_table(sim).to_csv(f"{outd}/coverage.csv", index=False)
    pipeline.compute_stability_table(sim).to_csv(f"{outd}/stability.csv", index=False)
    store = pipeline.run_mds_sweep(sim, sweep, stored, parallel=True, n_jobs=N_JOBS, verbose=True)
    es = pipeline.compute_embedding_stability(store)
    es = es.assign(target_test_retest=R, achieved_test_retest=float(r.achieved_tr_img20))
    es.to_csv(f"{outd}/embedding_stability.csv", index=False)
    pl = eval_helpers.plateau_num_subjects(es, group_by=("ndim", "trials_per_subject", "images_per_trial"))
    pl.insert(0, "target_test_retest", R); pl.insert(0, "subjects_noise_df", df)
    plateaus.append(pl)
    print(f"[{tag}] {len(store)} MDS results")

pd.concat(plateaus, ignore_index=True).to_csv("out/plateau_by_df_tr.csv", index=False)
print("\n[done] combined plateau -> out/plateau_by_df_tr.csv; noise map -> calibration/noise_map.csv")
PY

echo ">> [calibrate] uploading calibration artifacts to $S3_URI/calibration/ ..."
aws s3 sync calibration/ "$S3_URI/calibration/" --only-show-errors   # gt_pilot_coords.npy + calibrated_params.json + calibrate.log
else
# ===== default flavor: synthetic GT + swept guessed noise/dispersion =====
N_JOBS="$N_JOBS" USE_ISOTROPIC="$USE_ISOTROPIC" python - <<'PY'
import os
from SpAM_Simulations.core.config import TaskV3SimulationConfig, MDSSweepConfig
from SpAM_Simulations.core import pipeline

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
