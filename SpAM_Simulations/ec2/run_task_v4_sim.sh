#!/usr/bin/env bash
#
# Provision an EC2 instance, run the pilot-calibrated SpAM MDS sweep for the TASK-V4 simulation -
# the task-v3 generative coordinate-space model PLUS the deployed SpAM_Task v4.0 SCREENING BLOCK -
# and upload the results to S3.
#
# WHAT THIS RUN ANSWERS (four questions off one sweep):
#   1. GENERALIZABILITY - two independent cohorts of N subjects: how similar are the embedding
#      *spaces* they recover? Reported as Procrustes M^2 (lower = better) alongside the previous
#      runs' distance-vector Spearman, which measures a weaker thing (rank order only).
#   2. SCREENING VALIDATION - does excluding unreliable candidates lower required-N, and what does
#      it cost in extra recruitment? screening_min_reliability is swept, INCLUDING the deployed
#      0.0 and a no-exclusion control.
#   3. REQUIRED-N AT THE DEPLOYED DESIGN - 20 images/trial and 22 trials/subject, i.e. what the
#      live task_config.json actually collects, with N free to be re-derived.
#   4. ITEM-LEVEL RECOVERABILITY - top-k closest-pair Jaccard + per-image Procrustes residuals, for
#      the future "which items are too similar" study.
#
# UNLIKE run_task_v3_sim.sh THIS SCRIPT HAS ONE FLAVOR: pilot-calibrated. The synthetic-GT flavor
# exists in the v3 script and is not duplicated here - v4's whole purpose is a calibrated number.
#
# THE DESIGN GRID IS FIXED TO THE DEPLOYED TASK, not swept:
#   * images_per_trial = 20                     (task_config.json experimental_trials)
#   * trials_per_subject = 22                   screening 6+2 PLUS experimental 12+2: a retained
#                                               subject's screening trials ARE analysed data
#   * frac_trials_repeated = 4/22               the 2 screening + 2 experimental repeats
#   * screening_trials = 8, repeats = 2         the deployed screening block
# Swept: num_subjects x screening_min_reliability x target test-retest R (x noise df, single value).
#
# SCREENING SEMANTICS. screening_min_reliability = -1 runs the block but excludes nobody: that is
# the control arm, and it holds the number of COLLECTED trials fixed so the comparison isolates the
# effect of exclusion rather than confounding it with a shorter session. Candidates are drawn until
# num_subjects are RETAINED (the real recruit-until-N rule), so num_subjects always means "analysed
# cohort size" and n_candidates_screened records the recruitment cost.
#
# CAVEAT to carry into the write-up: the model has no num_moves or arrangement-SD component, so only
# the reliability criterion is simulated. The deployed task also screens on move-ratio and
# distance-SD fail rates. These results bound what RELIABILITY-BASED screening alone can buy.
#
# Sibling scripts: run_task_v3_sim.sh (same model without screening; its CALIBRATE=true flavor is
# the direct baseline for the no-exclusion arm), run_task_v2_4_sim.sh, run_task_v2_3_sim.sh,
# run_task_v0_1_sim.sh. All source the shared prepare_machine.sh (must be copied alongside - see
# README.md's "Running on EC2" cookbook).
#
# Target: Ubuntu 22.04/24.04 (apt + CRAN). On Amazon Linux swap the apt blocks for dnf.
#
# Prerequisites
#   * The commit/branch you want to run is PUSHED to the remote (this clones from it).
#   * The instance can reach the repo (public repo, or a PAT/deploy key in REPO_URL/ssh).
#   * S3 access: attach an IAM role with s3:PutObject on the bucket (preferred), or run
#     `aws configure` with the project IAM user's credentials before this script.
#   * Stage the pilot under $S3_URI/data/pilot (PRIVATE - human-subjects data), containing the
#     per-session + demographics CSVs and stimuli_manifest.json, e.g.:
#       aws s3 cp data/pilot/                     "$S3_URI/data/pilot/" --recursive
#       aws s3 cp SpAM_Task/stimuli_manifest.json "$S3_URI/data/pilot/"
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/task-v4   # FRESH prefix; PRIVATE
#   # export TR_LIST=0.24,0.35,0.5    # target within-subject test-retest R values
#   # export MINREL_LIST=-1,0,0.2,0.4 # screening thresholds (-1 = no exclusion control)
#   # export DF_LIST=5                # per-subject noise-heterogeneity df
#   # export DISP=0.2                 # fixed perspective dispersion (empirical)
#   # export REPS=6                   # cohorts per fit point -> C(6,2)=15 cohort PAIRS
#   bash run_task_v4_sim.sh
#
# Grid: 3 R x (5 num_subjects x 4 min_reliability) = 60 leaf configs x 6 reps x 4 ndims = ~1440 MDS
# fits (itmax=1000, precalc_init=True) - roughly 40% of the v3 multivariate sweep, so a c7i.4xlarge
# is ample. Screened arms simulate MORE subjects than they retain (rejected candidates are generated
# and discarded), so generation is slower than v3 by about the reciprocal of the pass rate; that is
# small next to SMACOF. REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/task-v4}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
GT_METHOD="${GT_METHOD:-smacof}"          # 'smacof' (needs R, canonical) or 'classical' (no-R smoke test)
REPS="${REPS:-6}"                         # cohorts per fit point; C(REPS,2) cohort pairs compared
# MDS worker processes. All vCPUs OOM'd a run (each worker holds its own R/smacof process), so
# default to 2/3 of them.
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# --------------------------------------------------------------------------- logging
# Capture EVERYTHING (this script + the sourced prepare_machine.sh + the python block) to a log file
# on the box, independent of how the caller redirects stdout - so `run.log` is never empty and survives
# an SSH drop. On ANY exit: scrub the pilot data, stamp the end time, and push the log to S3.
# NB: the log MUST live OUTSIDE $WORKDIR - prepare_machine.sh does `rm -rf "$WORKDIR"` for a clean clone,
# which would unlink a log placed inside it out from under the running tee (leaving an empty run.log).
LOGFILE="${WORKDIR%/}.log"      # e.g. ~/spam_run.log  (sibling of $WORKDIR, not wiped by the clean-clone rm)
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1
_START_TS=$(date -u +%s)
echo ">> [start] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (S3_URI=$S3_URI, task-v4 + screening)"

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

# --------------------------------------------------------------------------- pilot + calibration
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

N_JOBS="$N_JOBS" GT_METHOD="$GT_METHOD" REPS="$REPS" DF_LIST="${DF_LIST:-5}" \
  TR_LIST="${TR_LIST:-0.24,0.35,0.5}" MINREL_LIST="${MINREL_LIST:--1,0,0.2,0.4}" \
  DISP="${DISP:-0.2}" python - <<'PY' 2>&1 | tee calibration/calibrate.log
import os
import numpy as np
import pandas as pd
from SpAM_Simulations.config import TaskV4SimulationConfig, MDSSweepConfig
from SpAM_Simulations import pipeline, eval_helpers
from SpAM_Simulations.pilot import (
    load_pilot_subjects, build_gt_from_pilot, fit_noise_for_test_retest,
)

PILOT, MANIFEST = "data/pilot", "data/pilot/stimuli_manifest.json"
GT_METHOD = os.environ.get("GT_METHOD", "smacof")
N_JOBS = int(os.environ["N_JOBS"])
DF_LIST = [int(x) for x in os.environ["DF_LIST"].split(",")]
TR_LIST = [float(x) for x in os.environ["TR_LIST"].split(",")]
MINREL_LIST = [float(x) for x in os.environ["MINREL_LIST"].split(",")]
DISP = float(os.environ["DISP"])                 # perspective dispersion, FIXED (empirical 0.2)
REPS = int(os.environ.get("REPS", "6"))

# The DEPLOYED design (SpAM_Task/task_config.json v4.0), fixed rather than swept.
IMAGES_PER_TRIAL = 20
SCREEN_TRIALS, SCREEN_REPEATS = 8, 2             # screening_block: 6 distinct + 2 repeats
EXP_TRIALS, EXP_REPEATS = 14, 2                  # experimental_block: 12 distinct + 2 repeats
# `trials_per_subject` is the MAIN stage only - simulate_task_v4_experiment runs the screening
# block separately and pools a retained subject's screening trials into the aggregate. The
# effective session is therefore SCREEN_TRIALS + EXP_TRIALS = 22 trials / 360 distinct images.
TRIALS_PER_SUBJECT = EXP_TRIALS                                  # 14
FRAC_REPEATED = EXP_REPEATS / EXP_TRIALS                         # 2/14 -> 12 distinct + 2 repeats
FULL_N = [30, 50, 75, 150, 300]
NDIMS = [5, 6, 8, 10]

# A: pooled-pilot GT, built ONCE (deterministic; shared by every cell so comparisons aren't confounded).
allsub = load_pilot_subjects(PILOT, MANIFEST)
print(f"[gt] loaded {len(allsub)} completed sessions; calculating GT embeddings "
      f"({GT_METHOD}) - this may take a few minutes ...", flush=True)   # SMACOF runs silently in R
coords, gt_info = build_gt_from_pilot(allsub, method=GT_METHOD)
np.save("calibration/gt_pilot_coords.npy", coords)
print(f"[gt] {gt_info['method']}: N={coords.shape[0]}, n_dims={gt_info['n_dims']}, "
      f"observed {gt_info['observed_frac']:.1%} of pairs")

# B: invert subjects_noise_scale for each (df, target test-retest) at the DEPLOYED images=20.
# Targets are nominal; the ACHIEVED R is recorded and is what the analysis should group by. Note the
# achieved R here is the UNSCREENED population value - screening raises the retained cohort's R, and
# that realised value is reported per cell in coverage.csv's mean_test_retest.
rows = []
for df in DF_LIST:
    for R in TR_LIST:
        noise, ach = fit_noise_for_test_retest(coords, R, noise_df=df,
                                               images_per_trial=IMAGES_PER_TRIAL, reps=REPS)
        rows.append(dict(subjects_noise_df=df, target_test_retest=R,
                         subjects_noise_scale=noise, achieved_tr_unscreened=ach))
        print(f"[invert] df={df} targetR={R:.2f} -> noise={noise:.2f} (achieved {ach:.3f} @img20)")
noise_map = pd.DataFrame(rows)
noise_map.to_csv("calibration/noise_map.csv", index=False)
print("[save] noise map -> calibration/noise_map.csv")

# C: one sweep per (df, target R); num_subjects x screening_min_reliability swept INSIDE each cell so
# every threshold shares that cell's ground truth and noise level. Self-contained on THIS run's GT
# (a re-uploaded pilot changes the GT, so results across runs are not comparable - use a fresh prefix).
sweep = MDSSweepConfig(ndims=NDIMS, max_iters=1000, convergence_tol=1e-6, precalc_init=True)
plateaus = []
for _, r in noise_map.iterrows():
    df = int(r.subjects_noise_df); R = float(r.target_test_retest); noise = float(r.subjects_noise_scale)
    tag = f"df{df}_tr{int(round(R * 100)):02d}"
    print(f"\n===== {tag}: noise={noise:.2f}, disp={DISP}, N={FULL_N}, min_rel={MINREL_LIST} =====")
    cfg = TaskV4SimulationConfig(
        gt_embeddings=coords, num_subjects=FULL_N,
        trials_per_subject=[TRIALS_PER_SUBJECT], images_per_trial=[IMAGES_PER_TRIAL],
        subjects_noise_scale=[noise], subjects_noise_df=[df],
        frac_trials_repeated=[FRAC_REPEATED], perspective_dispersion=[DISP],
        screening_trials=[SCREEN_TRIALS], screening_repeats=[SCREEN_REPEATS],
        screening_min_reliability=MINREL_LIST,
        reps=REPS, seed=42,
    )
    outd, stored = f"out/{tag}", f"mds_store/{tag}"
    os.makedirs(outd, exist_ok=True)
    sim = pipeline.generate_task_v4_simulation(cfg, verbose=True)
    # coverage.csv carries the screening cost (n_candidates_screened, screening_pass_rate) and the
    # RETAINED cohort's realised reliability + noise - the trade this run exists to quantify.
    pipeline.compute_coverage_table(sim).to_csv(f"{outd}/coverage.csv", index=False)
    pipeline.compute_stability_table(sim).to_csv(f"{outd}/stability.csv", index=False)

    store = pipeline.run_mds_sweep(sim, sweep, stored, parallel=True, n_jobs=N_JOBS, verbose=True)

    # (1) distance-vector agreement, comparable with every previous run
    es = pipeline.compute_embedding_stability(store)
    es = es.assign(target_test_retest=R, achieved_tr_unscreened=float(r.achieved_tr_unscreened))
    es.to_csv(f"{outd}/embedding_stability.csv", index=False)
    # (2) configuration-space agreement: do two cohorts of N recover the same SPACE?
    eg = pipeline.compute_embedding_generalizability(store)
    eg = eg.assign(target_test_retest=R, achieved_tr_unscreened=float(r.achieved_tr_unscreened))
    eg.to_csv(f"{outd}/embedding_generalizability.csv", index=False)
    # (4) item-level: closest-pair reproducibility + per-image residuals. The per-image table is
    # n_images rows per group, so keep only the largest N (the interpretable case) to bound its size.
    pipeline.compute_topk_similar_pair_stability(store).to_csv(f"{outd}/topk_jaccard.csv", index=False)
    item = pipeline.compute_item_generalizability(store)
    item[item["num_subjects"] == max(FULL_N)].to_csv(f"{outd}/item_generalizability.csv", index=False)

    group_by = ("ndim", "screening_min_reliability")
    pl = eval_helpers.plateau_num_subjects(es, group_by=group_by)
    pl.insert(0, "target_test_retest", R); pl.insert(0, "subjects_noise_df", df)
    # the same plateau read off the Procrustes curve, which DECREASES with N - hence higher_is_better
    pl_m2 = eval_helpers.plateau_num_subjects(eg, y="mean_procrustes_m2", group_by=group_by,
                                              higher_is_better=False)
    pl = pl.merge(pl_m2.rename(columns={"plateau_num_subjects": "plateau_num_subjects_m2",
                                        "asymptote": "asymptote_m2"}),
                  on=list(group_by), how="outer", suffixes=("", "_m2"))
    plateaus.append(pl)
    print(f"[{tag}] {len(store)} MDS results")

pd.concat(plateaus, ignore_index=True).to_csv("out/plateau_by_df_tr.csv", index=False)
print("\n[done] combined plateau -> out/plateau_by_df_tr.csv; noise map -> calibration/noise_map.csv")
PY

echo ">> [calibrate] uploading calibration artifacts to $S3_URI/calibration/ ..."
aws s3 sync calibration/ "$S3_URI/calibration/" --only-show-errors   # gt_pilot_coords.npy + noise_map.csv + calibrate.log

upload_and_finish
