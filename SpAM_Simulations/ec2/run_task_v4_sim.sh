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
#   * trials_per_subject = 14                   the MAIN stage (12 distinct + 2 repeats); the
#                                               screening block's 8 trials are simulated separately
#                                               and pooled in, for 22 trials / 360 images per subject
#   * frac_trials_repeated = 2/14               the experimental block's 2 repeats
#   * screening_trials = 8, repeats = 2         the deployed screening block
# Swept: num_subjects x screening_min_reliability x target test-retest R.
#
# THE NOISE POPULATION'S SHAPE IS FITTED, NOT ASSUMED (step B0). Earlier runs calibrated only the
# median reliability and left the shape at |t(df=5)|, which checking against 36 real subjects showed
# is far too dispersed (CV 0.92 against roughly 0.36 empirically) - it invents a class of subjects
# with a catastrophic repeat that barely exists, and screening's entire apparent benefit came from
# truncating that invented tail. fit_noise_population fits scale AND shape jointly to the measured
# per-subject reliabilities, and the fitted shape is then held fixed while only the scale is
# re-inverted per target R, so the R axis stays a clean sensitivity on a realistic population.
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
#   * Stage the FLAT data dir under $S3_URI/data (PRIVATE - human-subjects data), containing the
#     per-session + demographics CSVs and stimuli_manifest.json, e.g.:
#       aws s3 cp data/                          "$S3_URI/data/" --recursive --exclude "*.pdf"
#       aws s3 cp SpAM_Task/stimuli_manifest.json "$S3_URI/data/"
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/task-v4   # FRESH prefix; PRIVATE
#   # export TR_LIST=0.24,0.35,0.5    # target within-subject test-retest R values
#   # export MINREL_LIST=-1,0,0.2,0.4 # screening thresholds (-1 = no exclusion control)
#   # export DISP=0.2                 # fixed perspective dispersion (empirical)
#   # export REPS=6                   # cohorts per fit point -> C(6,2)=15 cohort PAIRS
#   bash run_task_v4_sim.sh
#
# Grid: 3 R x 5 num_subjects x 4 min_reliability x 3 perspective_dispersion = 180 leaf configs
# x 6 reps x 4 ndims = ~4320 MDS fits (itmax=1000, precalc_init=True). At the ~3.2 s/fit measured on
# a c7i.4xlarge that is roughly 4 h. Dispersion is swept rather than fixed because it is identified
# only through between-subject agreement - the noisiest of the three anchors - so the fitted value
# is carried with one point either side. Screened arms simulate MORE subjects than they retain (rejected candidates are generated
# and discarded), so generation is slower than v3 by about the reciprocal of the pass rate; that is
# small next to SMACOF. REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/task-v4}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
GT_METHOD="${GT_METHOD:-smacof}"          # 'smacof' (needs R, canonical) or 'classical' (no-R smoke test)
# GT dimensionality. REQUIRED, no default: it used to be inferred from a mean-imputed eigenspectrum,
# which manufactures rank on this coverage and always returned its cap. Take it from the selection
# written by run_gt_construction.sh (gt/selection.json).
N_DIMS="${N_DIMS:?set N_DIMS, e.g. from gt/selection.json produced by run_gt_construction.sh}"
export N_DIMS
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
PILOT_DIR="data"
echo ">> [calibrate] fetching participant data from $S3_URI/data (pilot cohort is used) ..."
mkdir -p "$PILOT_DIR"
aws s3 sync "$S3_URI/data" "$PILOT_DIR/" --only-show-errors

# Fail fast if the pilot prefix was empty/misconfigured (before spending on the sweep).
shopt -s nullglob; _csvs=("$PILOT_DIR"/*.csv); shopt -u nullglob
if [ ${#_csvs[@]} -eq 0 ] || [ ! -f "$PILOT_DIR/stimuli_manifest.json" ]; then
  echo "!! [calibrate] no *.csv and/or stimuli_manifest.json under $S3_URI/data"
  echo "!! Stage the pilot first, e.g.:"
  echo "!!   aws s3 cp data/                          \"$S3_URI/data/\" --recursive --exclude \"*.pdf\""
  echo "!!   aws s3 cp SpAM_Task/stimuli_manifest.json \"$S3_URI/data/\""
  exit 1
fi
mkdir -p calibration

N_JOBS="$N_JOBS" GT_METHOD="$GT_METHOD" REPS="$REPS" DF_LIST="${DF_LIST:-5}" \
  TR_LIST="${TR_LIST:-0.24,0.35,0.5}" MINREL_LIST="${MINREL_LIST:--1,0,0.2,0.4}" \
  DISP="${DISP:-0.2}" python - <<'PY' 2>&1 | tee calibration/calibrate.log
import json
import os
import numpy as np
import pandas as pd
from SpAM_Simulations.core.config import TaskV4SimulationConfig, MDSSweepConfig
from SpAM_Simulations.core import pipeline
from SpAM_Simulations.notebooks import eval_helpers
from SpAM_Simulations.empirical.subjects import (
    load_pilot_subjects, build_gt_from_pilot, fit_noise_for_test_retest,
    subject_reliability_sample, fit_noise_population, fit_dispersion_for_agreement,
    between_subject_agreement,
)

# The parser reads a FLAT data/ dir and derives cohort from each file's deployment_mode;
# load_pilot_subjects defaults to cohorts=("pilot",) so production data never calibrates a sweep.
PILOT, MANIFEST = "data", "data/stimuli_manifest.json"
GT_METHOD = os.environ.get("GT_METHOD", "smacof")
N_JOBS = int(os.environ["N_JOBS"])
TR_LIST = [float(x) for x in os.environ["TR_LIST"].split(",")]
MINREL_LIST = [float(x) for x in os.environ["MINREL_LIST"].split(",")]
DISP = float(os.environ["DISP"])   # the value earlier runs ASSUMED; kept only as the
                                   # comparison point in the [disp] line and as the
                                   # (perspective-invariant) dispersion used while
                                   # fitting the noise shape. The swept value is DISP_LIST.
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
# GT dimensionality is no longer inferred here. build_gt_from_pilot used to default n_dims from the
# eigenspectrum of a mean-imputed aggregate; on 63.6%-unobserved data that manufactures rank and
# simply returned its cap of 15. N_DIMS is required, and comes from a run_gt_construction.sh scan.
N_DIMS = int(os.environ["N_DIMS"])
coords, gt_info = build_gt_from_pilot(allsub, n_dims=N_DIMS, method=GT_METHOD)
np.save("calibration/gt_pilot_coords.npy", coords)
print(f"[gt] {gt_info['method']}: N={coords.shape[0]}, n_dims={gt_info['n_dims']}, "
      f"observed {gt_info['observed_frac']:.1%} of pairs")

# B0: fit the noise population's SHAPE to the empirical reliability distribution.
# Matching only the median (step B) leaves the shape assumed, and the assumed |t(df)| shape is
# wrong at both tails - it invents too many catastrophic subjects and too many excellent ones.
# Screening can only truncate a distribution, so its entire yield is set by that shape. This fits
# scale AND shape jointly against the per-subject reliabilities measured from the real repeats.
emp = subject_reliability_sample(allsub)
print(f"\n[shape] empirical reliability sample: n={len(emp)} median={np.median(emp):.3f} "
      f"q10={np.quantile(emp,.1):.3f} q90={np.quantile(emp,.9):.3f}", flush=True)
fit = fit_noise_population(coords, emp, images_per_trial=IMAGES_PER_TRIAL,
                           perspective_dispersion=DISP, n_subjects=80, reps=3, verbose=True)
FB = fit["best"]
fit["grid"].to_csv("calibration/noise_shape_grid.csv", index=False)
with open("calibration/noise_shape_fit.json", "w") as fh:
    json.dump({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in FB.items()}, fh, indent=2, default=str)
print(f"[shape] best: family={FB['family']} shape={FB['shape']} scale={FB['noise_scale']} "
      f"W1={FB['distance']:.4f} CV={FB['cv']:.3f} boundary={FB['at_shape_boundary']}")
if FB["at_shape_boundary"]:
    print("[shape] WARNING: fit sits on a shape-grid boundary - the family, not the data, is the "
          "binding constraint. Widen the grid before trusting the required-N numbers.")
# The fitted shape is held FIXED across the sweep; only the scale is re-inverted per target R below,
# so the R axis stays a clean "what if reliability were higher" sensitivity on a realistic shape.
LOGN_SIGMA = float(FB["shape"]) if FB["family"] == "lognormal" else 0.0
SHAPE_DF = int(FB["shape"]) if FB["family"] == "t" else 5

# B1: re-fit perspective_dispersion against the empirical between-subject agreement, UNDER the
# newly fitted noise population. Agreement is g(noise_distribution, dispersion) - it depends on the
# whole distribution, not just its mean - so refitting the noise shape moves the agreement curve and
# invalidates any dispersion calibrated against the old one. Sequential, not joint: test-retest is
# perspective-invariant, so the noise fit above needed no reference to dispersion.
emp_agr = between_subject_agreement(np.vstack([s_.distances for s_ in allsub]),
                                    min_overlap=20)["mean_agreement"]
DISP_FIT, disp_ach = fit_dispersion_for_agreement(
    coords, emp_agr, noise_scale=float(FB["noise_scale"]), noise_df=SHAPE_DF,
    lognormal_sigma=LOGN_SIGMA, images_per_trial=IMAGES_PER_TRIAL, reps=REPS)
print(f"[disp] empirical between-subject agreement={emp_agr:.4f} -> dispersion={DISP_FIT:.2f} "
      f"(achieved {disp_ach:.4f}); previous runs assumed {DISP:.2f}")
# Dispersion is identified only through between-subject agreement, which is measured on the sparse
# pair overlap between subjects and is the noisiest of the three anchors - so it is carried as a
# SENSITIVITY AXIS rather than trusted as a point estimate.
DISP_LIST = sorted({round(max(DISP_FIT - 0.15, 0.0), 2), round(DISP_FIT, 2),
                    round(DISP_FIT + 0.15, 2)})
print(f"[disp] sweeping dispersion over {DISP_LIST}")
with open("calibration/dispersion_fit.json", "w") as fh:
    json.dump(dict(empirical_agreement=float(emp_agr), fitted=float(DISP_FIT),
                   achieved=float(disp_ach), swept=DISP_LIST,
                   noise_family=FB["family"], noise_shape=float(FB["shape"])), fh, indent=2)

# B: invert subjects_noise_scale for each target test-retest at the DEPLOYED images=20, holding the
# fitted shape fixed. Targets are nominal; the ACHIEVED R is recorded and is what the analysis
# should group by. Note the achieved R here is the UNSCREENED population value - screening raises
# the retained cohort's R, and that realised value is reported per cell in coverage.csv.
rows = []
for R in TR_LIST:
    # lognormal_sigma MUST be threaded here: the inversion has to use the SAME noise population
    # the sweep will run, or the achieved R it reports describes a different distribution and the
    # whole target-R axis is mislabelled (this bit the first task-v4-fitted run).
    noise, ach = fit_noise_for_test_retest(coords, R, noise_df=SHAPE_DF,
                                           lognormal_sigma=LOGN_SIGMA,
                                           images_per_trial=IMAGES_PER_TRIAL, reps=REPS)
    rows.append(dict(subjects_noise_df=SHAPE_DF, noise_family=FB["family"],
                     lognormal_sigma=LOGN_SIGMA, target_test_retest=R,
                     subjects_noise_scale=noise, achieved_tr_unscreened=ach))
    print(f"[invert] targetR={R:.2f} -> noise={noise:.2f} (achieved {ach:.3f} @img20)")
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
    tag = f"tr{int(round(R * 100)):02d}"
    # NB: report DISP_LIST (what is actually swept), not the DISP env default -- printing
    # the latter made the log claim dispersion=0.2 while the sweep ran the fitted 0.3.
    print(f"\n===== {tag}: noise={noise:.2f}, disp={DISP_LIST}, sigma={LOGN_SIGMA}, "
          f"N={FULL_N}, min_rel={MINREL_LIST} =====")
    cfg = TaskV4SimulationConfig(
        gt_embeddings=coords, num_subjects=FULL_N,
        trials_per_subject=[TRIALS_PER_SUBJECT], images_per_trial=[IMAGES_PER_TRIAL],
        subjects_noise_scale=[noise], subjects_noise_df=[df],
        subjects_noise_lognormal_sigma=[LOGN_SIGMA],
        frac_trials_repeated=[FRAC_REPEATED], perspective_dispersion=DISP_LIST,
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

    group_by = ("ndim", "screening_min_reliability", "perspective_dispersion")
    pl = eval_helpers.plateau_num_subjects(es, group_by=group_by)
    pl.insert(0, "target_test_retest", R); pl.insert(0, "noise_family", FB["family"])
    pl.insert(0, "noise_shape", FB["shape"])
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
