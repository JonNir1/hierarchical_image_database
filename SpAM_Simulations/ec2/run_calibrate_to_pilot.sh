#!/usr/bin/env bash
#
# Provision an EC2 instance and run the task-v3 PILOT CALIBRATION end to end (weighted-SMACOF ground
# truth from the pooled pilot + the noise/perspective fit from the v3.0 subjects), logging the output
# and uploading it - plus the fitted parameters and the pilot ground-truth coordinates - to S3.
#
# This is the calibration counterpart of the sweep scripts (run_task_v3_sim.sh etc.); it sources the
# same prepare_machine.sh (installs R + smacof + a minimal Python env, sparse-checks-out
# SpAM_Simulations/, and defines upload_and_finish). The HEAVY convergence sweep is a separate step -
# feed the GT + params this produces into run_task_v3_sim.sh's config (see README "Calibrating to
# pilot data").
#
# !! DATA POLICY !! The pilot CSVs are human-subjects data. They are gitignored and are NEVER in the
# repo clone, so this script pulls them (and the stimuli manifest) from S3. The raw CSVs are deleted
# from the instance at exit; the uploaded outputs (log, fitted scalar params, aggregate GT embedding)
# are pilot-DERIVED - keep the whole S3_URI prefix PRIVATE.
#
# CONVENTION - everything for a study lives under one S3_URI prefix (pilot path mirrors the local
# repo layout, data/pilot/):
#     $S3_URI/data/pilot/   <- INPUT  you stage once: the per-session CSVs + stimuli_manifest.json
#     $S3_URI/calibration/  <- OUTPUT this script writes: calibrate.log, calibrated_params.json, gt_pilot_coords.npy
#     $S3_URI/out/          <- OUTPUT the convergence sweep writes (run_task_v3_sim.sh)
#     $S3_URI/mds_store/
# So there is no separate pilot URI to manage - the pilot data is always read from $S3_URI/data/pilot/.
#
# Prerequisites
#   * The commit/branch is PUSHED to the remote (this clones from it).
#   * You have staged the pilot data under $S3_URI/data/pilot/ (a PRIVATE prefix), containing BOTH:
#       - the per-session CSVs  (e.g. data/pilot/*.csv)
#       - stimuli_manifest.json (the 725-image manifest; gitignored, so not in the repo)
#     e.g.:  aws s3 cp data/pilot/                 "$S3_URI/data/pilot/" --recursive
#            aws s3 cp SpAM_Task/stimuli_manifest.json "$S3_URI/data/pilot/"
#   * S3 access via an instance IAM role (preferred) or `aws configure`.
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/task-v3   # PRIVATE; pilot at <S3_URI>/data/pilot/, outputs at <S3_URI>/calibration/
#   bash run_calibrate_to_pilot.sh
#
# Calibration is light (one weighted MDS + a few hundred 11-subject sims); a small instance is fine.
# REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI (PRIVATE), e.g. s3://my-bucket/spam-simulations/task-v3}"
# Pilot data lives at $S3_URI/data/pilot/ by convention (mirrors the local repo layout).
PILOT_S3_URI="$S3_URI/data/pilot"
GT_METHOD="${GT_METHOD:-smacof}"   # 'smacof' (needs R, canonical) or 'classical' (no-R, provisional)
REPS="${REPS:-5}"                  # cohorts averaged per simulated calibration point
WORKDIR="${WORKDIR:-$HOME/spam_run}"
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"   # unused by calibration; kept for the prepare_machine contract
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"
# prepare_machine.sh leaves us in $WORKDIR/repo with the venv active, PYTHONPATH set, and out/ created.

# Calibration reuses analysis/pilot/parser.py (the canonical pilot loader), which prepare_machine's
# sparse checkout (SpAM_Simulations only) doesn't include - add it.
git sparse-checkout add analysis/pilot

# Human-subjects data must never be left on the box - remove it on ANY exit (success or failure).
PILOT_LOCAL="$PWD/pilot_data"
trap 'rm -rf "$PILOT_LOCAL"' EXIT

# --------------------------------------------------------------------------- fetch pilot data
echo ">> downloading pilot data + manifest from $PILOT_S3_URI ..."
mkdir -p "$PILOT_LOCAL"
aws s3 sync "$PILOT_S3_URI" "$PILOT_LOCAL/" --only-show-errors
test -f "$PILOT_LOCAL/stimuli_manifest.json" || {
  echo "!! stimuli_manifest.json not found under $PILOT_S3_URI - upload it alongside the CSVs"; exit 1; }

# --------------------------------------------------------------------------- calibrate
echo ">> running calibration (gt-method=$GT_METHOD, reps=$REPS) ..."
python -m SpAM_Simulations.calibrate_to_pilot \
  --pilot-dir "$PILOT_LOCAL" \
  --manifest "$PILOT_LOCAL/stimuli_manifest.json" \
  --gt-method "$GT_METHOD" \
  --reps "$REPS" \
  --save-gt out/gt_pilot_coords.npy \
  --save-params out/calibrated_params.json \
  2>&1 | tee out/calibrate.log

# --------------------------------------------------------------------------- upload + wrap-up
echo ">> uploading calibration outputs to $S3_URI/calibration/ ..."
aws s3 sync out/ "$S3_URI/calibration/" --only-show-errors   # calibrate.log + calibrated_params.json + gt_pilot_coords.npy

echo ">> ALL DONE. Calibration outputs at $S3_URI/calibration/"
echo ">>   calibrate.log            - full run log (targets, fitted params, sweep snippet)"
echo ">>   calibrated_params.json   - {subjects_noise_scale, perspective_dispersion, n_dims, ...}"
echo ">>   gt_pilot_coords.npy      - pilot ground-truth embedding (feed into the convergence sweep)"
echo ">> Raw pilot CSVs removed from this instance. Keep the $S3_URI prefix PRIVATE (pilot-derived)."
echo ">> !! TERMINATE THIS EC2 INSTANCE NOW to stop incurring charges !!"
