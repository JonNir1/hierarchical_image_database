#!/usr/bin/env bash
#
# STAGE 1 of sim-v6: rebuild the ground truth on the pilot PLUS the production subjects the analysis
# discards, and decide from evidence whether the rebuild is an improvement.
#
# WHY. v5's GT is fitted on 41 pre-SHINE pilot subjects and exceeds its own data's reliability
# ceiling - it interpolates where the pair graph is thin. Eight pre-SHINE production subjects are
# sitting unused (the deployed gate rejected them, or they cleared it and then failed the same rule
# on their experimental block), and adding them raises the number of pairs seen at least twice by
# ~49%. Replication is the direct cure for a ceiling problem.
#
# NOT CIRCULAR. All eight are excluded from the analysed pool by a rule fixed before this analysis,
# so no subject who will appear in the study's results contributes to the ground truth its design
# was chosen against. See the module docstring of cli/build_gt_v6.py.
#
# THE DECISION IS ONE-SIDED. These are the worst eight subjects available, and the augmented set has
# larger split-halves, which favours it for reasons unrelated to data quality. So a small gain is
# not treated as vindication; only a worsening beyond noise rejects the rebuild. On rejection the
# script still writes a pilot-only GT and says so, and gt/gt_v6_decision.json carries `accepted`.
#
# OUTPUTS (to $S3_URI/gt/)
#   gt/gt_pre_shine_v6_d8.npy      the rebuilt coordinates   <- GT_FILE for run_decision_v6.sh
#     (or gt_pre_shine_pilot_only_d8.npy, if the rebuild was rejected)
#   gt/gt_v6_comparison.csv        split-half + noise-ceiling diagnostics for BOTH subject sets
#   gt/gt_v6_decision.json         the decision, its margin, and every input count
#
# AFTER THIS RUNS, RE-RUN THE CALIBRATION GATE. The ground truth is a fingerprinted input to the
# calibration, so a rebuilt GT invalidates the cached fit BY DESIGN. The gate's current 6/6 pass was
# measured against v5's GT and does not transfer. Re-run it locally against the new file before
# launching the decision run:
#
#   python -m SpAM_Simulations.cli.run_v6_calibration_gate \
#     --gt gt/gt_pre_shine_v6_d8.npy --manifest SpAM_Task/stimuli_manifest.json --no-reuse
#
# Prerequisites (identical to the other EC2 scripts - see Cookbook.md)
#   * The commit/branch you want is PUSHED to the remote (this clones from it).
#   * S3 access via an instance role (preferred) or `aws configure`.
#   * The FLAT data dir staged under $S3_URI/data (PRIVATE - human-subjects data).
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/decision-v6   # PRIVATE
#   bash run_gt_v6.sh
#
# Cost: 2 subject sets x 30 split-half draws x 2 halves = 120 SMACOF fits at d=8, plus two final
# fits. ~10-20 min on a c7i.4xlarge. It is cheap enough to run on the same box as the decision run,
# and that is the intended flow: this first, then re-gate, then run_decision_v6.sh.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/decision-v6}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
GT_METHOD="${GT_METHOD:-smacof}"
GT_NDIM="${GT_NDIM:-8}"
N_DRAWS="${N_DRAWS:-30}"
# The expected subject counts are ASSERTED, not trusted. EXPECT_EXCLUDED especially: it grows as
# collection continues, and a GT rebuilt on a different subject set is not comparable to the one the
# decision run was calibrated against. Raising it must be a deliberate act.
EXPECT_PILOT="${EXPECT_PILOT:-41}"
EXPECT_EXCLUDED="${EXPECT_EXCLUDED:-8}"
# Each MDS worker holds its own R/smacof process, so all vCPUs OOM'd a previous run.
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# --------------------------------------------------------------------------- logging
# The log MUST live outside $WORKDIR: prepare_machine.sh does `rm -rf "$WORKDIR"` for a clean clone,
# which would unlink a log placed inside it out from under the running tee.
LOGFILE="${WORKDIR%/}_gt_v6.log"
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1
_START_TS=$(date -u +%s)
echo ">> [start] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (sim-v6 GT rebuild, S3_URI=$S3_URI)"

_on_exit() {
  local rc=$?
  # Human-subjects data never survives the run, whatever the exit path.
  [ -n "${DATA_DIR:-}" ] && rm -rf "$DATA_DIR"
  echo ">> [end] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (exit $rc, elapsed $(( $(date -u +%s) - _START_TS ))s)"
  if [ -n "${S3_URI:-}" ]; then
    sleep 1   # let the tee subprocess flush before the upload
    aws s3 cp "$LOGFILE" "$S3_URI/run_gt_v6.log" --only-show-errors || true
  fi
}
trap _on_exit EXIT

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# --------------------------------------------------------------------------- parser + task config
# Two paths outside the sparse checkout (SpAM_Simulations only) are needed: analysis/utils/parser.py
# loads the sessions, and SpAM_Task/task_config.json supplies the screening thresholds - which are
# read from the deployed config rather than from constants precisely so an audit cannot drift from
# the gate participants actually faced. `sparse-checkout add` alone can silently fail to materialise
# a tree on a --depth 1 cone clone, so verify and force a path checkout.
git sparse-checkout add analysis/utils SpAM_Task || true
for f in analysis/utils/parser.py SpAM_Task/task_config.json; do
  if [ ! -f "$f" ]; then
    echo ">> [gt-v6] sparse add didn't materialise $f; forcing a path checkout ..."
    git checkout "$GIT_REF" -- "$(dirname "$f")" 2>/dev/null \
      || git checkout HEAD -- "$(dirname "$f")" 2>/dev/null || true
  fi
  if [ ! -f "$f" ]; then
    echo "!! [gt-v6] $f still missing (checked out ref: $GIT_REF)."
    git sparse-checkout list || true
    exit 1
  fi
done
echo ">> [gt-v6] parser and task_config present"

# --------------------------------------------------------------------------- data
DATA_DIR="data"
echo ">> [gt-v6] fetching participant data from $S3_URI/data ..."
mkdir -p "$DATA_DIR"
aws s3 sync "$S3_URI/data" "$DATA_DIR/" --only-show-errors
shopt -s nullglob; _csvs=("$DATA_DIR"/*.csv); shopt -u nullglob
if [ ${#_csvs[@]} -eq 0 ] || [ ! -f "$DATA_DIR/stimuli_manifest.json" ]; then
  echo "!! [gt-v6] no *.csv and/or stimuli_manifest.json under $S3_URI/data"
  echo "!! Stage the data first, e.g.:"
  echo "!!   aws s3 cp data/                          \"$S3_URI/data/\" --recursive --exclude \"*.pdf\""
  echo "!!   aws s3 cp SpAM_Task/stimuli_manifest.json \"$S3_URI/data/\""
  exit 1
fi

# Resume-friendly: pull any existing gt/ so a re-run accumulates rather than replacing.
stage_pull gt

# --------------------------------------------------------------------------- rebuild
python -m SpAM_Simulations.cli.build_gt_v6 \
  --gt-dir gt \
  --data-dir "$DATA_DIR" \
  --manifest "$DATA_DIR/stimuli_manifest.json" \
  --config SpAM_Task/task_config.json \
  --ndim "$GT_NDIM" \
  --n-draws "$N_DRAWS" \
  --n-jobs "$N_JOBS" \
  --method "$GT_METHOD" \
  --expect-pilot "$EXPECT_PILOT" \
  --expect-excluded "$EXPECT_EXCLUDED"

stage_push gt

GT_FILE=$(python -c "import json;print(json.load(open('gt/gt_v6_decision.json'))['gt_file'])")
ACCEPTED=$(python -c "import json;print(json.load(open('gt/gt_v6_decision.json'))['accepted'])")
echo
echo ">> [gt-v6] DONE. accepted=$ACCEPTED  GT_FILE=$GT_FILE"
echo ">> Next, in order:"
echo ">>   1. Re-run the calibration gate against this GT (it invalidates the cached fit BY DESIGN):"
echo ">>        python -m SpAM_Simulations.cli.run_v6_calibration_gate \\"
echo ">>          --gt gt/$GT_FILE --manifest SpAM_Task/stimuli_manifest.json --no-reuse"
echo ">>   2. Only if it still passes 6/6:  GT_FILE=$GT_FILE bash run_decision_v6.sh"
echo ">> A trailing R warning about '/usr/lib/R/site-library ... contain no packages' is NOISE:"
echo ">> it is printed as R shuts down and only says the DEFAULT library paths are empty."
echo ">> !! TERMINATE THIS EC2 INSTANCE IF YOU ARE DONE WITH IT !!"
