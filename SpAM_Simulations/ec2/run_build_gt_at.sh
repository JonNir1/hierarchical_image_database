#!/usr/bin/env bash
#
# Build one additional ground-truth embedding at a chosen dimensionality, end to end.
#
# WHY THIS EXISTS AS A SCRIPT. Doing it by hand in an existing clone needs four environment pieces
# that a fresh SSH session does not inherit (the venv, PYTHONPATH, R_LIBS_USER, and the working
# directory), plus a re-fetch of the pilot data that every entrypoint's exit trap deliberately
# scrubs. Getting any one of them wrong costs a round trip. This script sources the same
# prepare_machine.sh the sweeps use, so the environment is correct by construction rather than by
# instruction, and its exit trap scrubs the data again on any exit.
#
# WHY YOU WOULD WANT A SECOND GT. run_gt_construction.sh chooses the dimensionality at which two
# ~20-subject halves still agree out of sample. That is a floor set by the pilot's size and coverage,
# not the intrinsic dimensionality of the perceptual space: above D~4 the fit has more freedom than
# 17%-coverage halves constrain, so the agreement curve falls from overfitting rather than from the
# higher dimensions being empty. On the pilot it selected 3 while the top-5% closest-pair Jaccard was
# still climbing at the largest candidate (D=20), and peak global agreement was only rho=0.233 -
# about 5% shared rank variance - so power was binding rather than geometry.
#
# A 3-D ground truth is EASIER to recover than the truth, so a planning simulation built on it
# understates required-N and overstates closest-pair recovery. Stage 2 therefore defaults to
# GT_NDIM=8, and this script is how that file gets made.
#
# It does NOT rewrite gt/selection.json. That records what the evidence chose and should keep saying
# 3; the deliberate departure lives in stage 2's GT_NDIM, documented where it is used. What this
# writes is gt/gt_pre_shine_d{NDIM}.npy plus a note in gt/extra_gts.json.
#
# Cost: one SMACOF fit on 41 subjects, a couple of minutes. Provisioning a fresh instance dominates,
# so prefer running it on a box that is already up from stage 1 - it is safe to re-run there, since
# prepare_machine.sh re-clones into a clean $WORKDIR either way.
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/gt-construction-v5   # where stage 1 wrote gt/
#   bash run_build_gt_at.sh 8            # or: NDIM=8 bash run_build_gt_at.sh
#
# Several dimensionalities in one go, if you want a GT-dimensionality sensitivity arm later:
#   for d in 5 8 12; do bash run_build_gt_at.sh "$d"; done

set -euo pipefail

# --------------------------------------------------------------------------- configuration
NDIM="${1:-${NDIM:-}}"
if [ -z "$NDIM" ]; then
  echo "!! usage: bash run_build_gt_at.sh <ndim>   (e.g. 8), or set NDIM=<ndim>"
  exit 1
fi
if ! [ "$NDIM" -gt 0 ] 2>/dev/null; then
  echo "!! ndim must be a positive integer, got '$NDIM'"
  exit 1
fi

REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI to the prefix where stage 1 wrote gt/}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
GT_METHOD="${GT_METHOD:-smacof}"
EXPECT_N_SUBJECTS="${EXPECT_N_SUBJECTS:-41}"
# Unused by this script (there is no MDS sweep here) but prepare_machine.sh echoes it.
N_JOBS="${N_JOBS:-1}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# --------------------------------------------------------------------------- logging
LOGFILE="${WORKDIR%/}_build_gt_d${NDIM}.log"
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1
_START_TS=$(date -u +%s)
echo ">> [start] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (build GT at ndim=$NDIM, $S3_URI)"

_on_exit() {
  local rc=$?
  # The whole reason the by-hand route kept failing: human-subjects data must not survive the run.
  [ -n "${PILOT_DIR:-}" ] && rm -rf "$PILOT_DIR"
  echo ">> [end] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (exit $rc, elapsed $(( $(date -u +%s) - _START_TS ))s)"
  if [ -n "${S3_URI:-}" ]; then
    sleep 1
    aws s3 cp "$LOGFILE" "$S3_URI/run_build_gt_d${NDIM}.log" --only-show-errors || true
  fi
}
trap _on_exit EXIT

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# --------------------------------------------------------------------------- pilot loader
# build_gt goes through analysis/utils/parser.py, which the sparse checkout (SpAM_Simulations only)
# does not include. `sparse-checkout add` alone can silently fail to materialise the tree on a
# --depth 1 cone clone, so verify and force a path checkout.
git sparse-checkout add analysis/utils || true
if [ ! -f analysis/utils/parser.py ]; then
  git checkout "$GIT_REF" -- analysis/utils 2>/dev/null \
    || git checkout HEAD -- analysis/utils 2>/dev/null || true
fi
if [ ! -f analysis/utils/parser.py ]; then
  echo "!! analysis/utils/parser.py missing (checked out ref: $GIT_REF)."
  git sparse-checkout list || true
  exit 1
fi

# --------------------------------------------------------------------------- inputs
PILOT_DIR="data"
echo ">> fetching participant data from $S3_URI/data ..."
mkdir -p "$PILOT_DIR"
aws s3 sync "$S3_URI/data" "$PILOT_DIR/" --only-show-errors
shopt -s nullglob; _csvs=("$PILOT_DIR"/*.csv); shopt -u nullglob
if [ ${#_csvs[@]} -eq 0 ] || [ ! -f "$PILOT_DIR/stimuli_manifest.json" ]; then
  echo "!! no *.csv and/or stimuli_manifest.json under $S3_URI/data"
  exit 1
fi

# Pull the existing gt/ so extra_gts.json accumulates rather than being replaced, and so a re-run
# on a fresh box does not lose the scan's outputs.
stage_pull gt

# --------------------------------------------------------------------------- build
python -m SpAM_Simulations.build_extra_gt \
  --ndim "$NDIM" \
  --pilot-dir "$PILOT_DIR" \
  --manifest "$PILOT_DIR/stimuli_manifest.json" \
  --gt-dir gt \
  --method "$GT_METHOD" \
  --expect-n-subjects "$EXPECT_N_SUBJECTS"

stage_push gt

echo
echo ">> DONE. gt/gt_pre_shine_d${NDIM}.npy is built and pushed to $S3_URI/gt/"
echo ">> Stage 2 will pick it up with GT_NDIM=${NDIM} (that is already the default for 8)."
echo ">> A trailing R warning about '/usr/lib/R/site-library ... contain no packages' is NOISE:"
echo ">> it is printed as R shuts down, and only says the DEFAULT library paths are empty."
echo ">> smacof lives in \$R_LIBS_USER and the fit above already used it. Check the [extra-gt] lines."
echo ">> !! TERMINATE THIS EC2 INSTANCE IF YOU ARE DONE WITH IT !!"
