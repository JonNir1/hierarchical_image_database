#!/usr/bin/env bash
#
# STAGE 1 of the simulation programme: build the ground-truth embedding the later sweeps generate
# subjects from, and choose its dimensionality from evidence rather than from a rule of thumb.
#
# WHY THIS STAGE EXISTS. Every earlier run inferred the GT dimensionality by reading a classical-MDS
# eigenspectrum off the MEAN-IMPUTED aggregate RDM and keeping the smallest number of dimensions
# explaining 90% of variance, capped at 15. That is invalid on this data. 63.6% of pairs are
# unobserved and were filled with a single constant, which asserts those point pairs are all
# equidistant; k mutually equidistant points form a regular simplex needing k-1 dimensions, so the
# fill MANUFACTURES rank instead of merely adding noise. Measured: a synthetic rank-8 space put
# through the identical mask and fill reports effective rank 193 and needs 239 dimensions for 90%
# variance - statistically indistinguishable from the real data's 213 and 216. The rule therefore
# always returned its cap and the resulting GT was near-isotropic, which is very likely why the
# recovered structure in previous runs looked so weak.
#
# WHAT REPLACES IT. Dimensionality is treated as a generalisation question, and NO imputation is used
# anywhere - weighted SMACOF treats weight 0 as missing, so only observed pairs enter any fit:
#   * SPLIT-HALF (primary):  fit each candidate ndim on two disjoint halves of the subjects and score
#                            how well the halves agree (Spearman over reconstructed distances,
#                            Procrustes M^2 over configurations, top-5% closest-pair Jaccard).
#   * LEAVE-k-OUT CV (check): fit on all but k subjects, then predict the held-out subjects' OWN
#                            observed distances. A different question - generalisation to unseen
#                            PEOPLE - so the two curves agreeing is real corroboration and their
#                            disagreeing is a finding that must stop the programme rather than be
#                            averaged away.
# Selection uses the ONE-SE RULE (Breiman; glmnet's lambda.1se): the smallest ndim whose mean is
# within one standard error of the best mean. Committed to BEFORE seeing the curve, because the curve
# is expected to be nearly flat and a plain argmax on a flat noisy curve drifts to high ndim.
#
# THE SUBJECT SET IS PRE-SHINE ONLY. Half the pilot saw the post-SHINE images, which are a different
# stimulus set; pooling both would build a GT for images that do not exist. `variants=("pre",)` on
# top of the default `cohorts=("pilot",)` leaves 41 subjects, and the script FAILS FAST if that count
# has changed, because a silently different subject set invalidates every downstream comparison.
#
# CONNECTIVITY IS THE BINDING CONSTRAINT. run_mds refuses a disconnected pair graph, and at this
# coverage a random half of 41 subjects is connected only ~60% of the time. Disconnected splits are
# discarded and redrawn, which COULD be a biased filter: if a half is disconnected precisely when it
# holds poorly-covered subjects, kept draws over-represent well-covered ones and the agreement curve
# is optimistic.
#
# READ THE GAP, NOT THE RATE. gt/discard_rates.json reports both. The rate alone says only that the
# pool is sparse; it is `coverage_gap_frac` - kept minus discarded binding-half coverage, relative to
# the kept level - that says the filter is SELECTIVE. Measured on this pilot they come apart sharply:
# ~40% discarded with a gap of +0.4%, i.e. sparse but not measurably biased. That is expected from the
# pilot's composition rather than the design's: 25 of the 41 pre-SHINE subjects ran task v1/v2 with
# 10 trials each (1,873 observed pairs) against v3's 16 subjects at 3,230, so halves drawn heavy on
# v1/v2 are disproportionately disconnected without being systematically lower-coverage. The deployed
# v4 session collects 18 distinct trials, so this is a pilot artifact, not a forecast for the study.
# Above ~5% relative gap the split-half curve is genuinely optimistic and leave-k-out should be used
# alone.
#
# OUTPUTS (to $S3_URI/gt/, pushed after EVERY dimensionality so a died-at-hour-7 run keeps its work):
#   gt/scan.csv              one row per (ndim, draw): all three split-half scores + solver status
#   gt/cv.csv                one row per (ndim, fold):  held-out Spearman + solver status
#   gt/discard_rates.json    the split-search diagnostics above
#   gt/splits.npz            the exact half-splits used, so a resumed run scores the SAME draws
#   gt/selection.json        the chosen ndim and the evidence for it  <- supplies N_DIMS downstream
#   gt/gt_pre_shine_d{K}.npy the final (n_images, K) coordinates fitted on all 41 subjects
#
# RESUMABLE. `stage_pull gt` at the top, then any ndim already present in scan.csv/cv.csv is skipped.
# Re-running after a crash costs only the dimensionalities that had not finished.
#
# Prerequisites (identical to the other EC2 scripts - see Cookbook.md)
#   * The commit/branch you want is PUSHED to the remote (this clones from it).
#   * S3 access via an instance role (preferred) or `aws configure`.
#   * The FLAT data dir staged under $S3_URI/data (PRIVATE - human-subjects data):
#       aws s3 cp data/                          "$S3_URI/data/" --recursive --exclude "*.pdf"
#       aws s3 cp SpAM_Task/stimuli_manifest.json "$S3_URI/data/"
#
# Usage:
#   export REPO_URL=https://github.com/<you>/hierarchical_image_database.git
#   export GIT_REF=main
#   export S3_URI=s3://<your-bucket>/spam-simulations/gt-construction   # PRIVATE
#   # export NDIMS=2,3,4,5,6,7,8,10,12,15,20   # candidate dimensionalities
#   # export N_DRAWS=50                        # split-half draws (shared across every ndim)
#   # export CV_K=5 CV_FOLDS=40                # leave-k-out settings
#   bash run_gt_construction.sh
#
# Cost: 11 ndims x 50 draws x 2 halves = 1100 split-half fits, plus 11 x 40 = 440 CV fits.
# At the 2.6 s/fit throughput measured on a c7i.4xlarge with N_JOBS=10 that is ~1.1 h if cost is
# dimensionality-independent and ~4.7 h if it scales linearly with ndim; BUDGET the upper end. A
# ~20-subject half sits at ~26% coverage and hits max_iters far more often than the dense
# 41-subject aggregate, and a max-iters fit pays the full 1000 iterations. The status/niter columns
# are in scan.csv precisely so the first pushed partial can be checked and the run killed early if
# the max-iters rate is high. REMEMBER TO TERMINATE THE INSTANCE WHEN DONE.

set -euo pipefail

# --------------------------------------------------------------------------- configuration
REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/gt-construction}"
WORKDIR="${WORKDIR:-$HOME/spam_run}"
GT_METHOD="${GT_METHOD:-smacof}"          # 'smacof' (needs R, canonical) or 'classical' (smoke test)
# MDS worker processes. All vCPUs OOM'd a previous run (each worker holds its own R/smacof process),
# so default to 2/3 of them.
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# --------------------------------------------------------------------------- logging
# Capture EVERYTHING (this script + the sourced prepare_machine.sh + the python block) to a log file
# on the box, independent of how the caller redirects stdout - so `run.log` is never empty and
# survives an SSH drop. On ANY exit: scrub the pilot data, stamp the end time, push the log to S3.
# NB: the log MUST live OUTSIDE $WORKDIR - prepare_machine.sh does `rm -rf "$WORKDIR"` for a clean
# clone, which would unlink a log placed inside it out from under the running tee.
LOGFILE="${WORKDIR%/}.log"
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1
_START_TS=$(date -u +%s)
echo ">> [start] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (S3_URI=$S3_URI, GT construction)"

_on_exit() {
  local rc=$?
  [ -n "${PILOT_DIR:-}" ] && rm -rf "$PILOT_DIR"   # human-subjects data never left on the box
  echo ">> [end] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (exit $rc, elapsed $(( $(date -u +%s) - _START_TS ))s)"
  if [ -n "${S3_URI:-}" ]; then
    sleep 1   # let the tee subprocess flush the final lines before we upload
    aws s3 cp "$LOGFILE" "$S3_URI/run_gt_construction.log" --only-show-errors || true
  fi
}
trap _on_exit EXIT

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# --------------------------------------------------------------------------- pilot loader
# The GT is built from real subjects via analysis/utils/parser.py (the canonical pilot loader), which
# prepare_machine's sparse checkout (SpAM_Simulations only) doesn't include - add it. `sparse-checkout
# add` alone can silently fail to materialise the tree on a --depth 1 cone clone, so verify and force
# a path checkout.
git sparse-checkout add analysis/utils || true
if [ ! -f analysis/utils/parser.py ]; then
  echo ">> [gt] sparse add didn't materialise analysis/utils; forcing a path checkout ..."
  git checkout "$GIT_REF" -- analysis/utils 2>/dev/null \
    || git checkout HEAD -- analysis/utils 2>/dev/null || true
fi
if [ ! -f analysis/utils/parser.py ]; then
  echo "!! [gt] analysis/utils/parser.py still missing (checked out ref: $GIT_REF)."
  echo "!! sparse-checkout list:"; git sparse-checkout list || true
  echo "!! tracked analysis parser paths:"; git ls-tree -r --name-only HEAD | grep -i "analysis/.*parser.py" || true
  exit 1
fi
echo ">> [gt] pilot parser present: analysis/utils/parser.py"

# Fetch the (gitignored, human-subjects) pilot data; the EXIT trap deletes it on any exit.
PILOT_DIR="data"
echo ">> [gt] fetching participant data from $S3_URI/data ..."
mkdir -p "$PILOT_DIR"
aws s3 sync "$S3_URI/data" "$PILOT_DIR/" --only-show-errors

shopt -s nullglob; _csvs=("$PILOT_DIR"/*.csv); shopt -u nullglob
if [ ${#_csvs[@]} -eq 0 ] || [ ! -f "$PILOT_DIR/stimuli_manifest.json" ]; then
  echo "!! [gt] no *.csv and/or stimuli_manifest.json under $S3_URI/data"
  echo "!! Stage the pilot first, e.g.:"
  echo "!!   aws s3 cp data/                          \"$S3_URI/data/\" --recursive --exclude \"*.pdf\""
  echo "!!   aws s3 cp SpAM_Task/stimuli_manifest.json \"$S3_URI/data/\""
  exit 1
fi

# Resume: anything already computed comes back, and finished dimensionalities are skipped below.
stage_pull gt

# --------------------------------------------------------------------------- the scan
N_JOBS="$N_JOBS" GT_METHOD="$GT_METHOD" S3_URI="$S3_URI" \
  NDIMS="${NDIMS:-2,3,4,5,6,7,8,10,12,15,20}" N_DRAWS="${N_DRAWS:-50}" \
  CV_K="${CV_K:-5}" CV_FOLDS="${CV_FOLDS:-40}" SEED="${SEED:-0}" \
  EXPECT_N_SUBJECTS="${EXPECT_N_SUBJECTS:-41}" \
  python - <<'PY'
import json
import os
import pathlib
import subprocess

import numpy as np
import pandas as pd

from SpAM_Simulations import gt_construction as gtc
from SpAM_Simulations.pilot import load_pilot_subjects

GT = pathlib.Path("gt")
GT.mkdir(exist_ok=True)
PILOT, MANIFEST = "data", "data/stimuli_manifest.json"
METHOD = os.environ.get("GT_METHOD", "smacof")
N_JOBS = int(os.environ["N_JOBS"])
NDIMS = [int(x) for x in os.environ["NDIMS"].split(",")]
N_DRAWS, SEED = int(os.environ["N_DRAWS"]), int(os.environ["SEED"])
CV_K, CV_FOLDS = int(os.environ["CV_K"]), int(os.environ["CV_FOLDS"])
EXPECT_N = int(os.environ["EXPECT_N_SUBJECTS"])


def push(msg: str) -> None:
    """Publish gt/ so far. Called after every dimensionality; a partial run keeps its work.

    Non-fatal by design: a transient S3 error must not throw away the hours of fits that are
    already on local disk, and the next push (or the final stage_push) will carry them.
    """
    print(f"[push] {msg}", flush=True)
    rc = subprocess.run(["aws", "s3", "sync", "gt/", f"{os.environ['S3_URI']}/gt/",
                         "--only-show-errors"], check=False).returncode
    if rc != 0:
        print(f"[push] WARNING: sync exited {rc}; results are still on local disk", flush=True)


# --- subjects -------------------------------------------------------------------------------
# cohorts=("pilot",) is the default and excludes production data. variants=("pre",) additionally
# drops the post-SHINE half: those subjects judged a DIFFERENT stimulus set, so pooling them would
# fit a ground truth for images that do not exist.
subjects = load_pilot_subjects(PILOT, MANIFEST, variants=("pre",))
print(f"[gt] loaded {len(subjects)} pre-SHINE pilot subjects "
      f"(expected {EXPECT_N}); coverage {gtc.coverage_of(subjects):.1%} of pairs", flush=True)
if len(subjects) != EXPECT_N:
    raise SystemExit(
        f"expected {EXPECT_N} pre-SHINE pilot subjects, got {len(subjects)}. The subject set "
        f"defines the ground truth, so a silent change here invalidates every downstream "
        f"comparison. If the pilot really did change, set EXPECT_N_SUBJECTS deliberately."
    )
if not gtc.is_connected(subjects):
    raise SystemExit("the pooled pre-SHINE pair graph is disconnected; MDS cannot be fitted on it")

# --- splits ---------------------------------------------------------------------------------
# Drawn ONCE and persisted, for two reasons: every ndim must be scored on the SAME draws or the
# comparison is unpaired (and the curve is expected to be flat enough that between-draw variance
# would swamp between-ndim differences), and a resumed run must continue the same experiment.
splits_path = GT / "splits.npz"
if splits_path.exists():
    z = np.load(splits_path)
    splits = [(z[f"a{i}"], z[f"b{i}"]) for i in range(int(z["n"]))]
    diagnostics = json.loads((GT / "discard_rates.json").read_text())
    print(f"[gt] reusing {len(splits)} persisted half-splits", flush=True)
else:
    splits, diagnostics = gtc.draw_valid_splits(subjects, N_DRAWS, np.random.default_rng(SEED))
    np.savez(splits_path, n=len(splits),
             **{f"a{i}": a for i, (a, _) in enumerate(splits)},
             **{f"b{i}": b for i, (_, b) in enumerate(splits)})
    (GT / "discard_rates.json").write_text(json.dumps(diagnostics, indent=2))
print(f"[splits] half_size={diagnostics['half_size']} discard_rate={diagnostics['discard_rate']:.1%} "
      f"binding coverage kept={diagnostics['mean_binding_coverage_kept']:.4f} "
      f"discarded={diagnostics['mean_binding_coverage_discarded']:.4f} "
      f"gap={diagnostics['coverage_gap']:+.4f} ({diagnostics['coverage_gap_frac']:+.1%})", flush=True)
# The GAP is the diagnostic, not the rate. A high discard rate only says the subject pool is sparse;
# it is a gap between kept and discarded draws that says the filter is SELECTIVE, i.e. that the
# retained splits over-represent well-covered subjects and the agreement curve is optimistic.
# On the pilot these come apart sharply - ~40% discarded with a 0.4% relative gap - because 25 of the
# 41 pre-SHINE subjects ran v1/v2 at 10 trials each, so halves drawn heavy on them are
# disproportionately disconnected without being systematically lower-coverage.
if diagnostics["coverage_gap_frac"] > 0.05:
    print("[splits] WARNING: kept splits are >5% better covered than discarded ones, so the filter "
          "IS selective and the split-half curve is optimistic. Prefer the leave-k-out curve.",
          flush=True)
elif diagnostics["discard_rate"] > 0.30:
    print(f"[splits] NOTE: {diagnostics['discard_rate']:.0%} of draws were discarded, but the "
          f"coverage gap is only {diagnostics['coverage_gap_frac']:+.1%}, so the filter is not "
          f"measurably selective - the pool is sparse rather than the estimate biased. Expected on "
          f"this pilot: 25 of 41 subjects ran v1/v2 with 10 trials against v3's 18.", flush=True)

folds = gtc.leave_k_out_folds(len(subjects), k=CV_K, n_folds=CV_FOLDS,
                              rng=np.random.default_rng(SEED))
aggregates = gtc.split_aggregates(subjects, splits)


def _resume(path, done_col="ndim"):
    """Existing rows for this artifact, plus the set of dimensionalities already finished."""
    if not path.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(path)
    return df, set(df[done_col].astype(int))


scan_df, scan_done = _resume(GT / "scan.csv")
cv_df, cv_done = _resume(GT / "cv.csv")
if scan_done or cv_done:
    print(f"[resume] scan has {sorted(scan_done)}, cv has {sorted(cv_done)}", flush=True)

# --- per-dimensionality loop ----------------------------------------------------------------
for ndim in NDIMS:
    if ndim in scan_done and ndim in cv_done:
        print(f"\n===== ndim={ndim}: already complete, skipping =====", flush=True)
        continue
    print(f"\n===== ndim={ndim} =====", flush=True)
    if ndim not in scan_done:
        rows = gtc.scan_ndim_parallel(aggregates, ndim, n_jobs=N_JOBS, verbose=True)
        scan_df = pd.concat([scan_df, rows], ignore_index=True)
        scan_df.to_csv(GT / "scan.csv", index=False)
        stat = rows["status_a"].tolist() + rows["status_b"].tolist()
        print(f"[scan] ndim={ndim} spearman={rows.spearman.mean():.3f} "
              f"m2={rows.procrustes_m2.mean():.3f} jaccard={rows.topk_jaccard.mean():.3f} "
              f"max_iters={stat.count('max_iters') / len(stat):.0%} "
              f"failed={sum(s not in ('success', 'max_iters') for s in stat)}", flush=True)
    if ndim not in cv_done:
        rows = gtc.cross_validate_ndim_parallel(subjects, ndim, folds, n_jobs=N_JOBS, verbose=True)
        cv_df = pd.concat([cv_df, rows], ignore_index=True)
        cv_df.to_csv(GT / "cv.csv", index=False)
        print(f"[cv] ndim={ndim} held-out spearman={rows.spearman.mean():.3f}", flush=True)
    push(f"ndim={ndim} done")

# --- selection ------------------------------------------------------------------------------
# Reported on BOTH resolutions, selected on the GLOBAL one. Local (top-5% Jaccard) is the
# decision-relevant quantity downstream, but it is far noisier per draw, and a dimensionality that
# reproduces the whole geometry is the safer basis for generating simulated subjects.
scan_summary = gtc.summarise_scan(scan_df)
cv_summary = gtc.summarise_scan(cv_df, metrics=["spearman"])
print("\n[scan summary]\n" + scan_summary.to_string(index=False), flush=True)
print("\n[cv summary]\n" + cv_summary.to_string(index=False), flush=True)

choices = {
    "split_half_spearman": gtc.select_ndim(scan_df, "spearman", rule="one_se"),
    "split_half_procrustes_m2": gtc.select_ndim(scan_df, "procrustes_m2", rule="one_se"),
    "split_half_topk_jaccard": gtc.select_ndim(scan_df, "topk_jaccard", rule="one_se"),
    "cv_spearman": gtc.select_ndim(cv_df, "spearman", rule="one_se"),
    "split_half_spearman_argmax": gtc.select_ndim(scan_df, "spearman", rule="argmax"),
}
N_DIMS = choices["split_half_spearman"]
print(f"\n[select] one-SE choices: {choices}")
print(f"[select] SELECTED n_dims={N_DIMS} (split-half Spearman, one-SE rule)")
if choices["cv_spearman"] != N_DIMS:
    print(f"[select] NOTE: leave-k-out CV prefers {choices['cv_spearman']}. The two curves ask "
          f"different questions, so a small gap is expected; a large one means the scan is not "
          f"identifying a dimensionality and the downstream sweeps should not be trusted.")

# --- the final embedding --------------------------------------------------------------------
coords, info = gtc.build_gt(subjects, N_DIMS, method=METHOD)
gt_file = f"gt_pre_shine_d{N_DIMS}.npy"
np.save(GT / gt_file, coords)
scan_summary.to_csv(GT / "scan_summary.csv", index=False)
cv_summary.to_csv(GT / "cv_summary.csv", index=False)
(GT / "selection.json").write_text(json.dumps({
    "n_dims": int(N_DIMS), "gt_file": gt_file, "rule": "one_se",
    "criterion": "split_half_spearman", "choices": {k: int(v) for k, v in choices.items()},
    "candidate_ndims": NDIMS, "n_draws": len(splits), "cv_k": CV_K, "cv_folds": CV_FOLDS,
    "split_diagnostics": diagnostics, "gt_info": info,
}, indent=2, default=str))
print(f"\n[done] n_dims={N_DIMS}, coords {coords.shape} -> gt/{gt_file}")
print(f"[done] pass N_DIMS={N_DIMS} to run_design_comparison.sh (or read gt/selection.json)")
PY

stage_push gt
echo ">> [gt] stage 1 complete. gt/selection.json now supplies N_DIMS to stage 2."
echo ">> !! TERMINATE THIS EC2 INSTANCE NOW to stop incurring charges !!"
