#!/usr/bin/env bash
# =============================================================================================
# sim-v6: a DECISION run, not a survey.
#
# v5 swept 17,280 fits over a grid and averaged over it. This run answers one question: which of
# three ways to finish the study is best.
#
#     A   N=50 per cohort, screening rho > 0     (what is deployed now)
#     B   N=50 per cohort, screening rho > 0.1
#     C   N=75 per cohort, screening rho > 0
#
# All on the random allocation arm, because that is what is deployed and the decision not to switch
# has already been taken.
#
# FIVE DIFFERENCES FROM v5, and nothing else changes:
#
#   1. Participants are recalibrated on the 84 PRODUCTION candidates rather than the 41 pilot
#      subjects. v5's noise scale (0.30) is ~25% too high for the deployed cohort, which is why it
#      predicts a 71% screening pass rate against the 93% actually observed.
#   2. The ground truth is rebuilt from the pilot PLUS the 8 pre-SHINE production subjects who are
#      excluded from the analysed pool anyway (see run_gt_v6.sh). Not circular: no analysed subject
#      contributes to it.
#   3. Retained cohorts EXCLUDE false alarms, matching the analysis we would actually run.
#   4. RQ2 power is measured, by simulating cohorts on deliberately perturbed ground truths.
#   5. Participants may DRIFT within a session: DRIFT scales their noise for the experimental block.
#      Production shows the degradation qualitatively (every move-ratio failure falls in that block,
#      none in the screening block) and v5 had no mechanism for it.
#
# THE GATE. `cli.run_v6_calibration_gate` must pass locally BEFORE this is launched. It checks the
# recalibrated model against six quantities production already shows. A model that cannot reproduce
# what we can measure is not worth an instance for what we cannot.
#
#   Gate state: PASSES 6/6, at noise_scale=0.22, lognormal sigma=0.25, dispersion=0.10.
#
# ON DRIFT, AND WHY IT IS SWEPT RATHER THAN CHOSEN. The gate passes 6/6 at BOTH drift=1.0 (none) and
# drift=1.1 (the value fitted to prod's 14.1% false-positive rate), so the data cannot choose between
# them - at 1.0 the false-positive row reads 0.098, already inside its 0.10 tolerance.
#
# The choice is not neutral for the decision this run exists to make, which is why it is not made by
# hand. Drift lowers retained reliability and raises the false-alarm rate, so it makes every
# downstream recovery metric worse and every retained subject more expensive - pushing the answer
# toward "N=50 is not enough", i.e. toward the options that cost more. Picking a value would tilt a
# spending decision on an assumption the data does not settle.
#
#   drift=1.0  parsimonious; the false-positive row stays EVIDENCE rather than a fitted target.
#   drift=1.1  matches the point estimates more closely, and is the conservative end.
#
# So DRIFT_LIST sweeps both and the comparison is reported under each. If the three options separate
# the same way at both ends, the question is moot and the report says so. If they do not, that IS
# the finding, and no single-value run could have surfaced it.
#
# COST, with drift swept over two values:
#   main    3 cells x 3 dispersion x 3 softness x 2 drift x 15 reps =   810 cohorts, x4 ndim = 3,240
#   RQ2     3 cells x 4 rho_true x 2 drift x 15 reps                =   360 cohorts, x1 ndim =   360
#   total                                                              1,170 cohorts and 3,600 fits
# Against v5's 2,880 cohorts and 17,280 fits: 41% of the generation, 21% of the SMACOF. Budget
# 6-8 h on a c7i.4xlarge - roughly double the single-drift plan. On-Demand, not Spot.
# =============================================================================================
set -euo pipefail

REPO_URL="${REPO_URL:?set REPO_URL to your repo (https with PAT, or git@ ssh)}"
GIT_REF="${GIT_REF:-main}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://my-bucket/spam-simulations/decision-v6}"
GT_S3_URI="${GT_S3_URI:-$S3_URI}"          # where run_gt_v6.sh wrote gt/
WORKDIR="${WORKDIR:-$HOME/spam_run}"

# 15 reps gives C(15,2)=105 cohort pairs per cell. The RQ2 power estimate is a proportion over
# those pairs, so its own precision is ~sqrt(p(1-p)/105) ~ 0.05 at p=0.5 - enough to separate
# "underpowered" from "adequate", which is all this run is asked to do.
REPS="${REPS:-15}"
if [ "$REPS" -lt 10 ]; then
  echo "!! REPS=$REPS gives only $(( REPS * (REPS - 1) / 2 )) cohort pairs, and the RQ2 power"
  echo "!! estimate is a proportion over them. Use REPS>=10 or override deliberately."
  exit 1
fi
N_JOBS="${N_JOBS:-$(( $(nproc) * 2 / 3 ))}"
export R_LIBS_USER="${R_LIBS_USER:-$HOME/R/library}"

# --------------------------------------------------------------------------- logging
LOGFILE="${WORKDIR%/}.log"
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1
_START_TS=$(date -u +%s)
echo ">> [start] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (S3_URI=$S3_URI, three-cell decision run)"

_on_exit() {
  local rc=$?
  if [ -n "${_PUSHER_PID:-}" ]; then
    kill "$_PUSHER_PID" 2>/dev/null || true
    wait "$_PUSHER_PID" 2>/dev/null || true
  fi
  # Every store, not just the main one: the RQ2 arms are as expensive to regenerate as the rest.
  for d in "${WORKDIR%/}"/repo/mds_store*; do
    [ -d "$d" ] || continue
    echo ">> [push] final snapshot of $(basename "$d") ..."
    aws s3 sync "$d/" "$S3_URI/$(basename "$d")/" \
      --exclude "*confdists.f32" --only-show-errors || true
  done
  [ -n "${PROD_DIR:-}" ] && rm -rf "$PROD_DIR"   # human-subjects data never left on the box
  echo ">> [end] $(date -u +'%Y-%m-%dT%H:%M:%SZ')  (exit $rc, elapsed $(( $(date -u +%s) - _START_TS ))s)"
  if [ -n "${S3_URI:-}" ]; then
    sleep 1
    aws s3 cp "$LOGFILE" "$S3_URI/run_decision_v6.log" --only-show-errors || true
  fi
}
trap _on_exit EXIT

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/prepare_machine.sh"

# ------------------------------------------------------------------- session loader + config
# TWO paths outside the sparse checkout (SpAM_Simulations only), and both are needed before any
# participant data can be read: analysis/utils/parser.py loads the sessions, and
# SpAM_Task/task_config.json supplies the screening thresholds. The thresholds are read from the
# DEPLOYED config rather than from constants precisely so an audit cannot drift from the gate
# participants actually faced - which means the file has to be on the box.
#
# Checking both in one loop, because checking only the first is exactly the bug this had: the
# run reached the calibration, spent 58s pulling stores, and died on a missing task_config.json.
git sparse-checkout add analysis/utils SpAM_Task || true
for f in analysis/utils/parser.py SpAM_Task/task_config.json; do
  if [ ! -f "$f" ]; then
    echo ">> [setup] sparse add did not materialise $f; forcing a path checkout ..."
    git checkout "$GIT_REF" -- "$(dirname "$f")" 2>/dev/null \
      || git checkout HEAD -- "$(dirname "$f")" 2>/dev/null || true
  fi
  if [ ! -f "$f" ]; then
    echo "!! [setup] $f still missing (checked out ref: $GIT_REF)."
    git sparse-checkout list || true
    exit 1
  fi
done
echo ">> [setup] parser and task_config present"

PROD_DIR="data"
echo ">> [data] fetching participant data from $S3_URI/data ..."
mkdir -p "$PROD_DIR"
aws s3 sync "$S3_URI/data" "$PROD_DIR/" --only-show-errors
shopt -s nullglob; _csvs=("$PROD_DIR"/*.csv); shopt -u nullglob
if [ ${#_csvs[@]} -eq 0 ] || [ ! -f "$PROD_DIR/stimuli_manifest.json" ]; then
  echo "!! no *.csv and/or stimuli_manifest.json under $S3_URI/data"; exit 1
fi

echo ">> [gt] pulling the v6 ground truth from $GT_S3_URI/gt/ ..."
mkdir -p gt
aws s3 sync "$GT_S3_URI/gt/" gt/ --only-show-errors || true
GT_FILE="${GT_FILE:-gt_pre_shine_v6_d8.npy}"   # what run_gt_v6.sh writes when the rebuild is accepted
if [ ! -f "gt/$GT_FILE" ]; then
  echo "!! gt/$GT_FILE not found under $GT_S3_URI/gt/."
  echo "!! This run never rebuilds the ground truth - doing so here would silently change it"
  echo "!! between the main arm and the RQ2 arms. Run run_gt_v6.sh first, or set GT_FILE to the"
  echo "!! pilot-only GT if its diagnostics gate failed."
  exit 1
fi

for d in mds_store out; do stage_pull "$d"; done
for r in 099 098 095 090; do stage_pull "mds_store_post_r$r"; done

PUSH_EVERY_S="${PUSH_EVERY_S:-1800}"
(
  while sleep "$PUSH_EVERY_S"; do
    for d in mds_store*; do
      [ -d "$d" ] || continue
      aws s3 sync "$d/" "$S3_URI/$d/" --exclude "*confdists.f32" --only-show-errors || true
    done
  done
) &
_PUSHER_PID=$!
echo ">> [push] background store sync every ${PUSH_EVERY_S}s (pid $_PUSHER_PID)"

N_JOBS="$N_JOBS" REPS="$REPS" S3_URI="$S3_URI" GT_FILE="$GT_FILE" SEED="${SEED:-42}" \
  SOFTNESS_LIST="${SOFTNESS_LIST:-3,4,8}" NDIMS="${NDIMS:-5,8,10,13}" \
  RHO_LIST="${RHO_LIST:-0.99,0.98,0.95,0.90}" \
  TABLES_ONLY="${TABLES_ONLY:-0}" REUSE_CALIBRATION="${REUSE_CALIBRATION:-1}" \
  DRIFT_LIST="${DRIFT_LIST:-1.0,1.1}" \
  python - <<'PY'
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd

from SpAM_Simulations.core import pipeline
from SpAM_Simulations.core.config import MDSSweepConfig, TaskV5SimulationConfig
from SpAM_Simulations.core.storage import ResultStore
from SpAM_Simulations.empirical import gt_diagnostics, screening_audit
from SpAM_Simulations.empirical.calibrate_v5 import calibrate
from SpAM_Simulations.empirical.gt_perturbation import build_perturbed_set
from SpAM_Simulations.empirical.subjects import load_prod_subjects
from SpAM_Simulations.measures import rq2_power, validity
from SpAM_Simulations.models import canvas as cv
from SpAM_Simulations.cli.run_v6_calibration_gate import TARGETS, _observed, _simulate_cell

N_JOBS = int(os.environ["N_JOBS"]); REPS = int(os.environ["REPS"]); SEED = int(os.environ["SEED"])
S3_URI = os.environ["S3_URI"]; GT_FILE = os.environ["GT_FILE"]
SOFTNESS = [float(x) for x in os.environ["SOFTNESS_LIST"].split(",")]
NDIMS = sorted({int(x) for x in os.environ["NDIMS"].split(",")})
RHOS = [float(x) for x in os.environ["RHO_LIST"].split(",")]
TABLES_ONLY = os.environ.get("TABLES_ONLY", "0") == "1"
REUSE = os.environ.get("REUSE_CALIBRATION", "1") == "1"

OUT = Path("out"); OUT.mkdir(exist_ok=True)
CAL = Path("calibration"); CAL.mkdir(exist_ok=True)
MANIFEST = "data/stimuli_manifest.json"

# The deployed design, from SpAM_Task/task_config.json.
IMAGES_PER_TRIAL, SCREEN_TRIALS, SCREEN_REPEATS = 20, 8, 2
EXP_TRIALS, EXP_REPEATS = 14, 2
FRAC_REPEATED = EXP_REPEATS / EXP_TRIALS
RANDOM_ARM, EXCLUDE_FA = 0.0, 1.0
# Fitted locally by `cli.run_v6_calibration_gate`, against production's 14.1% false-positive
# rate, and carried here as a CONSTANT. It is one free parameter tuned to one observed number, so
# the gate's false-positive row below is no longer evidence about the model. The other five rows
# are, and the drift also moves retained test-retest, so the gate can still fail.
DRIFT = [float(x) for x in os.environ["DRIFT_LIST"].split(",")]


def push(prefix, what):
    os.system(f'aws s3 sync {prefix}/ "{S3_URI}/{prefix}/" --exclude "*confdists.f32" '
              f'--only-show-errors || true')
    print(f"[push] {what}", flush=True)


coords = np.load(f"gt/{GT_FILE}")
print(f"[gt] {GT_FILE} {coords.shape}", flush=True)
simulator = cv.make_canvas_trial_simulator(sample_per_trial=True, softness=cv.DEFAULT_SOFTNESS)

# ---------------------------------------------------------------- calibration + the gate
reliability, agreement, n_attempted, kept = _observed("data", MANIFEST,
                                                      "SpAM_Task/task_config.json")
print(f"[prod] {n_attempted} candidates, {len(kept)} retained at rho>0 (false alarms excluded)",
      flush=True)
cal = calibrate(coords, kept, images_per_trial=IMAGES_PER_TRIAL, reps=6, cal_dir=CAL,
                trial_simulator=simulator, softness=cv.DEFAULT_SOFTNESS, gt_file=GT_FILE,
                n_dims=int(coords.shape[1]), reliability=reliability,   # screening-block MINIMA
                fit_n_repeats=SCREEN_REPEATS, fit_statistic="min",
                scale_from="distribution", reuse=REUSE)
print(f"[calibrated] noise_scale={cal['subjects_noise_scale']} family={cal['noise_family']} "
      f"shape={cal['noise_shape']} dispersion={cal['dispersion']} drift_swept={DRIFT}",
      flush=True)

gate_rows, failures = [], []
# The gate is checked at the FIRST swept drift. It passes 6/6 at both 1.0 and 1.1, and running it
# twice would only restate that; the sweep below is what carries the comparison.
GATE_DRIFT = DRIFT[0]
at0 = _simulate_cell(coords, cal, 0.0, n_subjects=50, reps=8, seed=SEED, simulator=simulator,
                     drift=GATE_DRIFT)
at01 = _simulate_cell(coords, cal, 0.1, n_subjects=50, reps=8, seed=SEED, simulator=simulator,
                      drift=GATE_DRIFT)
got = {"pass_rate_rho0": at0["pass_rate"], "pass_rate_rho01": at01["pass_rate"],
       "retained_tr_rho0": at0["median_tr"], "retained_tr_rho01": at01["median_tr"],
       "false_positive_rate_rho0": at0["false_positive_rate"],
       "agreement_retained": at0["agreement"]}
for name, (target, tol) in TARGETS.items():
    ok = bool(np.isfinite(got[name]) and abs(got[name] - target) <= tol)
    gate_rows.append({"quantity": name, "observed": target, "simulated": round(got[name], 4),
                      "tolerance": tol, "within_tolerance": ok})
    if not ok:
        failures.append(name)
pd.DataFrame(gate_rows).to_csv(OUT / "validation_gate.csv", index=False)
print(pd.DataFrame(gate_rows).to_string(index=False), flush=True)
if failures:
    # Loud, but not fatal: the gate is authoritative locally, before the instance is bought. By the
    # time we are here the money is spent, so finish the run and let the report carry the caveat.
    print(f"!! [gate] FAILED on {failures}. The run continues, but every downstream number "
          f"inherits this and the report must say so.", flush=True)
push("out", "validation gate")

# ---------------------------------------------------------------- the three cells
COMMON = dict(
    gt_embeddings=coords, trials_per_subject=[EXP_TRIALS], images_per_trial=[IMAGES_PER_TRIAL],
    subjects_noise_scale=[cal["subjects_noise_scale"]], subjects_noise_df=[cal["noise_df"]],
    subjects_noise_lognormal_sigma=[cal["noise_lognormal_sigma"]],
    frac_trials_repeated=[FRAC_REPEATED],
    perspective_dispersion=cal["dispersion_swept"],
    screening_trials=[SCREEN_TRIALS], screening_repeats=[SCREEN_REPEATS],
    allocation_mode=[RANDOM_ARM], canvas_softness=SOFTNESS,
    exclude_false_positives=[EXCLUDE_FA], within_session_drift=DRIFT, reps=REPS, seed=SEED,
)
# TWO configs, not one grid. num_subjects and screening_min_reliability are independent axes, so a
# single Cartesian product would also generate (N=75, rho>0.1) - a fourth cell that is out of budget
# and would cost a real 25% of the run. Both sweeps append to ONE store; run_mds_sweep keys on
# (params, rep, ndim), so they cannot collide.
CONFIGS = [
    ("N=50, rho>0 and rho>0.1", TaskV5SimulationConfig(
        num_subjects=[50], screening_min_reliability=[0.0, 0.1], **COMMON)),
    ("N=75, rho>0", TaskV5SimulationConfig(
        num_subjects=[75], screening_min_reliability=[0.0], **COMMON)),
]

sweep = MDSSweepConfig(ndims=NDIMS, max_iters=1000, convergence_tol=1e-6, precalc_init=True)
if TABLES_ONLY:
    store = ResultStore.open("mds_store")
    print(f"[tables-only] reopened mds_store with {len(store)} results", flush=True)
else:
    frames = []
    for label, cfg in CONFIGS:
        print(f"\n[cells] {label}", flush=True)
        sim = pipeline.generate_task_v5_simulation(cfg, verbose=True)
        frames.append((pipeline.compute_coverage_table(sim), pipeline.compute_stability_table(sim)))
        store = pipeline.run_mds_sweep(sim, sweep, "mds_store", parallel=True, n_jobs=N_JOBS,
                                       store_conf=True, verbose=True)
    # Written after both configs so the CSVs cover all three cells, but computed per config before
    # its sweep - neither table needs an MDS fit, and a run that dies mid-sweep keeps them.
    pd.concat([f[0] for f in frames], ignore_index=True).to_csv(OUT / "coverage.csv", index=False)
    pd.concat([f[1] for f in frames], ignore_index=True).to_csv(OUT / "stability.csv", index=False)
    push("out", "coverage + stability")
    print(f"\n[sweep] {len(store)} MDS results", flush=True)

# ---------------------------------------------------------------- the standard tables
gt_condensed = None
from scipy.spatial.distance import pdist
gt_condensed = pdist(coords)
pipeline.compute_embedding_stability(store).to_csv(OUT / "embedding_stability.csv", index=False)
pipeline.compute_embedding_generalizability(store).to_csv(
    OUT / "embedding_generalizability.csv", index=False)
pipeline.compute_topk_similar_pair_stability(store).to_csv(OUT / "topk_jaccard.csv", index=False)
pipeline.compute_recovery_vs_gt(store, gt_condensed).to_csv(OUT / "recovery_vs_gt.csv", index=False)
push("out", "embedding + recovery tables")

# ---------------------------------------------------------------- RQ2 power
# The NULL is free: two cohorts from the SAME ground truth is exactly what the main store holds.
null_draws = pipeline.embedding_stability_draws(store)
null_draws.to_csv(OUT / "rq2_null_draws.csv", index=False)

# The alternative needs cohorts on a perturbed ground truth. Restricted to the CALIBRATED setting
# (fitted dispersion, softness 4, ndim 8) - the sensitivity axes answer a different question and
# would multiply this arm by 12 for no gain.
alt_common = dict(COMMON)
alt_common["perspective_dispersion"] = [cal["dispersion"]]
alt_common["canvas_softness"] = [cv.DEFAULT_SOFTNESS]
alt_sweep = MDSSweepConfig(ndims=[8], max_iters=1000, convergence_tol=1e-6, precalc_init=True)

perturbed = build_perturbed_set(coords, RHOS, seed=SEED)
pd.DataFrame([info for _, info in perturbed.values()]).to_csv(
    OUT / "rq2_perturbations.csv", index=False)
for rho, (post_coords, info) in perturbed.items():
    if not info["converged"]:
        print(f"!! [rq2] perturbation for rho={rho} did not converge: {info}", flush=True)
    tag = f"{rho:.2f}".replace("0.", "r0")
    path = f"mds_store_post_{tag}"
    print(f"\n[rq2] rho_true={rho} -> {path} (achieved {info['achieved_rho']:.4f})", flush=True)
    alt_store = None
    for _, base_cfg in CONFIGS:
        cfg = TaskV5SimulationConfig(
            num_subjects=base_cfg.num_subjects,
            screening_min_reliability=base_cfg.screening_min_reliability,
            **{**alt_common, "gt_embeddings": post_coords})
        sim = pipeline.generate_task_v5_simulation(cfg, verbose=True)
        alt_store = pipeline.run_mds_sweep(sim, alt_sweep, path, parallel=True, n_jobs=N_JOBS,
                                           store_conf=True, verbose=True)
    draws = pipeline.cross_gt_draws(store, alt_store)
    draws.insert(0, "target_rho", rho)
    draws.to_csv(OUT / f"rq2_alt_draws_{tag}.csv", index=False)
    push("out", f"rq2 draws for rho={rho}")

alt_by_rho = {}
for rho in RHOS:
    tag = f"{rho:.2f}".replace("0.", "r0")
    f = OUT / f"rq2_alt_draws_{tag}.csv"
    if f.is_file():
        alt_by_rho[rho] = pd.read_csv(f)
if alt_by_rho:
    # The null comes from the full grid; restrict it to the calibrated cell the alternative used, or
    # its spread would carry the sensitivity axes the alternative never varied.
    cal_null = null_draws[
        np.isclose(null_draws["canvas_softness"], cv.DEFAULT_SOFTNESS)
        & np.isclose(null_draws["perspective_dispersion"], cal["dispersion"])
        & (null_draws["ndim"] == 8)]
    curve = rq2_power.power_curve(cal_null, alt_by_rho)
    curve.to_csv(OUT / "rq2_power.csv", index=False)
    rq2_power.minimum_detectable_effect(curve).to_csv(OUT / "rq2_mde.csv", index=False)
    print("\n--- RQ2 power ---", flush=True)
    print(curve.to_string(index=False), flush=True)
push("out", "rq2 power")
print("\n[done]", flush=True)
PY

upload_and_finish
