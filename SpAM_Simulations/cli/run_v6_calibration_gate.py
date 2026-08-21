"""Recalibrate on production, then check the model against what production already shows.

This is the go/no-go for the v6 run and is meant to be run **locally, before any machine is
provisioned**. It needs no MDS, no R and no store: fitting the noise population and simulating
screened cohorts are both pure numpy.

The logic. v5's participants were fitted to 41 pilot subjects on task versions 1.0-3.06 and are
about 25-50% too noisy for the deployed cohort, which is why its screening predictions miss badly
(a 71% pass rate against the observed 93%). v6 refits them on production. If the refitted model
still cannot reproduce the pass rates, retained reliability and false-positive rate we can already
*measure*, then its predictions about cells we cannot measure are not worth buying an instance for,
and this exits non-zero.

**The empirical sample is the screening block's own MINIMUM reliability, for all 84 candidates.**
Three reasons, and all three matter. The 67 retained subjects are a *truncated* sample of the
population the gate operates on, so fitting on them alone understates the lower tail - exactly the
part that determines what screening yields. Every candidate has exactly two screening repeats,
whereas `subject_reliability_sample` would average two repeats for the screened-out and four for the
completers; feeding a mixture of two- and four-repeat estimates into a distribution fit biases its
spread. And the gate itself (`_passes_screening`) thresholds the **minimum** of the repeats, not
their mean or median - so fitting the median puts the population's mass in the wrong place. The two
statistics agree near zero and diverge above it: the median sample rejects ~8% at rho>0.1 while the
minimum, which is what the gate reads, rejects 31% - against an observed 32.1%.

Usage::

    python -m SpAM_Simulations.cli.run_v6_calibration_gate \\
        --gt SpAM_Simulations/sim_results/v5/gt/gt_pre_shine_d8.npy \\
        --manifest SpAM_Task/stimuli_manifest.json --out SpAM_Simulations/sim_results/v6/calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Deployed design constants, from SpAM_Task/task_config.json.
IMAGES_PER_TRIAL = 20
SCREEN_TRIALS, SCREEN_REPEATS = 8, 2
EXP_TRIALS, EXP_REPEATS = 14, 2
TRIALS_PER_SESSION = EXP_TRIALS + SCREEN_TRIALS
FRAC_REPEATED = EXP_REPEATS / EXP_TRIALS

# What production already shows. The model has to land near these or the run is not worth having.
# Tolerances are deliberately loose: the point is to catch a model that is qualitatively wrong, not
# to demand a fit tighter than the 84 candidates behind these numbers can support.
TARGETS = {
    "pass_rate_rho0": (0.929, 0.10),
    "pass_rate_rho01": (0.679, 0.12),
    "retained_tr_rho0": (0.281, 0.06),
    "retained_tr_rho01": (0.350, 0.07),
    "false_positive_rate_rho0": (0.141, 0.10),
    "agreement_retained": (0.180, 0.06),
}


def _observed(data_dir: str, manifest: str, config: str, statistic: str = "min"):
    """The empirical reliability sample and the agreement target, from production."""
    from analysis.utils.parser import load_data
    from SpAM_Simulations.empirical import screening_audit as sa
    from SpAM_Simulations.empirical.subjects import (
        between_subject_agreement, load_prod_subjects, stack_distances)

    thr = sa.load_thresholds(config)
    data = load_data(data_dir)
    participants, trials = data["participants"], data["trials"]
    attempted = participants[(participants["cohort"] == "production")
                             & (participants["status"].isin(["full data", "screened out"]))].copy()
    other = ((attempted["move_ratio_fail_rate"] > thr["move_ratio_max_fail_rate"])
             | (attempted["distance_sd_fail_rate"] > thr["distance_sd_max_fail_rate"])).fillna(False)

    # Homogeneous across all 84: two screening repeats each, collapsed by the same statistic the
    # gate reads. See the module docstring.
    reliability = attempted[f"{statistic}_reliability"].astype(float).dropna().to_numpy()

    subjects = {s.participant_id: s for s in load_prod_subjects(data_dir, manifest)}
    screen_min = dict(zip(attempted["participant_id"], attempted["min_reliability"]))
    other_fail = dict(zip(attempted["participant_id"], other))
    experimental = {}
    for pid in subjects:
        mine = trials[(trials["participant_id"] == pid) & (~trials["is_catch"].astype(bool))]
        if "block_type" in mine.columns:
            mine = mine[mine["block_type"] == "experimental"]
        audit = sa.evaluate_screening(mine, thr)
        # BOTH criteria, not just reliability. Six production subjects fail the experimental block
        # on move-ratio alone, and omitting that check here would call them retained while every
        # other analysis calls them false alarms - a 73-vs-67 disagreement in the retained count.
        experimental[pid] = (
            audit["min_reliability"],
            audit["move_ratio_fails"] / max(audit["n_trials"], 1) > thr["move_ratio_max_fail_rate"]
            or audit["distance_sd_fails"] / max(audit["n_trials"], 1)
            > thr["distance_sd_max_fail_rate"])

    def retained(threshold):
        keep = []
        for pid in attempted["participant_id"]:
            if other_fail.get(pid, False) or (screen_min[pid] is not None
                                              and screen_min[pid] < threshold):
                continue
            if pid not in subjects:
                continue
            m, exp_other_fail = experimental[pid]
            if exp_other_fail or (m is not None and m < threshold):
                continue
            keep.append(subjects[pid])
        return keep

    kept = retained(0.0)
    agreement = between_subject_agreement(stack_distances(kept), min_overlap=20)["mean_agreement"]
    return reliability, float(agreement), len(attempted), kept


def _simulate_cell(coords, cal, threshold, *, n_subjects, reps, seed, simulator, drift=1.0,
                   softness=None):
    """Pass rate, retained reliability, agreement and false-positive rate at one gate.

    Builds a **v5** parameter tuple even though it calls the v4 simulator. v5's tuple is v4's fields
    in the same order plus the canvas and drift levers, and the v4 simulator reads its parameters by
    name, so this is exactly what ``simulate_task_v5_experiment`` does internally - it is how
    ``within_session_drift`` reaches the loop without a v4 field that the bit-exactness fixtures
    would notice.
    """
    from SpAM_Simulations.empirical.subjects import between_subject_agreement
    from SpAM_Simulations.models.canvas import DEFAULT_SOFTNESS
    from SpAM_Simulations.models.task_v4_experiment import simulate_task_v4_experiment
    from SpAM_Simulations.models.task_v5_experiment import TaskV5ExperimentParameters

    rows = []
    for r in range(reps):
        params = TaskV5ExperimentParameters(
            num_subjects=n_subjects, trials_per_subject=EXP_TRIALS,
            images_per_trial=IMAGES_PER_TRIAL,
            subjects_noise_scale=cal["subjects_noise_scale"], subjects_noise_df=cal["noise_df"],
            frac_trials_repeated=FRAC_REPEATED,
            perspective_dispersion=cal["dispersion"],
            screening_trials=SCREEN_TRIALS, screening_repeats=SCREEN_REPEATS,
            screening_min_reliability=threshold,
            subjects_noise_lognormal_sigma=cal["noise_lognormal_sigma"],
            allocation_mode=0.0,
            canvas_softness=DEFAULT_SOFTNESS if softness is None else softness,
            exclude_false_positives=0.0, within_session_drift=drift)
        _, res, per_subject = simulate_task_v4_experiment(
            params, coords, np.random.default_rng(seed + r), verbose=False,
            trial_simulator=simulator, return_per_subject=True)
        agreement = between_subject_agreement(per_subject, min_overlap=20)["mean_agreement"]
        rows.append({
            "pass_rate": res.screening_pass_rate,
            "median_tr": float(np.nanmedian(res.subject_test_retest)),
            "false_positive_rate": res.screening_false_positive_rate,
            "agreement": float(agreement),
        })
    return pd.DataFrame(rows).mean().to_dict()


DRIFT_GRID = (1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.8, 2.0)


def _fit_drift(coords, cal, *, reps, seed, simulator, grid=DRIFT_GRID):
    """Pick the single drift scalar whose false-positive rate best matches production's 14.1%.

    One free parameter fitted to one number, and it is worth being explicit that this is a *fit*,
    not a prediction: after it the false-positive row of the gate is no longer evidence about the
    model. The other five rows still are, and the drift also moves retained test-retest downward
    (a retained subject's analysed data now carries the drifted noise), so it is over-identified
    enough to fail.
    """
    target = TARGETS["false_positive_rate_rho0"][0]
    rows = []
    for d in grid:
        got = _simulate_cell(coords, cal, 0.0, n_subjects=50, reps=reps, seed=seed,
                             simulator=simulator, drift=float(d))
        rows.append({"drift": float(d), "false_positive_rate": got["false_positive_rate"],
                     "median_tr": got["median_tr"], "pass_rate": got["pass_rate"],
                     "gap": abs(got["false_positive_rate"] - target)})
        print(f"  drift={d:<5} false-positive {got['false_positive_rate']:.3f} "
              f"(target {target:.3f})  retained tr {got['median_tr']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    return float(table.loc[table["gap"].idxmin(), "drift"]), table


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--config", default="SpAM_Task/task_config.json")
    p.add_argument("--out", type=Path, default=Path("SpAM_Simulations/sim_results/v6/calibration"))
    p.add_argument("--reps", type=int, default=8, help="cohorts per validation cell")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reuse", action="store_true", help="ignore any cached calibration")
    p.add_argument("--fit-statistic", default="min", choices=("min", "median"),
                   help="how each candidate's two screening repeats collapse to one number. 'min' "
                        "is what the deployed gate thresholds; 'median' (= their mean, at two "
                        "repeats) exists to test whether that choice actually matters. Only these "
                        "two are stored per participant by analysis/utils/parser.py.")
    p.add_argument("--drift", type=float, default=None,
                   help="within-session drift multiplier; omit to fit it to the observed "
                        "false-positive rate")
    p.add_argument("--drift-reps", type=int, default=4, help="cohorts per drift-grid point")
    args = p.parse_args(argv)

    from SpAM_Simulations.empirical.calibrate_v5 import calibrate
    from SpAM_Simulations.models import canvas as cv

    coords = np.load(args.gt)
    reliability, agreement, n_attempted, kept = _observed(
        args.data_dir, str(args.manifest), args.config, statistic=args.fit_statistic)
    print(f"production: {n_attempted} candidates who sat the task, {len(kept)} retained at rho>0 "
          f"(false alarms excluded)")
    print(f"  screening-block {args.fit_statistic} reliability: n={len(reliability)} "
          f"median={np.median(reliability):.3f} q10={np.quantile(reliability, .1):.3f} "
          f"q90={np.quantile(reliability, .9):.3f}")
    print(f"  between-subject agreement of the retained: {agreement:.3f}")

    simulator = cv.make_canvas_trial_simulator(sample_per_trial=True, softness=cv.DEFAULT_SOFTNESS)
    args.out.mkdir(parents=True, exist_ok=True)
    cal = calibrate(coords, kept, images_per_trial=IMAGES_PER_TRIAL, reps=6, cal_dir=args.out,
                    trial_simulator=simulator, softness=cv.DEFAULT_SOFTNESS,
                    gt_file=args.gt.name, n_dims=int(coords.shape[1]),
                    reliability=reliability, fit_n_repeats=SCREEN_REPEATS,
                    fit_statistic=args.fit_statistic, scale_from="distribution",
                    reuse=not args.no_reuse)
    print(f"\n[calibrated] noise_scale={cal['subjects_noise_scale']} "
          f"family={cal['noise_family']} shape={cal['noise_shape']} "
          f"dispersion={cal['dispersion']}")
    print(f"             v5 used noise_scale=0.30, dispersion=0.25")

    drift = args.drift
    if drift is None:
        print("\nfitting within-session drift to the observed false-positive rate ...", flush=True)
        drift, drift_table = _fit_drift(coords, cal, reps=args.drift_reps, seed=args.seed,
                                        simulator=simulator)
        drift_table.to_csv(args.out / "drift_fit.csv", index=False)
        print(f"[drift] fitted within_session_drift={drift}", flush=True)
    else:
        print(f"\n[drift] within_session_drift={drift} (given, not fitted)", flush=True)

    print("\nsimulating the validation cells ...", flush=True)
    at0 = _simulate_cell(coords, cal, 0.0, n_subjects=50, reps=args.reps, seed=args.seed,
                         simulator=simulator, drift=drift)
    at01 = _simulate_cell(coords, cal, 0.1, n_subjects=50, reps=args.reps, seed=args.seed,
                          simulator=simulator, drift=drift)
    got = {
        "pass_rate_rho0": at0["pass_rate"],
        "pass_rate_rho01": at01["pass_rate"],
        "retained_tr_rho0": at0["median_tr"],
        "retained_tr_rho01": at01["median_tr"],
        "false_positive_rate_rho0": at0["false_positive_rate"],
        "agreement_retained": at0["agreement"],
    }

    rows, failures = [], []
    for name, (target, tol) in TARGETS.items():
        value = got[name]
        ok = bool(np.isfinite(value) and abs(value - target) <= tol)
        rows.append({"quantity": name, "observed": target, "simulated": round(float(value), 4),
                     "gap": round(float(value - target), 4), "tolerance": tol,
                     "within_tolerance": ok})
        if not ok:
            failures.append(name)
    table = pd.DataFrame(rows)
    print("\n--- validation gate ---")
    print(table.to_string(index=False))
    table.to_csv(args.out / "validation_gate.csv", index=False)
    (args.out / "validation_gate.json").write_text(json.dumps(
        {"targets": {k: v[0] for k, v in TARGETS.items()}, "simulated": got,
         "within_session_drift": float(drift), "drift_was_fitted": args.drift is None,
         "failures": failures}, indent=2))

    if failures:
        print(f"\nGATE FAILED on {len(failures)}: {', '.join(failures)}")
        print("The refitted model still does not reproduce what production already shows, so its")
        print("predictions about the cells we cannot measure are not worth provisioning for.")
        return 1
    print("\nGATE PASSED - the recalibrated model reproduces every measurable quantity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
