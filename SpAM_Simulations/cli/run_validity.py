"""Run the noise-shape validity check locally, against a downloaded run's calibration.

This is the sharpest of the validity checks and the only one that does not need the sweep. The
empirical finding it tests (``analysis/pilot/figures.py``, "Reliability vs. distance"): for every
pair a subject judged twice, RMSE between the two judgements against their mean is an inverted U.
Clearly-similar and clearly-dissimilar pairs are judged consistently; the ambiguous middle is not.
The high-distance turnover is the discriminating half, because it requires a bounded canvas - a pair
already at opposite corners cannot move much further apart. The model is not fitted to any of this,
which is what makes it worth measuring.

It exists as a separate entrypoint for two reasons. It measures a property of the **calibrated noise
model**, not of any sweep cell, so it needs no MDS, no store, and no R, and runs in seconds; and when
it ran only at the tail of a 15-hour EC2 script, a ``TypeError`` on its last call destroyed the
output of a completed run that was otherwise entirely sound. Everything it needs is recorded in
``calibration/calibration.json``, so it is reproducible from a downloaded run forever after.

Usage (from the repo root, after cookbook step E)::

    python -m SpAM_Simulations.cli.run_validity \\
        --run SpAM_Simulations/sim_results/design-comparison-v5 \\
        --manifest SpAM_Task/stimuli_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from SpAM_Simulations.models import canvas as cv
from SpAM_Simulations.measures import validity

# The stage-2 defaults. They are arguments rather than constants because a run that used different
# ones must be scored with its own, but in practice they come from the run being reproduced.
DEFAULT_IMAGES_PER_TRIAL = 20
DEFAULT_SEED = 42
DEFAULT_N_SUBJECTS = 60
DEFAULT_TRIALS_PER_SUBJECT = 4
DEFAULT_CURVE_DRAWS = 20
DEFAULT_NULL_TRIALS = 2000


def _resolve_gt(run: Path, calibration: dict, override: Optional[Path]) -> Path:
    """Find the ground-truth coordinates, looking in ``gt/`` as well as the run root.

    Stage 1 writes the coordinates into ``<run>/gt/``, which is where they belong; earlier ad-hoc
    downloads dropped a copy at the run root. Both are searched so neither layout breaks this.
    """
    if override is not None:
        return override
    name = str(calibration.get("gt_file", ""))
    for candidate in (run / name, run / "gt" / name):
        if name and candidate.is_file():
            return candidate
    found = sorted(run.glob("*.npy")) + sorted((run / "gt").glob("*.npy"))
    if len(found) == 1:
        return found[0]
    raise SystemExit(
        f"cannot locate the ground truth. calibration.json names {name!r}, which is at neither "
        f"{run / name} nor {run / 'gt' / name}, and globbing found {len(found)} .npy files.\n"
        f"  Sync stage 1's gt/ prefix beside the run, or pass --gt explicitly."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path,
                   help="downloaded run directory (holds calibration/ and out/)")
    p.add_argument("--manifest", required=True, type=Path, help="stimuli_manifest.json")
    p.add_argument("--pilot-dir", default="data", help="directory of pilot session CSVs")
    p.add_argument("--gt", type=Path, default=None,
                   help="ground-truth .npy; defaults to the file calibration.json names")
    p.add_argument("--out", type=Path, default=None, help="defaults to <run>/out")
    p.add_argument("--softness", type=float, default=None,
                   help="canvas softness; defaults to canvas.DEFAULT_SOFTNESS, which is what the "
                        "calibration itself was run at")
    p.add_argument("--n-subjects", type=int, default=DEFAULT_N_SUBJECTS)
    p.add_argument("--trials-per-subject", type=int, default=DEFAULT_TRIALS_PER_SUBJECT)
    p.add_argument("--images-per-trial", type=int, default=DEFAULT_IMAGES_PER_TRIAL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--curve-draws", type=int, default=DEFAULT_CURVE_DRAWS,
                   help="independent simulated cohorts behind the binned-RMSE curve. Their SD is "
                        "the spread the report plots; the per-bin SEM is a precision, not a spread")
    p.add_argument("--null-trials", type=int, default=DEFAULT_NULL_TRIALS,
                   help="trials of uniformly-random placement behind the null distance summary")
    args = p.parse_args(argv)

    from SpAM_Simulations.empirical.pilot import load_pilot_subjects

    cal_path = args.run / "calibration" / "calibration.json"
    if not cal_path.is_file():
        raise SystemExit(f"no calibration at {cal_path}; sync the run's calibration/ prefix first")
    cal = json.loads(cal_path.read_text())

    noise = float(cal["subjects_noise_scale"])
    dispersion = float(cal["dispersion"])
    family = cal["noise_family"]
    lognormal_sigma = float(cal["noise_shape"]) if family == "lognormal" else 0.0
    noise_df = int(cal["noise_shape"]) if family == "t" else 5
    softness = cv.DEFAULT_SOFTNESS if args.softness is None else args.softness

    coords = np.load(_resolve_gt(args.run, cal, args.gt))
    print(f"GT {coords.shape}; noise={noise} family={family} shape={cal['noise_shape']} "
          f"dispersion={dispersion} softness={softness} seed={args.seed}")

    # The same simulator the calibration was routed through: aspect and fill sampled per trial from
    # the pilot's measured marginals, at the softness the noise scale was fitted at. Passing it is
    # the entire point - the turnover is caused by the canvas bound.
    simulator = cv.make_canvas_trial_simulator(sample_per_trial=True, softness=softness)

    def draw_pairs(seed: int):
        return validity.simulate_repeat_pairs(
            coords, subjects_noise_scale=noise, n_subjects=args.n_subjects,
            trials_per_subject=args.trials_per_subject, images_per_trial=args.images_per_trial,
            perspective_dispersion=dispersion, noise_df=noise_df,
            lognormal_sigma=lognormal_sigma, seed=seed, trial_simulator=simulator)

    sim_pairs = draw_pairs(args.seed)

    subjects = load_pilot_subjects(args.pilot_dir, str(args.manifest))
    print(f"pilot subjects: {len(subjects)} (calibration recorded {cal.get('n_pilot_sessions')})")
    if len(subjects) != cal.get("n_pilot_sessions"):
        print("!! subject count differs from the calibration's - the comparison is against a "
              "different cohort than the run used", flush=True)
    pilot_pairs = validity.repeat_pairs(subjects)
    print(f"repeat pairs: sim={sim_pairs[0].size}, pilot={pilot_pairs[0].size}")
    if not pilot_pairs[0].size:
        raise SystemExit("no pilot repeat trials found; nothing to compare against")

    report = validity.compare_noise_vs_distance(sim_pairs, pilot_pairs)
    out = (args.run / "out") if args.out is None else args.out
    out.mkdir(parents=True, exist_ok=True)
    report["curves"].to_csv(out / "noise_vs_distance.csv", index=False)
    report["shape"].to_csv(out / "noise_curve_shape.csv", index=False)

    # Under task-v5 both sources are already canvas-diagonal normalised, so the curve can also be
    # read in native units, with a spread that is a spread (SD across cohorts) rather than a
    # within-bin precision. The median-rescaled table above is kept for the shape verdicts.
    print(f"\n{args.curve_draws} cohorts for the native-unit curve ...", flush=True)
    native_sim = validity.noise_vs_distance_draws(
        draw_pairs, n_draws=args.curve_draws, base_seed=args.seed, rescale="none")
    native_pilot = validity.noise_vs_distance(*pilot_pairs, rescale="none")
    native = pd.concat([native_sim.assign(source="sim"),
                        native_pilot.assign(source="pilot", n_draws=1, sd_rmse=np.nan)],
                       ignore_index=True)
    native.to_csv(out / "noise_vs_distance_native.csv", index=False)

    nulls = validity.null_distance_summary(
        np.concatenate(sim_pairs), np.concatenate(pilot_pairs),
        num_dots=args.images_per_trial, num_trials=args.null_trials, seed=args.seed)
    nulls.to_csv(out / "null_distances.csv", index=False)
    print("\n--- distances vs random placement ---")
    print(nulls.round(4).to_string(index=False))

    print("\n--- shape summary ---")
    print(report["shape"].to_string(index=False))
    print(f"\nturnover_matches={report['turnover_matches']} "
          f"low_end_matches={report['low_end_matches']} drop_gap={report['drop_gap']:.3f}")
    if not report["turnover_matches"]:
        print("[noise] the curve does not turn over as the pilot's does. Check the canvas bound is "
              "active - an unbounded model cannot produce this.", flush=True)
    if not report["low_end_matches"]:
        print("[noise] WARNING: the LOW end does not match either. That one is near-forced for any "
              "additive-noise model, so a mismatch there points at a real problem.", flush=True)
    print(f"\nwrote noise_vs_distance.csv, noise_vs_distance_native.csv, noise_curve_shape.csv "
          f"and null_distances.csv to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
