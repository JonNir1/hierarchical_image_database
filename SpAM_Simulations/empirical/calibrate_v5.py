"""Fit the task-v5 generative model to the pilot, once, and cache the result.

Three constants control every simulated cohort: the noise population (family, shape, scale), the
perspective dispersion, and the noise level that reproduces the empirical test-retest reliability.
All three are fitted against the pilot; none is chosen by hand.

**Why this is a module and not ninety lines inside an EC2 heredoc.** It used to be the latter, which
made it unreachable except by running the whole stage on an instance - so it could not be tested,
could not be re-run locally against a downloaded run, and had to be re-derived from scratch on every
resume, at roughly ten minutes a time. It is also the part of the pipeline where a silent mistake is
most expensive: a mis-scaled noise population mis-calibrates every result downstream, and two runs
were lost that way (a grid whose floor sat above the optimum, and a stale ground truth).

**Caching is fingerprint-checked, not assumed.** The fits are deterministic in their inputs, so a
resume that re-derives them wastes time for nothing. But reusing a calibration across a *changed*
input would mis-scale the run invisibly, which is far worse than the waste - so the fingerprint
covers the actual arrays, including a hash of the ground-truth coordinates rather than its filename.
A rebuilt GT written to the same path invalidates the cache, as it must.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from SpAM_Simulations.empirical.subjects import (
    between_subject_agreement, fit_dispersion_for_agreement, fit_noise_for_test_retest,
    fit_noise_population, subject_reliability_sample,
)

# The noise grid MUST be given explicitly. `fit_noise_population`'s own default spans [0.4, 2.6],
# written for the v3/v4 parameterisation where noise is a ratio to each trial's arrangement spread.
# Under the canvas it is an absolute fraction of canvas WIDTH and the optimum is an order of
# magnitude smaller: the pilot's median reliability of 0.243 is reproduced at ~0.22, BELOW the old
# grid's floor. Left at the default the fit pins to 0.4 and reports an achieved median of ~0.11.
CANVAS_NOISE_GRID = tuple(np.round(np.arange(0.02, 0.81, 0.02), 2))

# Dispersion is identified only through between-subject agreement, measured on the sparse pair
# overlap between subjects - the noisiest of the three anchors. So the fit is carried as a point
# estimate PLUS one step either side rather than trusted alone. DISP_MAX is the top of the range
# `fit_dispersion_for_agreement` searches: sweeping outside it would probe values the calibration
# could never have selected. The floor of 0 is real - it means every subject shares the ground-truth
# geometry exactly, and between-subject signal disagreement vanishes.
DISP_STEP, DISP_MAX = 0.15, 1.2

CALIBRATION_FILE = "calibration.json"
_MEDIAN_GAP_TOLERANCE = 0.05


class CalibrationError(RuntimeError):
    """Raised when a fit cannot be trusted, rather than returning a number that looks fine."""


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()[:16]


def fingerprint(coords: np.ndarray, reliability: np.ndarray, agreement: float,
                images_per_trial: int, reps: int, noise_grid: Sequence[float],
                softness: float) -> Dict[str, object]:
    """Everything that can change the fitted constants, hashed where it is an array.

    The ground truth is hashed rather than named. A GT rebuilt at a different dimensionality and
    written to the same filename is a different input, and a filename check would miss it - which is
    the exact shape of a bug that already cost two aborted runs.
    """
    return {
        "gt_sha": _sha(coords),
        "gt_shape": list(coords.shape),
        "reliability_sha": _sha(np.asarray(reliability)),
        "n_reliability": int(np.size(reliability)),
        "agreement": round(float(agreement), 10),
        "images_per_trial": int(images_per_trial),
        "reps": int(reps),
        "noise_grid": [round(float(x), 6) for x in noise_grid],
        "softness": round(float(softness), 6),
        "disp_step": DISP_STEP,
        "disp_max": DISP_MAX,
    }


def dispersion_sweep(dispersion: float) -> list:
    """The fitted dispersion plus one step either side, clamped to the fitter's own search range."""
    return sorted({round(min(max(dispersion + d, 0.0), DISP_MAX), 2)
                   for d in (-DISP_STEP, 0.0, DISP_STEP)})


def load_cached(cal_dir: Path, expected: Dict[str, object],
                verbose: bool = True) -> Optional[Dict[str, object]]:
    """A previous calibration, but only if every fingerprinted input is unchanged."""
    path = Path(cal_dir) / CALIBRATION_FILE
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if verbose:
            print(f"[calibrate] ignoring unreadable {path}: {exc}", flush=True)
        return None
    if cached.get("fingerprint") != expected:
        if verbose:
            differing = sorted(k for k in expected
                               if cached.get("fingerprint", {}).get(k) != expected[k])
            print(f"[calibrate] cached calibration is stale; inputs changed: {differing}",
                  flush=True)
        return None
    if verbose:
        print(f"[calibrate] reusing {path} - every fitted input is unchanged. "
              f"Set REUSE_CALIBRATION=0 to force a re-fit.", flush=True)
    return cached


def calibrate(coords: np.ndarray, subjects: Sequence, *, images_per_trial: int, reps: int,
              cal_dir: Path, trial_simulator, softness: float, gt_file: str = "",
              n_dims: Optional[int] = None, scan_selected_n_dims: Optional[int] = None,
              noise_grid: Sequence[float] = CANVAS_NOISE_GRID, reuse: bool = True,
              write: bool = True, verbose: bool = True,
              reliability: Optional[np.ndarray] = None,
              fit_n_repeats: int = 4) -> Dict[str, object]:
    """Fit (or reuse) the three constants. Returns the dict written to ``calibration.json``.

    ``subjects`` should be ALL subjects of the cohort being calibrated on, both SHINE variants:
    reliability and between-subject agreement are properties of people, not of the stimulus set,
    so the post-SHINE half is perfectly good evidence about them even though it is unusable for a
    ground truth over the pre-SHINE set.

    ``trial_simulator`` is fitted at ONE softness deliberately. Re-fitting the noise population per
    softness value would confound the sensitivity arm with three different noise scales; the arm
    then varies softness around this one calibrated level.

    ``reliability`` overrides the per-subject sample derived from ``subjects``. Pass it when the
    subjects' own reliabilities are not homogeneously estimated: the production cohort's completers
    have four repeats while its screened-out candidates have only the screening block's two, and
    feeding a mixture of two- and four-repeat estimates into a Wasserstein *distribution* fit biases
    the fitted spread. The screening block's own ``median_reliability`` is two repeats for every
    candidate, which is the homogeneous choice.

    ``fit_n_repeats`` must then match how that sample was measured, because it sets how many repeats
    each *simulated* subject contributes. Leaving it at 4 against a two-repeat empirical sample makes
    the simulated distribution artificially tight and the fitted population artificially narrow.

    Raises :class:`CalibrationError` when the noise scale pins to an edge of ``noise_grid``, because
    a grid that could not reach the data mis-calibrates everything downstream - worth aborting a
    fifteen-hour run for.
    """
    cal_dir = Path(cal_dir)
    cal_dir.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords)

    # Both cheap, and both needed to fingerprint the expensive fits that follow.
    reliability = (subject_reliability_sample(subjects) if reliability is None
                   else np.asarray(reliability, dtype=float))
    reliability = reliability[np.isfinite(reliability)]
    agreement = between_subject_agreement(
        np.vstack([s.distances for s in subjects]), min_overlap=20)["mean_agreement"]
    prints = fingerprint(coords, reliability, agreement, images_per_trial, reps, noise_grid,
                         softness)
    # Part of the cache key: the same reliability sample fitted at a different repeat count is a
    # different fit, and a filename-blind cache would happily serve the wrong one.
    prints["fit_n_repeats"] = int(fit_n_repeats)

    if reuse:
        cached = load_cached(cal_dir, prints, verbose=verbose)
        if cached is not None:
            return cached

    if verbose:
        print(f"\n[calibrate] {len(subjects)} sessions (both SHINE variants) for the agreement target",
              flush=True)
        print(f"[shape] empirical reliability: n={len(reliability)} "
              f"median={np.median(reliability):.3f} q10={np.quantile(reliability, .1):.3f} "
              f"q90={np.quantile(reliability, .9):.3f}", flush=True)

    fit = fit_noise_population(coords, reliability, images_per_trial=images_per_trial,
                               perspective_dispersion=0.2, n_subjects=80, reps=3, verbose=verbose,
                               n_repeats=fit_n_repeats,
                               noise_grid=tuple(noise_grid), trial_simulator=trial_simulator)
    best = fit["best"]
    if write:
        fit["grid"].to_csv(cal_dir / "noise_shape_grid.csv", index=False)
        (cal_dir / "noise_shape_fit.json").write_text(json.dumps(
            {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in best.items()}, indent=2, default=str))
    if verbose:
        print(f"[shape] best: family={best['family']} shape={best['shape']} "
              f"scale={best['noise_scale']} W1={best['distance']:.4f} CV={best['cv']:.3f} "
              f"boundary={best['at_shape_boundary']}", flush=True)
        if best["at_shape_boundary"]:
            print("[shape] WARNING: fit sits on a shape-grid boundary - the family, not the data, "
                  "is the binding constraint.", flush=True)
    if best["at_noise_boundary"]:
        raise CalibrationError(
            f"the fitted noise scale {best['noise_scale']:.3f} sits on the edge of the search grid "
            f"[{best['noise_grid_min']:.2f}, {best['noise_grid_max']:.2f}], so the grid could not "
            f"reach the data: achieved median reliability {best['sim_median']:.3f} against an "
            f"empirical {best['empirical_median']:.3f} (gap {best['median_gap']:+.3f}). Widen the "
            f"grid in the direction of the boundary and re-run; continuing would mis-calibrate "
            f"every result in this sweep."
        )
    if verbose and abs(best["median_gap"]) > _MEDIAN_GAP_TOLERANCE:
        print(f"[shape] WARNING: best fit lands {best['median_gap']:+.3f} from the empirical median "
              f"({best['sim_median']:.3f} vs {best['empirical_median']:.3f}) without hitting a grid "
              f"edge, so the noise FAMILY may be the binding constraint rather than its scale.",
              flush=True)

    lognormal_sigma = float(best["shape"]) if best["family"] == "lognormal" else 0.0
    noise_df = int(best["shape"]) if best["family"] == "t" else 5

    # Re-fitted UNDER the newly fitted noise population: agreement depends on the whole noise
    # distribution, so a dispersion calibrated against a different one does not transfer.
    dispersion, dispersion_achieved = fit_dispersion_for_agreement(
        coords, agreement, noise_scale=float(best["noise_scale"]), noise_df=noise_df,
        lognormal_sigma=lognormal_sigma, images_per_trial=images_per_trial, reps=reps,
        trial_simulator=trial_simulator)
    swept = dispersion_sweep(dispersion)
    if verbose:
        print(f"[disp] empirical agreement={agreement:.4f} -> dispersion={dispersion:.2f} "
              f"(achieved {dispersion_achieved:.4f})", flush=True)
        print(f"[disp] sweeping dispersion over {swept}"
              + (" (clamped, so fewer than 3 values)" if len(swept) < 3 else ""), flush=True)

    # One noise level, at the empirically achieved test-retest. The R axis was swept in
    # task-v4-fitted and is not what this run is about: holding it fixed keeps every fit paying for
    # the arm contrast.
    target_r = float(np.median(reliability))
    noise, achieved_r = fit_noise_for_test_retest(
        coords, target_r, noise_df=noise_df, lognormal_sigma=lognormal_sigma,
        images_per_trial=images_per_trial, reps=reps, trial_simulator=trial_simulator)
    if verbose:
        print(f"[invert] targetR={target_r:.3f} -> noise={noise:.2f} "
              f"(achieved {achieved_r:.3f} @img{images_per_trial})", flush=True)

    result = dict(
        n_pilot_sessions=len(subjects), empirical_agreement=float(agreement),
        dispersion=float(dispersion), dispersion_achieved=float(dispersion_achieved),
        dispersion_swept=swept, target_test_retest=target_r,
        achieved_tr_unscreened=float(achieved_r), subjects_noise_scale=float(noise),
        noise_family=best["family"], noise_shape=float(best["shape"]),
        noise_df=noise_df, noise_lognormal_sigma=lognormal_sigma,
        n_dims=n_dims, gt_file=gt_file, scan_selected_n_dims=scan_selected_n_dims,
        fingerprint=prints,
    )
    if write:
        (cal_dir / CALIBRATION_FILE).write_text(json.dumps(result, indent=2))
    return result
