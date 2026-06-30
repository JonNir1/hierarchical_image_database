"""Calibrate the task-v3 simulation to the SpAM pilot, end to end.

Pipeline (see ``SpAM_Simulations/pilot.py`` for the building blocks and the plan for rationale):

  A. Ground truth  - pool ALL completed pilot subjects -> weighted MDS -> coordinates (inherits the
     real spectrum + cluster structure, so `decay`/`n_clusters` are moot; `n_dims` from the spectrum).
  B. Targets       - from the v3.0 subjects (matched 20x20 design, 3 repeats): within-subject
     test-retest and between-subject agreement.
  C. Fit           - sequential 1-D: test-retest -> `subjects_noise_scale` (perspective-invariant),
     then between-subject agreement -> `perspective_dispersion`.
  D. Required-N    - print a calibrated `TaskV3SimulationConfig` to feed the existing convergence
     sweep (`pipeline.generate_task_v3_simulation` -> `run_mds_sweep` -> `plateau_num_subjects`).

Steps A (default) and D need R + rpy2 (weighted SMACOF). Pass ``--gt-method classical`` for a no-R
**provisional** ground truth (numpy classical MDS) to get a preliminary calibration without R; the
final numbers should use SMACOF on the R-enabled machine.

Usage (from the repo root):
    python -m SpAM_Simulations.calibrate_to_pilot --pilot-dir data/pilot \
        --manifest SpAM_Task/stimuli_manifest.json [--gt-method smacof|classical]

Reads only; writes nothing (pilot data and pilot-derived artifacts stay local).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from SpAM_Simulations import pilot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-dir", default="data/pilot")
    ap.add_argument("--manifest", default="SpAM_Task/stimuli_manifest.json")
    ap.add_argument("--gt-method", choices=["smacof", "classical"], default="smacof",
                    help="GT embedding solver; 'classical' is a no-R provisional fallback")
    ap.add_argument("--n-dims", type=int, default=None, help="override GT dimensionality")
    ap.add_argument("--reps", type=int, default=5, help="cohorts averaged per simulated point")
    ap.add_argument("--save-gt", default=None,
                    help="if set, np.save the pilot GT coordinates here (e.g. out/gt_pilot_coords.npy)")
    ap.add_argument("--save-params", default=None,
                    help="if set, write the fitted parameters as JSON here (e.g. out/calibrated_params.json)")
    args = ap.parse_args()

    # --- load -------------------------------------------------------------------------------
    allsub = pilot.load_pilot_subjects(args.pilot_dir, args.manifest)
    v3 = [s for s in allsub if s.task_version == "3.0"]
    print(f"[load] {len(allsub)} completed sessions; {len(v3)} are v3.0 (matched design)")
    if not v3:
        raise SystemExit("no v3.0 subjects found - calibration targets need the matched design")

    # --- B. targets (no R) ------------------------------------------------------------------
    tr = pilot.cohort_test_retest(v3)
    agr = pilot.between_subject_agreement(pilot.stack_distances(v3))
    print(f"[targets] within-subject test-retest (median) = {tr:.4f}")
    print(f"[targets] between-subject agreement = {agr['mean_agreement']:.4f} "
          f"(SEM {agr['sem_agreement']:.4f}, {agr['n_dyads']} dyads, median overlap {agr['median_overlap']:.0f})")

    # --- A. ground truth --------------------------------------------------------------------
    coords, info = pilot.build_gt_from_pilot(allsub, n_dims=args.n_dims, method=args.gt_method)
    print(f"[gt] {info['method']} embedding: N={coords.shape[0]}, n_dims={info['n_dims']}, "
          f"observed {info['observed_frac']:.1%} of pairs")
    if args.gt_method == "classical":
        print("[gt] WARNING: provisional no-R ground truth (numpy classical MDS). "
              "Re-run with --gt-method smacof for the final numbers.")

    # --- C. fit -----------------------------------------------------------------------------
    fit = pilot.calibrate(coords, v3, reps=args.reps)
    print("\n=== CALIBRATED PARAMETERS ===")
    print(f"  subjects_noise_scale   = {fit['subjects_noise_scale']:.3f}  "
          f"(sim test-retest {fit['simulated_test_retest']:.3f} vs pilot {fit['pilot_test_retest']:.3f})")
    print(f"  perspective_dispersion = {fit['perspective_dispersion']:.3f}  "
          f"(sim agreement {fit['simulated_between_agreement']:.3f} vs pilot {fit['pilot_between_agreement']:.3f})")
    print(f"  subjects_noise_df      = {fit['subjects_noise_df']}   |   n_dims = {info['n_dims']}")

    # --- optional artifacts (for the downstream sweep) --------------------------------------
    if args.save_gt:
        np.save(args.save_gt, coords)
        print(f"[save] GT coordinates -> {args.save_gt}  {coords.shape}")
    if args.save_params:
        with open(args.save_params, "w", encoding="utf-8") as fh:
            json.dump({**fit, "n_dims": info["n_dims"], "gt_method": args.gt_method}, fh, indent=2)
        print(f"[save] fitted parameters -> {args.save_params}")

    # --- D. ready-to-run convergence sweep --------------------------------------------------
    print("\n=== NEXT: calibrated convergence sweep (needs R) ===")
    print(f"""    from SpAM_Simulations.config import TaskV3SimulationConfig, MDSSweepConfig
    from SpAM_Simulations import pipeline, eval_helpers, pilot
    coords, _ = pilot.build_gt_from_pilot(allsub, n_dims={info['n_dims']}, method='smacof')
    cfg = TaskV3SimulationConfig(                              # GT = the calibrated pilot embedding
        gt_embeddings=coords, num_subjects=[20, 50, 100, 200, 350, 500],
        trials_per_subject=[20], images_per_trial=[20],
        subjects_noise_scale=[{fit['subjects_noise_scale']:.3f}], subjects_noise_df=[1],
        frac_trials_repeated=[0.15], perspective_dispersion=[{fit['perspective_dispersion']:.3f}],
        reps=5, seed=42)
    sim = pipeline.generate_task_v3_simulation(cfg)
    store = pipeline.run_mds_sweep(sim, MDSSweepConfig(min_ndim=2), 'mds_store', parallel=True)
    es = pipeline.compute_embedding_stability(store)
    print(eval_helpers.plateau_num_subjects(es))             # required-N per ndim""")


if __name__ == "__main__":
    main()
