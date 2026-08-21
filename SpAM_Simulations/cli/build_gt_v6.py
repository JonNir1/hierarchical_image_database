"""Rebuild the ground truth from the pilot PLUS the production subjects the analysis discards.

WHY. v5's ground truth is fitted on 41 pre-SHINE pilot subjects and its diagnosed weakness is that
it exceeds its own data's reliability ceiling: it interpolates where the pair graph is thin. The
direct cure for that is replication, not more dimensions, and there are eight pre-SHINE production
subjects sitting unused - the ones the deployed gate rejected in-task, plus the ones who cleared it
and then failed the same rule on their experimental block.

Adding them is a real gain in exactly the quantity that is binding:

    pairs seen at least twice   14,123 -> 21,022   (+49%)
    pair-observations          108,300 -> 139,080  (+28%)
    pairs covered ever          82,645 -> 99,169   (31.5% -> 37.8%)

The first row is the point: replication, not reach, is what the ceiling is short of.

NOT CIRCULAR, and this is the part worth being careful about. Every one of the eight is excluded
from the analysed pool by a rule fixed before this analysis: the deployed gate, plus that same gate
re-scored against the experimental block. No subject who will appear in the study's results
contributes to the ground truth the study's design was chosen against. They are also pre-SHINE, so
they judged the same stimulus set the GT is a geometry over.

THE GATE IS ONE-SIDED, DELIBERATELY. These are the *worst* eight subjects available; they add
coverage and noise together. But the comparison is not symmetric: the augmented set has 49 subjects
against 41, so its split-halves are larger and agree better for reasons that have nothing to do with
whether the added data is any good. A small improvement is therefore weak evidence, while a
*worsening* - despite that mechanical advantage - is strong evidence the eight are harmful. So this
rejects the rebuild only on a worsening beyond noise, and never claims a small gain as vindication.

Usage::

    python -m SpAM_Simulations.cli.build_gt_v6 --gt-dir gt --ndim 8
        --data-dir data --manifest data/stimuli_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# The pilot's own pre-SHINE count, and the number of pre-SHINE production candidates the analysis
# discards. Both are asserted rather than trusted: the subject set defines the ground truth, so a
# silent change to either invalidates every downstream comparison.
EXPECT_PILOT = 41
EXPECT_EXCLUDED = 8


def excluded_prod_subjects(data_dir: str, manifest: str, config: str, *,
                           variants: Sequence[str] = ("pre",)) -> List:
    """The production candidates the analysis discards, loaded as ``Subject`` records.

    Both discard routes, since both are decided by a rule fixed in advance: early fails (rejected
    in-task, paid the reduced rate) and false alarms (cleared the gate, paid in full, then failed
    the same rule on the experimental block). Early fails carry only their screening block, which is
    real data over real pairs and is exactly what this is for.
    """
    from analysis.utils.parser import load_data
    from SpAM_Simulations.empirical import screening_audit as sa
    from SpAM_Simulations.empirical.subjects import load_prod_subjects

    thr = sa.load_thresholds(config)
    data = load_data(data_dir)
    part = sa.partition_candidates(data["participants"], data["trials"], thr, threshold=0.0)
    discarded = set(part["early_fail"]) | set(part["false_alarm"])
    # statuses must include "screened out" or the early fails are dropped before we can see them.
    everyone = load_prod_subjects(data_dir, manifest, variants=tuple(variants),
                                  statuses=("full data", "screened out"))
    return [s for s in everyone if s.participant_id in discarded]


def _diagnostics(subjects: Sequence, ndim: int, *, n_draws: int, n_jobs: int, seed: int,
                 images: Sequence[str], verbose: bool = True) -> Dict[str, float]:
    """Split-half agreement at one dimensionality, plus the raw data's own noise ceiling."""
    from SpAM_Simulations.empirical import gt_construction as gtc
    from SpAM_Simulations.empirical.gt_diagnostics import raw_noise_ceiling
    from SpAM_Simulations.measures.validity import hierarchy_levels

    splits, split_info = gtc.draw_valid_splits(subjects, n_draws, np.random.default_rng(seed))
    aggregates = gtc.split_aggregates(subjects, splits)
    rows = gtc.scan_ndim_parallel(aggregates, ndim, n_jobs=n_jobs, verbose=verbose)
    ceiling = raw_noise_ceiling(subjects, hierarchy_levels(images), rng=np.random.default_rng(seed))
    statuses = rows["status_a"].tolist() + rows["status_b"].tolist()
    return {
        "n_subjects": len(subjects),
        "coverage": float(gtc.coverage_of(subjects)),
        "split_half_spearman": float(rows["spearman"].mean()),
        # The SEM is what makes the one-sided gate meaningful: "worse" has to mean worse than the
        # draw-to-draw noise, not worse than the point estimate.
        "split_half_spearman_sem": float(rows["spearman"].std(ddof=1) / np.sqrt(len(rows))),
        "split_half_procrustes_m2": float(rows["procrustes_m2"].mean()),
        "split_half_topk_jaccard": float(rows["topk_jaccard"].mean()),
        "max_iters_rate": float(statuses.count("max_iters") / max(len(statuses), 1)),
        "mean_noise_ceiling_full": float(ceiling["ceiling_full"].mean()),
        "half_size": int(split_info["half_size"]),
        "discard_rate": float(split_info["discard_rate"]),
        "coverage_gap_frac": float(split_info["coverage_gap_frac"]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-dir", type=Path, default=Path("gt"))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--config", default="SpAM_Task/task_config.json")
    p.add_argument("--ndim", type=int, default=8)
    p.add_argument("--n-draws", type=int, default=30, help="split-half draws per subject set")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--method", default="smacof", choices=("smacof", "classical"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tolerance", type=float, default=2.0,
                   help="how many SEMs of split-half Spearman the augmented set may fall below the "
                        "pilot-only set before the rebuild is rejected")
    p.add_argument("--expect-pilot", type=int, default=EXPECT_PILOT)
    p.add_argument("--expect-excluded", type=int, default=EXPECT_EXCLUDED)
    p.add_argument("--skip-comparison", action="store_true",
                   help="build the augmented GT without scoring either set (for a smoke test)")
    args = p.parse_args(argv)

    from SpAM_Simulations.empirical import gt_construction as gtc
    from SpAM_Simulations.empirical.subjects import load_pilot_subjects

    images = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
    pilot = load_pilot_subjects(args.data_dir, str(args.manifest), variants=("pre",))
    extra = excluded_prod_subjects(args.data_dir, str(args.manifest), args.config)
    print(f"[gt-v6] {len(pilot)} pre-SHINE pilot subjects + {len(extra)} discarded pre-SHINE "
          f"production subjects")
    if len(pilot) != args.expect_pilot:
        raise SystemExit(
            f"expected {args.expect_pilot} pre-SHINE pilot subjects, got {len(pilot)}. The subject "
            f"set defines the ground truth; set --expect-pilot deliberately if it really changed.")
    if len(extra) != args.expect_excluded:
        raise SystemExit(
            f"expected {args.expect_excluded} discarded pre-SHINE production subjects, got "
            f"{len(extra)}. This number GROWS as collection continues, which is expected - but it "
            f"must be an explicit choice, because a GT rebuilt on a different subject set is not "
            f"comparable to the one the decision run was calibrated against. Set "
            f"--expect-excluded deliberately.")

    augmented = list(pilot) + list(extra)
    if not gtc.is_connected(augmented):
        raise SystemExit("the augmented pre-SHINE pair graph is disconnected; MDS needs one "
                         "component")
    print(f"[gt-v6] coverage: pilot-only {gtc.coverage_of(pilot):.1%} -> augmented "
          f"{gtc.coverage_of(augmented):.1%}")

    args.gt_dir.mkdir(parents=True, exist_ok=True)
    decision: Dict[str, object] = {"ndim": int(args.ndim), "method": args.method,
                                   "n_pilot": len(pilot), "n_excluded_prod": len(extra),
                                   "tolerance_sems": args.tolerance}

    accepted = True
    if args.skip_comparison:
        print("[gt-v6] --skip-comparison: building the augmented GT without scoring it")
        decision["comparison"] = "skipped"
    else:
        print(f"\n[gt-v6] scoring the pilot-only set at d={args.ndim} ...", flush=True)
        before = _diagnostics(pilot, args.ndim, n_draws=args.n_draws, n_jobs=args.n_jobs,
                              seed=args.seed, images=images)
        print(f"\n[gt-v6] scoring the augmented set at d={args.ndim} ...", flush=True)
        after = _diagnostics(augmented, args.ndim, n_draws=args.n_draws, n_jobs=args.n_jobs,
                             seed=args.seed, images=images)
        table = pd.DataFrame([{"set": "pilot_only", **before}, {"set": "augmented", **after}])
        table.to_csv(args.gt_dir / "gt_v6_comparison.csv", index=False)
        print("\n--- rebuild comparison ---")
        print(table.to_string(index=False))

        # One-sided, for the reason in the module docstring: the augmented set has larger halves,
        # so it is favoured mechanically and a small gain proves nothing. Only a real loss blocks.
        margin = args.tolerance * max(before["split_half_spearman_sem"], 1e-9)
        drop = before["split_half_spearman"] - after["split_half_spearman"]
        accepted = bool(drop <= margin)
        decision.update({"comparison": "run", "before": before, "after": after,
                         "split_half_drop": float(drop), "margin": float(margin)})
        print(f"\n[gt-v6] split-half Spearman {before['split_half_spearman']:.4f} -> "
              f"{after['split_half_spearman']:.4f} (drop {drop:+.4f}, margin {margin:.4f})")
        print(f"[gt-v6] noise ceiling {before['mean_noise_ceiling_full']:.4f} -> "
              f"{after['mean_noise_ceiling_full']:.4f}")
        if not accepted:
            print("\n[gt-v6] REJECTED: the augmented set agrees with itself WORSE than the "
                  "pilot alone, despite having larger halves. The eight add more noise than "
                  "signal. Keeping the pilot-only ground truth; the decision run should use v5's.")

    subjects = augmented if accepted else pilot
    tag = "v6" if accepted else "pilot_only"
    coords, info = gtc.build_gt(subjects, args.ndim, method=args.method)
    gt_file = f"gt_pre_shine_{tag}_d{args.ndim}.npy"
    np.save(args.gt_dir / gt_file, coords)
    decision.update({"gt_file": gt_file, "n_subjects_used": len(subjects), "gt_info": info,
                     "accepted": accepted})
    (args.gt_dir / "gt_v6_decision.json").write_text(json.dumps(decision, indent=2, default=str))
    print(f"\n[gt-v6] wrote {gt_file}  {coords.shape}  (from {len(subjects)} subjects)")
    print(f"[gt-v6] pass GT_FILE={gt_file} to run_decision_v6.sh")
    # Rejection is a legitimate, informative outcome and the pilot-only GT is still written, so this
    # exits 0 either way. gt_v6_decision.json carries `accepted`.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
