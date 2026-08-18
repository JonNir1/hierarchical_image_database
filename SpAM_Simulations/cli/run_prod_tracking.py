"""Score the live study against the simulation's predictions, and write the report's prod tables.

Runs locally in seconds: no MDS, no R, no store. Everything it computes is either a property of
participants (reliability, agreement, canvas usage, screening outcomes) or a count over what has
been observed (pair coverage, connectivity, allocation balance). It builds no embedding and takes
no mean rating per semantic level, because collection is still running and those are the registered
analysis rather than a design diagnostic.

Usage (from the repo root)::

    python -m SpAM_Simulations.cli.run_prod_tracking \\
        --data-dir data \\
        --manifest SpAM_Task/stimuli_manifest.json \\
        --out SpAM_Simulations/sim_results/v5/prod

The output directory sits inside the gitignored part of ``sim_results/``, and must stay there: the
per-subject reliability table is participant data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_OUT = Path("SpAM_Simulations/sim_results/v5/prod")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="flat directory of session CSVs")
    p.add_argument("--manifest", required=True, type=Path, help="stimuli_manifest.json")
    p.add_argument("--config", default="SpAM_Task/task_config.json",
                   help="task config the screening thresholds are read from")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    from SpAM_Simulations.empirical import prod_tracking, screening_audit
    from SpAM_Simulations.empirical.subjects import load_manifest, load_prod_subjects

    images, _ = load_manifest(str(args.manifest))
    subjects = load_prod_subjects(args.data_dir, str(args.manifest))
    if not subjects:
        raise SystemExit(
            f"no production subjects found under {args.data_dir}.\n"
            f"  The parser derives the cohort from each file's deployment_mode; if the live "
            f"sessions are there, check that column rather than the filenames.")

    per_variant = {}
    for s in subjects:
        per_variant[s.shine_variant] = per_variant.get(s.shine_variant, 0) + 1
    print(f"retained production subjects: {len(subjects)} "
          f"({', '.join(f'{k or 'unknown'}={v}' for k, v in sorted(per_variant.items()))})")
    thresholds = screening_audit.load_thresholds(args.config)
    print(f"deployed gate: min_reliability={thresholds['min_reliability']}, "
          f"median_reliability={thresholds['median_reliability']}, "
          f"move_ratio_max_fail_rate={thresholds['move_ratio_max_fail_rate']}, "
          f"distance_sd_max_fail_rate={thresholds['distance_sd_max_fail_rate']}")

    tables = prod_tracking.track(subjects, images, args.data_dir, str(args.manifest))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
        print(f"  {name:<28} {len(frame):>5} rows")

    summary = tables.get("prod_screening_summary")
    if summary is not None and len(summary):
        print("\n--- screening outcomes ---")
        cols = [c for c in ("group", "n_candidates", "clean_pass", "early_fail", "false_positive",
                            "pass_rate", "failed_reliability", "failed_move_ratio",
                            "failed_distance_sd") if c in summary.columns]
        print(summary[cols].to_string(index=False))

    coverage = tables.get("prod_coverage")
    if coverage is not None and len(coverage):
        print("\n--- coverage and connectivity, per cohort ---")
        print(coverage[[c for c in ("cohort", "n_subjects", "pair_coverage", "n_components",
                                    "connected") if c in coverage.columns]].to_string(index=False))

    print(f"\nwrote {len(tables)} tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
