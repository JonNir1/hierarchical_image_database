"""Recompute the four store-derived metric tables locally, after a grouping fix.

These four were computed on EC2 while the store still contained duplicate ``(configuration, rep)``
fits - 449 of them, appended by a resume that did not recognise completed work. Every one of these
metrics compares reps pairwise, so a duplicate entered as a self-comparison of an identical cohort
and biased the result upward, concentrated in the 48 affected groups (all at ``num_subjects=30``,
``perspective_dispersion=0.1``).

``pipeline._grouped_successful`` now drops the duplicates, so re-running these against the same
store is enough to correct them. Nothing here needs cohorts, R, or the network: the store plus the
ground-truth coordinates are the whole input.

Usage (from the repo root)::

    python -m SpAM_Simulations.cli.recompute_store_tables \\
        --run SpAM_Simulations/sim_results/design-comparison-v5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist

from SpAM_Simulations.core import pipeline
from SpAM_Simulations.core.storage import ResultStore

# name -> (csv, needs the ground truth?)
TABLES = {
    "embedding_stability": ("embedding_stability.csv", False),
    "embedding_generalizability": ("embedding_generalizability.csv", False),
    "topk_jaccard": ("topk_jaccard.csv", False),
    "recovery_vs_gt": ("recovery_vs_gt.csv", True),
}


def _resolve_gt(run: Path, override: Optional[Path]) -> Path:
    """Find the ground-truth coordinates, looking in ``gt/`` as well as the run root.

    Stage 1 writes the coordinates into ``<run>/gt/``, which is where they belong; earlier ad-hoc
    downloads dropped a copy at the run root. Both are searched so neither layout breaks this.
    """
    if override is not None:
        return override
    named = ""
    cal = run / "calibration" / "calibration.json"
    if cal.is_file():
        named = str(json.loads(cal.read_text()).get("gt_file", ""))
        for candidate in (run / named, run / "gt" / named):
            if named and candidate.is_file():
                return candidate
    found = sorted(run.glob("*.npy")) + sorted((run / "gt").glob("*.npy"))
    if len(found) == 1:
        return found[0]
    raise SystemExit(
        f"cannot locate the ground truth under {run}. calibration.json names {named!r}, which is "
        f"at neither {run / named} nor {run / 'gt' / named}, and globbing found "
        f"{len(found)} .npy files. Pass --gt explicitly.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="downloaded run directory")
    p.add_argument("--store", type=Path, default=None, help="defaults to <run>/mds_store")
    p.add_argument("--out", type=Path, default=None, help="defaults to <run>/out")
    p.add_argument("--gt", type=Path, default=None, help="ground-truth .npy for recovery_vs_gt")
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated subset of " + ",".join(TABLES))
    args = p.parse_args(argv)

    store_path = args.store or (args.run / "mds_store")
    out = args.out or (args.run / "out")
    out.mkdir(parents=True, exist_ok=True)
    store = ResultStore.open(store_path)

    # Reported explicitly: this is the whole reason the tables are being rebuilt, and a run that
    # silently found nothing to drop would mean the fix is not reaching this store.
    grouped, _ = pipeline._grouped_successful(store, None)
    print(f"[store] {len(store)} records -> {grouped.ngroups} groups, "
          f"{int(grouped.size().sum())} fits after de-duplication "
          f"(group sizes: {grouped.size().value_counts().to_dict()})", flush=True)

    wanted = list(TABLES) if args.only is None else [t.strip() for t in args.only.split(",")]
    unknown = [t for t in wanted if t not in TABLES]
    if unknown:
        raise SystemExit(f"unknown table(s) {unknown}; choose from {list(TABLES)}")

    gt_dists = None
    if any(TABLES[t][1] for t in wanted):
        gt_path = _resolve_gt(args.run, args.gt)
        coords = np.load(gt_path)
        gt_dists = pdist(coords)
        print(f"[gt] {coords.shape} from {gt_path.name}", flush=True)

    for name in wanted:
        filename, needs_gt = TABLES[name]
        started = time.perf_counter()
        print(f"\n[{name}] computing ...", flush=True)
        if needs_gt:
            frame = pipeline.compute_recovery_vs_gt(store, gt_dists)
        else:
            frame = getattr(pipeline, {
                "embedding_stability": "compute_embedding_stability",
                "embedding_generalizability": "compute_embedding_generalizability",
                "topk_jaccard": "compute_topk_similar_pair_stability",
            }[name])(store)
        frame.to_csv(out / filename, index=False)
        print(f"[{name}] {len(frame)} rows -> {out / filename} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)

    print("\n[done] every table rebuilt against the de-duplicated grouping", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
