"""
One-off migration check: compares analysis.utils.parser.load_pilot_data (run
separately against a pilot dir and a prod dir) against analysis.utils.parser_v2.load_data
(run against a single flattened dir), to confirm the new parser reproduces the old
parser's results before data/pilot and data/prod are deleted.

Usage (from repo root):
    python -m analysis.utils.verify_parser_v2 data --pilot-dir data/pilot --prod-dir data/prod

Known, expected exceptions (pass via --exclude-pids, comma-separated, not failures):
  - A participant whose only session file has JSON-corrupted `pairwise_distances`
    that also corrupts a session-level field (deployment_mode/shine_variant/etc.)
    makes the new parser raise loudly, by design (see parser_v2's constancy checks).
    Remove that file from the flat dir before running this script, and exclude the
    same participant_id here so the old-side counts still line up.
  - A participant with a genuine tie between two full, differently-content session
    files: the old parser's max() keeps an arbitrary one, the new parser deliberately
    keeps the most recent (see parser_v2._resolve_session_file). Exclude these too.
"""
from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from analysis.utils.parser import load_pilot_data
from analysis.utils.parser_v2 import load_data

_DEMO_COLS = [
    "age", "sex", "ethnicity", "country_of_birth", "country_of_residence",
    "nationality", "language", "student_status", "employment_status",
    "submission_id", "prolific_duration_s",
]


def run(flat_dir: str, pilot_dir: str, prod_dir: str, exclude_pids: set[str]) -> bool:
    """Returns True if every check passed."""
    failures: list[str] = []

    print("=" * 70)
    print(f"Loading old parser ({pilot_dir}, {prod_dir})")
    print("=" * 70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_pilot = load_pilot_data(pilot_dir)
        old_prod = load_pilot_data(prod_dir)

    print("=" * 70)
    print(f"Loading new parser ({flat_dir})")
    print("=" * 70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = load_data(flat_dir)
    df_p, df_t = new["participants"], new["trials"]

    # --- 1. Demographics bit-exact for cohort == "pilot" ---
    print("\n=== 1. Demographics bit-exact (cohort=='pilot') ===")
    old_status = old_pilot["status"]
    old_status = old_status[old_status["completion_status"] != "revoked_consent"]
    new_pilot = df_p[df_p["cohort"] == "pilot"]
    merged = new_pilot.merge(old_status, on="participant_id", suffixes=("_new", "_old"), how="inner")
    print(f"matched {len(merged)} participants (new pilot cohort N={len(new_pilot)}, old completed pilot N={len(old_status)})")
    for col in _DEMO_COLS:
        a, b = merged[f"{col}_new"], merged[f"{col}_old"]
        mism = ~(a.astype(str) == b.astype(str)) & ~(a.isna() & b.isna())
        if mism.any():
            failures.append(f"demo field '{col}': {mism.sum()} mismatches")
            print(f"  MISMATCH in {col}: {mism.sum()} rows")
        else:
            print(f"  {col}: OK")

    coincide = (new_pilot["task_version"] < 4).all()
    print(f"cohort=='pilot' coincides with task_version<4 for all rows: {coincide}")
    if not coincide:
        failures.append("cohort=='pilot' does not coincide with task_version<4")

    # --- 2. Trial-level equivalence ---
    print("\n=== 2. Trial-level equivalence (rt, num_moves, qc_flag, locations) ===")
    old_trials_union = pd.concat([
        old_pilot["trials"].assign(is_catch=False),
        old_pilot["catch_trials"].assign(is_catch=True),
        old_prod["trials"].assign(is_catch=False),
        old_prod["catch_trials"].assign(is_catch=True),
    ], ignore_index=True)
    old_trials_union = old_trials_union[~old_trials_union["participant_id"].isin(exclude_pids)]
    df_t_check = df_t[~df_t["participant_id"].isin(exclude_pids)]

    merged_trials = old_trials_union.merge(
        df_t_check, on=["participant_id", "pairwise_distances"], suffixes=("_old", "_new"), how="inner"
    )
    print(f"matched {len(merged_trials)} trial rows via (participant_id, pairwise_distances)")
    print(f"old total (excl. {sorted(exclude_pids)}): {len(old_trials_union)}  new total: {len(df_t_check)}")

    checks = {
        "rt": (merged_trials["rt_old"].round(3) != merged_trials["rt_new"].round(3)).sum(),
        "num_moves": (merged_trials["n_moves"] != merged_trials["num_moves"]).sum(),
        "qc_flag": (merged_trials["qc_flag_old"].astype(bool) != merged_trials["qc_flag_new"].astype(bool)).sum(),
        "final_locations": (merged_trials["final_locations_old"] != merged_trials["final_locations_new"]).sum(),
    }
    for name, n in checks.items():
        print(f"  {name} mismatches: {n}")
        if n:
            failures.append(f"trial field '{name}': {n} mismatches")

    unmatched_old = len(old_trials_union) - len(merged_trials)
    print(f"  old rows with no match in new: {unmatched_old}")
    if unmatched_old:
        failures.append(f"{unmatched_old} old trial rows had no match in new trials (by pairwise_distances)")

    # --- 3. Row-count reconciliation ---
    print("\n=== 3. Row-count reconciliation ===")
    old_counts = old_trials_union.groupby("participant_id").size()
    new_counts = df_t_check.groupby("participant_id").size()
    common_pids = set(old_counts.index) & set(new_counts.index)
    mismatched = [pid for pid in common_pids if old_counts[pid] != new_counts[pid]]
    print(f"participants compared: {len(common_pids)}, mismatched row counts: {len(mismatched)}")
    if mismatched:
        for pid in mismatched[:10]:
            print(f"  {pid}: old={old_counts[pid]} new={new_counts[pid]}")
        failures.append(f"{len(mismatched)} participants have mismatched row counts")

    # --- 4. Status distribution reconciliation ---
    print("\n=== 4. Status distribution reconciliation ===")
    print("new status counts:")
    print(df_p["status"].value_counts())
    print("\nold completion_status counts (pilot + prod):")
    old_combined = pd.concat([old_pilot["status"], old_prod["status"]], ignore_index=True)
    print(old_combined["completion_status"].value_counts())
    old_screening = pd.concat([old_pilot["screening"], old_prod["screening"]], ignore_index=True)
    old_screened_out = old_screening[old_screening.get("pass") == False] if not old_screening.empty else old_screening
    print(f"old screening pass==False count: {len(old_screened_out)}")

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print(" -", f)
        return False
    print("ALL CHECKS PASSED")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("flat_dir", nargs="?", default="data")
    ap.add_argument("--pilot-dir", default="data/pilot")
    ap.add_argument("--prod-dir", default="data/prod")
    ap.add_argument("--exclude-pids", default="", help="Comma-separated participant_ids to exclude (known exceptions)")
    args = ap.parse_args()

    exclude = {p for p in args.exclude_pids.split(",") if p}
    ok = run(args.flat_dir, args.pilot_dir, args.prod_dir, exclude)
    if not ok:
        sys.exit(1)
