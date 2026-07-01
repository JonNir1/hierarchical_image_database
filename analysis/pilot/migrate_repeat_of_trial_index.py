"""
One-time migration for pilot session CSVs collected before v3.04 (commit 3bd5ae5):
rename ``repeat_of_trial_index`` (0-based position in the full trial sequence, main +
catch interleaved) to ``repeat_of_trial_number`` (1-based, matching ``trial_type``'s
``trial_N`` numbering), recomputing every value into the new space.

Only ``repeat_of_trial_index`` is touched -- ``trial_index`` itself is left in place,
since it remains a valid standalone column (0-based full-sequence position).

Usage (from repo root):
    python analysis/pilot/migrate_repeat_of_trial_index.py data/pilot          # dry run
    python analysis/pilot/migrate_repeat_of_trial_index.py data/pilot --apply  # writes changes

Each migrated file is backed up alongside itself as ``<name>.csv.bak`` before being
overwritten. Files edited with the standard library ``csv`` module (not pandas) so no
column other than the renamed one is reformatted.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

_TRIAL_NUMBER_RE = re.compile(r"^trial_(\d+)$")

_DEMOGRAPHICS_PREFIX = "participant_demographics"


def _trial_number_by_full_index(rows: list[dict]) -> dict[int, int]:
    """Map each main trial row's `trial_index` (0-based, full sequence) to its
    `trial_number` (1-based, extracted from `trial_type`)."""
    mapping: dict[int, int] = {}
    for row in rows:
        m = _TRIAL_NUMBER_RE.match(row.get("trial_type", "") or "")
        if m and row.get("trial_index", "") not in ("", None):
            mapping[int(row["trial_index"])] = int(m.group(1))
    return mapping


def migrate_file(path: Path, apply: bool) -> str:
    """Migrate one session CSV. Returns a one-line human-readable status message."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "repeat_of_trial_index" not in fieldnames:
        return "skip (no repeat_of_trial_index column)"
    if "repeat_of_trial_number" in fieldnames:
        return "skip (already migrated)"

    index_to_number = _trial_number_by_full_index(rows)

    n_resolved = 0
    n_unresolved = 0
    for row in rows:
        old_val = row.pop("repeat_of_trial_index")
        if old_val in ("", "null", None):
            new_val = ""
        else:
            resolved = index_to_number.get(int(old_val))
            if resolved is None:
                new_val = ""
                n_unresolved += 1
            else:
                new_val = str(resolved)
                n_resolved += 1
        row["repeat_of_trial_number"] = new_val

    if n_unresolved:
        return (f"ABORTED ({n_unresolved} repeat row(s) could not be resolved to a "
                f"trial_number -- inspect the file manually before migrating)")

    new_fieldnames = [
        "repeat_of_trial_number" if f == "repeat_of_trial_index" else f
        for f in fieldnames
    ]

    if not apply:
        return f"would migrate ({n_resolved} repeat row(s) resolved)"

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return f"migrated ({n_resolved} repeat row(s) resolved; backup at {backup.name})"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("data_dir", help="Directory of pilot session CSVs (e.g. data/pilot)")
    ap.add_argument("--apply", action="store_true", help="Write changes to disk (default: dry run)")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"Not a directory: {data_dir}")

    csv_paths = sorted(
        p for p in data_dir.glob("*.csv")
        if not p.stem.startswith(_DEMOGRAPHICS_PREFIX)
    )
    if not csv_paths:
        print(f"No session CSVs found in {data_dir}")
        return

    n_touched = 0
    for path in csv_paths:
        status = migrate_file(path, args.apply)
        print(f"{path.name}: {status}")
        if status.startswith(("migrated", "would migrate")):
            n_touched += 1

    if not args.apply:
        print(f"\n{n_touched} file(s) would be migrated. Re-run with --apply to write changes.")
    else:
        print(f"\n{n_touched} file(s) migrated.")


if __name__ == "__main__":
    main()
