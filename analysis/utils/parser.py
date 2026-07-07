"""
Loader for SpAM session data -- works for both data/pilot and data/prod.

The two differ only in demographics filename (participant_demographics*.csv
vs demographic*.csv, both tried automatically) and in that prod participants
can have more than one session file (a reconnect after an abandoned
attempt); when that happens, the session with the most trial/catch rows is
used, so callers never have to know which directory they're pointed at.

Usage (from repo root):
    from analysis.utils.parser import load_pilot_data
    data = load_pilot_data("data/pilot")   # or "data/prod"
    df_trials = data["trials"]        # one row per experimental trial, completed subjects only
    df_status = data["status"]        # one row per participant with completion_status
    df_catch  = data["catch_trials"]  # one row per catch trial, completed subjects only
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

# analysis/utils/parser.py → analysis/utils/ → analysis/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# pilot demographics files are "participant_demographics*.csv"; prod's is "demographic*.csv"
_DEMOGRAPHICS_GLOBS = ["participant_demographics*.csv", "demographic*.csv"]
_EXPERIMENTAL_TRIAL_TYPE_RE = r"^trial_\d+$"
_CATCH_TRIAL_TYPE_RE = r"^catch_\d+$"

_DEMOGRAPHICS_RENAME = {
    "Submission id": "submission_id",
    "Participant id": "participant_id",
    "Status": "prolific_status",
    "Time taken": "prolific_duration_s",
    "Age": "age",
    "Sex": "sex",
    "Ethnicity simplified": "ethnicity",
    "Country of birth": "country_of_birth",
    "Country of residence": "country_of_residence",
    "Nationality": "nationality",
    "Language": "language",
    "Student status": "student_status",
    "Employment status": "employment_status",
}

# Prolific-internal columns with no analysis value
_DEMOGRAPHICS_DROP = {
    "Custom study tncs accepted at",
    "Started at",
    "Completed at",
    "Reviewed at",
    "Archived at",
    "Completion code",
    "Total approvals",
}

_SESSION_KEEP = [
    "trial_type",
    "rt",
    "qc_flag",
    "shine_variant",
    "task_version",
    "deployment_mode",
    "moves",
    "pairwise_distances",
    "final_locations",
    "init_locations",
    "is_trial_repeat",
    "repeat_of_trial_number",
]

_CATCH_SESSION_KEEP = [
    "trial_type",
    "rt",
    "qc_flag",
    "shine_variant",
    "task_version",
    "deployment_mode",
    "moves",
    "pairwise_distances",
    "final_locations",
    "catch_trial_target_location",
    "centroid_x",
    "centroid_y",
    "cluster_mean_distance",
]

_PROLIFIC_SENTINEL = "CONSENT_REVOKED"
_EXPIRED_SENTINEL = "DATA_EXPIRED"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pilot_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load SpAM pilot data from *data_dir*.

    Returns a dict with three keys:
      "trials"       -- one row per experimental trial per completed participant
      "status"       -- one row per participant with a completion_status column
      "catch_trials" -- one row per catch trial per completed participant (QC validation)

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist.
    NotADirectoryError
        If *data_dir* exists but is not a directory.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = _REPO_ROOT / data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {data_dir}")

    demo_paths = sorted({p for pattern in _DEMOGRAPHICS_GLOBS for p in data_dir.glob(pattern)})
    if not demo_paths:
        raise FileNotFoundError(
            f"No demographics file found in: {data_dir} (expected one of {_DEMOGRAPHICS_GLOBS})"
        )

    df_demo = pd.concat([_load_demographics(p) for p in demo_paths], ignore_index=True)
    dup_pids = df_demo["participant_id"][df_demo["participant_id"].duplicated()]
    if not dup_pids.empty:
        warnings.warn(
            f"Duplicate participant_id(s) across demographics files: {sorted(set(dup_pids))}",
            UserWarning,
            stacklevel=2,
        )

    session_files = _index_session_files(data_dir)

    status_rows: list[dict] = []
    trial_rows: list[pd.DataFrame] = []
    catch_rows: list[pd.DataFrame] = []

    for _, participant in df_demo.iterrows():
        pid = participant["participant_id"]
        prolific_status = participant["prolific_status"]

        if prolific_status in {"RETURNED", "REJECTED"}:
            warnings.warn(
                f"Participant {pid}: revoked consent (status={prolific_status}), excluded from trials.",
                UserWarning,
                stacklevel=2,
            )
            status_rows.append({**participant.to_dict(), "completion_status": "revoked_consent"})
            continue

        session_paths = session_files.get(pid, [])
        if not session_paths:
            warnings.warn(
                f"Participant {pid}: APPROVED but no session file found (erroneous completion), excluded from trials.",
                UserWarning,
                stacklevel=2,
            )
            status_rows.append({**participant.to_dict(), "completion_status": "erroneous_completion"})
            continue

        # A participant can have more than one session file (e.g. a reconnect after
        # an abandoned attempt, seen in prod); use whichever has the most trial/catch
        # rows. A no-op when there's only one, which is the common pilot case.
        session_path = max(session_paths, key=_session_row_count) if len(session_paths) > 1 else session_paths[0]

        df_trials = _load_session_trials(session_path, participant)
        trial_rows.append(df_trials)
        catch_rows.append(_load_session_catch_trials(session_path, participant))
        status_rows.append({**participant.to_dict(), "completion_status": "completed"})

    df_status = pd.DataFrame(status_rows).reset_index(drop=True)

    if trial_rows:
        df_trials_all = pd.concat(trial_rows, ignore_index=True)
        # bool columns can upcast to object after concat; re-cast explicitly
        if "qc_flag" in df_trials_all.columns:
            df_trials_all["qc_flag"] = df_trials_all["qc_flag"].astype(bool)
        if "is_trial_repeat" in df_trials_all.columns:
            # absent entirely for v1/v2 sessions (no trial-repeat mechanism) -> False
            df_trials_all["is_trial_repeat"] = df_trials_all["is_trial_repeat"].fillna(False).astype(bool)
        if "repeat_of_trial_number" in df_trials_all.columns:
            df_trials_all["repeat_of_trial_number"] = pd.to_numeric(
                df_trials_all["repeat_of_trial_number"], errors="coerce"
            )
    else:
        df_trials_all = pd.DataFrame()

    if catch_rows:
        df_catch_all = pd.concat(catch_rows, ignore_index=True)
        if "qc_flag" in df_catch_all.columns:
            df_catch_all["qc_flag"] = df_catch_all["qc_flag"].astype(bool)
    else:
        df_catch_all = pd.DataFrame()

    return {"trials": df_trials_all, "status": df_status, "catch_trials": df_catch_all}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in _DEMOGRAPHICS_DROP if c in df.columns])
    df = df.rename(columns={k: v for k, v in _DEMOGRAPHICS_RENAME.items() if k in df.columns})
    # Mask Prolific sentinel strings as NA
    df = df.replace({_PROLIFIC_SENTINEL: pd.NA, _EXPIRED_SENTINEL: pd.NA})
    # Cast numeric where possible
    for col in ("age", "prolific_duration_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _index_session_files(data_dir: Path) -> dict[str, list[Path]]:
    """Return {participant_id: [session paths]} for all session CSVs in data_dir.

    Participant IDs are read from the participant_id column inside each CSV
    because the filenames use literal placeholder tokens rather than actual IDs.
    A participant can have more than one session file (e.g. a reconnect after
    an abandoned attempt), so every match is kept, not just the last one seen.
    """
    index: dict[str, list[Path]] = {}
    demo_paths = {p for pattern in _DEMOGRAPHICS_GLOBS for p in data_dir.glob(pattern)}
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path in demo_paths:
            continue
        try:
            # Read only the first data row to extract participant_id cheaply
            df_head = pd.read_csv(csv_path, usecols=["participant_id"], nrows=1)
            pid = str(df_head["participant_id"].iloc[0])
            index.setdefault(pid, []).append(csv_path)
        except (KeyError, IndexError, pd.errors.ParserError, ValueError):
            continue
    return index


def _session_row_count(session_path: Path) -> int:
    """Number of main-trial + catch-trial rows in a session CSV, used to pick the most
    complete session file when a participant has more than one."""
    trial_types = pd.read_csv(session_path, usecols=["trial_type"])["trial_type"].astype(str)
    return int(
        trial_types.str.match(_EXPERIMENTAL_TRIAL_TYPE_RE).sum()
        + trial_types.str.match(_CATCH_TRIAL_TYPE_RE).sum()
    )


def _load_session_trials(session_path: Path, participant: pd.Series) -> pd.DataFrame:
    df = pd.read_csv(session_path)

    # Read canvas dimensions before filtering (present on every row)
    canvas_w = int(df["sort_area_width"].iloc[0])
    canvas_h = int(df["sort_area_height"].iloc[0])

    main_mask = df["trial_type"].astype(str).str.match(_EXPERIMENTAL_TRIAL_TYPE_RE)
    df = df[main_mask].copy()

    # Derived columns
    df["trial_number"] = df["trial_type"].str.extract(r"trial_(\d+)").astype(int)
    df["n_moves"] = df["moves"].apply(_count_moves).astype(int)
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    # qc_flag / is_trial_repeat are written by JS as lowercase "true"/"false" strings
    for bool_col in ("qc_flag", "is_trial_repeat"):
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].isin([True, "true", "True", 1])
    # repeat_of_trial_number (v3.03+) is written directly in trial_number space
    # (1-based, main trials only) -- no re-indexing needed on read.
    if "repeat_of_trial_number" in df.columns:
        df["repeat_of_trial_number"] = pd.to_numeric(
            df["repeat_of_trial_number"], errors="coerce"
        )

    # Normalise all pixel x/y coordinates to [0, 1] using this session's canvas
    # size, so coordinates are screen-independent. sort_area is not kept.
    for col in ("final_locations", "init_locations", "moves"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda s: _normalise_locations(s, canvas_w, canvas_h)
            )

    # Keep only planned session columns
    keep = [c for c in _SESSION_KEEP + ["trial_number", "n_moves"] if c in df.columns]
    df = df[keep].copy()

    # Broadcast participant demographics + session filename onto every trial row
    df["session_file"] = session_path.stem
    demo_cols = {k: participant[k] for k in participant.index if k not in ("prolific_status",)}
    for col, val in demo_cols.items():
        df[col] = val

    return df.reset_index(drop=True)


def _load_session_catch_trials(session_path: Path, participant: pd.Series) -> pd.DataFrame:
    """Catch trials, kept separate from main trials (different schema: target
    location, centroid, cluster_mean_distance instead of trial_number/n_moves
    progression fields). Used for validating catch_trials.* QC thresholds.
    """
    df = pd.read_csv(session_path)

    canvas_w = int(df["sort_area_width"].iloc[0])
    canvas_h = int(df["sort_area_height"].iloc[0])

    df = df[df["trial_type"].astype(str).str.match(_CATCH_TRIAL_TYPE_RE)].copy()

    df["catch_number"] = df["trial_type"].str.extract(r"catch_(\d+)").astype(int)
    df["n_moves"] = df["moves"].apply(_count_moves).astype(int)
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    if "qc_flag" in df.columns:
        df["qc_flag"] = df["qc_flag"].isin([True, "true", "True", 1])

    for col in ("final_locations", "moves"):
        if col in df.columns:
            df[col] = df[col].apply(lambda s: _normalise_locations(s, canvas_w, canvas_h))

    keep = [c for c in _CATCH_SESSION_KEEP + ["catch_number", "n_moves"] if c in df.columns]
    df = df[keep].copy()

    # Canvas size kept (unlike main trials) -- needed to recompute the
    # location_tolerance check, which is not a stored field.
    df["sort_area_width"] = canvas_w
    df["sort_area_height"] = canvas_h
    df["session_file"] = session_path.stem
    demo_cols = {k: participant[k] for k in participant.index if k not in ("prolific_status",)}
    for col, val in demo_cols.items():
        df[col] = val

    return df.reset_index(drop=True)


def _normalise_locations(locs_json: str, canvas_w: int, canvas_h: int) -> str:
    """Normalise pixel x/y to [0, 1] in a final_locations, init_locations, or moves JSON string."""
    if pd.isna(locs_json) or locs_json == "":
        return locs_json
    try:
        items = json.loads(locs_json)
        for item in items:
            item["x"] = item["x"] / canvas_w
            item["y"] = item["y"] / canvas_h
        return json.dumps(items)
    except (json.JSONDecodeError, TypeError, KeyError):
        return locs_json


def _count_moves(moves_json: str) -> int:
    if pd.isna(moves_json) or moves_json == "":
        return 0
    try:
        return len(json.loads(moves_json))
    except (json.JSONDecodeError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Public parsing helpers (shared across figures.py and analysis notebooks)
# ---------------------------------------------------------------------------


def parse_pairwise_distances(pw_json: str) -> dict[tuple[str, str], float]:
    """Parse one trial's pairwise_distances JSON into {sorted_image_pair: distance}."""
    if pd.isna(pw_json) or pw_json == "":
        return {}
    try:
        items = json.loads(pw_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return {
        tuple(sorted([item["src1"], item["src2"]])): item["distance"]
        for item in items
    }


def _trial_image_set(pw_json: str) -> frozenset[str]:
    """Union of images referenced in a trial's pairwise_distances JSON."""
    images: set[str] = set()
    for a, b in parse_pairwise_distances(pw_json):
        images.add(a)
        images.add(b)
    return frozenset(images)


def validate_trial_repeat_image_sets(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity-check the ``is_trial_repeat`` / ``repeat_of_trial_number`` mechanism: for every
    trial flagged as a verbatim repeat, match it (within the same session) to the trial
    numbered ``repeat_of_trial_number`` and verify both show the same set of images -- the
    invariant ``insertTrialRepeats`` (SpAM_Task/js/trial_generator.js) is meant to guarantee.

    Image sets are derived from each trial's ``pairwise_distances`` column (the union of
    ``src1``/``src2`` across all pairs), not stored directly as a column.

    Returns one row per repeat trial found, with columns: participant_id, session_file,
    trial_number, repeat_of_trial_number, images_match, n_images_original, n_images_repeat.
    Never raises on a mismatch -- inspect the `images_match` column, or use
    `report["images_match"].all()` for a single pass/fail check.
    """
    required = {"participant_id", "session_file", "trial_number", "is_trial_repeat",
                "repeat_of_trial_number", "pairwise_distances"}
    missing = required - set(df_trials.columns)
    if missing:
        raise KeyError(f"validate_trial_repeat_image_sets: df_trials missing column(s): {sorted(missing)}")

    rows = []
    for (pid, session), group in df_trials.groupby(["participant_id", "session_file"]):
        images_by_number = {
            int(row["trial_number"]): _trial_image_set(row["pairwise_distances"])
            for _, row in group.iterrows()
        }
        repeats = group[group["is_trial_repeat"] & group["repeat_of_trial_number"].notna()]
        for _, row in repeats.iterrows():
            trial_num = int(row["trial_number"])
            orig_num  = int(row["repeat_of_trial_number"])
            orig_images = images_by_number.get(orig_num)
            rep_images  = images_by_number.get(trial_num)
            rows.append({
                "participant_id":          pid,
                "session_file":            session,
                "trial_number":            trial_num,
                "repeat_of_trial_number":  orig_num,
                "images_match":            orig_images is not None and orig_images == rep_images,
                "n_images_original":       len(orig_images) if orig_images is not None else 0,
                "n_images_repeat":         len(rep_images)  if rep_images  is not None else 0,
            })

    return pd.DataFrame(rows, columns=[
        "participant_id", "session_file", "trial_number", "repeat_of_trial_number",
        "images_match", "n_images_original", "n_images_repeat",
    ])


# {task_version: (expected main trials per subject, expected repeated trials per subject)}
# v1.0/v2.0 predate the whole-trial-repeat mechanism (frac_trials_repeated); v3.x always
# repeats exactly 3 of its 20 trials (see SpAM_Task/js/trial_generator.js).
_EXPECTED_TRIAL_COUNTS: dict[float, tuple[int, int]] = {
    1.0:  (10, 0),
    2.0:  (10, 0),
    3.0:  (20, 3),
    3.06: (20, 3),
}


def validate_trial_counts(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    Check that every session has the expected number of main trials and repeated
    trials for its task_version (see ``_EXPECTED_TRIAL_COUNTS``).

    Returns one row per (participant_id, session_file), with columns:
    participant_id, session_file, task_version, n_trials, n_repeats,
    expected_n_trials, expected_n_repeats, trials_match, repeats_match. A
    task_version absent from ``_EXPECTED_TRIAL_COUNTS`` gets NaN expected
    values and both ``*_match`` columns False. Never raises on a mismatch --
    inspect the ``trials_match``/``repeats_match`` columns, or use
    ``report[["trials_match", "repeats_match"]].all(axis=None)`` for a single
    pass/fail check.
    """
    required = {"participant_id", "session_file", "task_version", "trial_number", "is_trial_repeat"}
    missing = required - set(df_trials.columns)
    if missing:
        raise KeyError(f"validate_trial_counts: df_trials missing column(s): {sorted(missing)}")

    rows = []
    for (pid, session, task_version), group in df_trials.groupby(
        ["participant_id", "session_file", "task_version"]
    ):
        n_trials = int(group["trial_number"].nunique())
        n_repeats = int(group["is_trial_repeat"].astype(bool).sum())
        expected = _EXPECTED_TRIAL_COUNTS.get(task_version)
        expected_n_trials, expected_n_repeats = expected if expected is not None else (None, None)
        rows.append({
            "participant_id":      pid,
            "session_file":        session,
            "task_version":        task_version,
            "n_trials":            n_trials,
            "n_repeats":           n_repeats,
            "expected_n_trials":   expected_n_trials,
            "expected_n_repeats":  expected_n_repeats,
            "trials_match":        expected_n_trials is not None and n_trials == expected_n_trials,
            "repeats_match":       expected_n_repeats is not None and n_repeats == expected_n_repeats,
        })

    return pd.DataFrame(rows, columns=[
        "participant_id", "session_file", "task_version", "n_trials", "n_repeats",
        "expected_n_trials", "expected_n_repeats", "trials_match", "repeats_match",
    ])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Validate trial-repeat image sets (is_trial_repeat / "
                    "repeat_of_trial_number) and per-version trial/repeat counts.",
    )
    ap.add_argument("data_dir", nargs="?", default="data/pilot",
                     help="Pilot data directory (default: data/pilot)")
    args = ap.parse_args()

    df_trials = load_pilot_data(args.data_dir)["trials"]
    failed = False

    image_report = validate_trial_repeat_image_sets(df_trials)
    if image_report.empty:
        print("No trial repeats found.")
    else:
        print(image_report.to_string(index=False))
        n_bad = int((~image_report["images_match"]).sum())
        print(f"\n{len(image_report)} repeat trial(s) checked, {n_bad} mismatch(es).")
        failed = failed or bool(n_bad)

    print()
    count_report = validate_trial_counts(df_trials)
    bad_counts = count_report[~(count_report["trials_match"] & count_report["repeats_match"])]
    if bad_counts.empty:
        print(f"All {len(count_report)} session(s) have the expected trial/repeat counts for their task_version.")
    else:
        print(bad_counts.to_string(index=False))
        print(f"\n{len(bad_counts)} of {len(count_report)} session(s) have unexpected trial/repeat counts.")
        failed = True

    if failed:
        raise SystemExit(1)
