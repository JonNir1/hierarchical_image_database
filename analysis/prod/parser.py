"""
Loader for SpAM prod data.

Mirrors analysis/utils/parser.py's load_pilot_data, adapted for two
differences in the prod dataset:
  - the demographics file is named "demographic.csv" (singular, no
    "participant_" prefix), not "participant_demographics*.csv";
  - a completed Prolific submission does not guarantee a full session --
    some prod sessions stop partway through (abandoned/timed out), so a
    session must have all 20 main trials + 4 catch trials to count as
    "completed" here.

TODO(analysis): analysis/pilot/show_trials.py and analysis/prod/show_trials.py
are near-duplicates of each other, and this loader duplicates most of
analysis/utils/parser.py's load_pilot_data. Once prod data collection settles,
unify both loaders (data_dir + demographics-glob + completeness-predicate as
parameters) and merge the two show_trials.py scripts into one.

Usage (from repo root):
    from analysis.prod.parser import load_prod_data
    data = load_prod_data("data/prod")
    df_trials = data["trials"]        # one row per experimental trial, completed sessions only
    df_status = data["status"]        # one row per participant with completion_status
    df_catch  = data["catch_trials"]  # one row per catch trial, completed sessions only
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from analysis.utils.parser import _count_moves, _load_demographics, _normalise_locations

# analysis/prod/parser.py → analysis/prod/ → analysis/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEMOGRAPHICS_GLOB = "demographic*.csv"
_EXPERIMENTAL_TRIAL_TYPE_RE = r"^trial_\d+$"
_CATCH_TRIAL_TYPE_RE = r"^catch_\d+$"
_N_MAIN_TRIALS = 20
_N_CATCH_TRIALS = 4

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_prod_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load SpAM prod data from *data_dir*.

    Returns a dict with three keys:
      "trials"       -- one row per experimental trial per completed session
      "status"       -- one row per participant with a completion_status column
      "catch_trials" -- one row per catch trial per completed session (QC validation)

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

    demo_paths = sorted(data_dir.glob(_DEMOGRAPHICS_GLOB))
    if not demo_paths:
        raise FileNotFoundError(
            f"No demographics file found in: {data_dir} (expected {_DEMOGRAPHICS_GLOB})"
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

        # A participant can have multiple session files (reconnects/retries after an
        # abandoned attempt); use whichever one is actually complete, if any.
        session_path = next((p for p in session_paths if _is_session_complete(p)), None)
        if session_path is None:
            warnings.warn(
                f"Participant {pid}: {len(session_paths)} session file(s) found but none complete "
                f"(< {_N_MAIN_TRIALS} main trials + {_N_CATCH_TRIALS} catch trials), excluded from trials.",
                UserWarning,
                stacklevel=2,
            )
            status_rows.append({**participant.to_dict(), "completion_status": "incomplete_session"})
            continue

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


def _index_session_files(data_dir: Path) -> dict[str, list[Path]]:
    """Return {participant_id: [session paths]} for all session CSVs in data_dir.

    Participant IDs are read from the participant_id column inside each CSV
    because the filenames are just Pavlovia session timestamps, not IDs. A
    participant can have more than one session file (e.g. a reconnect after
    an abandoned attempt), so every match is kept, not just the last one seen.
    """
    index: dict[str, list[Path]] = {}
    demo_paths = {p.resolve() for p in data_dir.glob(_DEMOGRAPHICS_GLOB)}
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path.resolve() in demo_paths:
            continue
        try:
            df_head = pd.read_csv(csv_path, usecols=["participant_id"], nrows=1)
            pid = str(df_head["participant_id"].iloc[0])
            index.setdefault(pid, []).append(csv_path)
        except (KeyError, IndexError, pd.errors.ParserError, ValueError):
            continue
    return index


def _is_session_complete(session_path: Path) -> bool:
    """A prod session is complete iff it has all 20 main trials and all 4 catch trials
    (a completed Prolific submission doesn't guarantee the participant didn't abandon
    the task partway through)."""
    df = pd.read_csv(session_path, usecols=["trial_type"])
    trial_types = df["trial_type"].astype(str)

    main_numbers, catch_numbers = set(), set()
    for tt in trial_types:
        if m := re.fullmatch(r"trial_(\d+)", tt):
            main_numbers.add(int(m.group(1)))
        elif m := re.fullmatch(r"catch_(\d+)", tt):
            catch_numbers.add(int(m.group(1)))

    return main_numbers == set(range(1, _N_MAIN_TRIALS + 1)) and catch_numbers == set(range(1, _N_CATCH_TRIALS + 1))


def _load_session_trials(session_path: Path, participant: pd.Series) -> pd.DataFrame:
    df = pd.read_csv(session_path)

    canvas_w = int(df["sort_area_width"].iloc[0])
    canvas_h = int(df["sort_area_height"].iloc[0])

    main_mask = df["trial_type"].astype(str).str.match(_EXPERIMENTAL_TRIAL_TYPE_RE)
    df = df[main_mask].copy()

    df["trial_number"] = df["trial_type"].str.extract(r"trial_(\d+)").astype(int)
    df["n_moves"] = df["moves"].apply(_count_moves).astype(int)
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    for bool_col in ("qc_flag", "is_trial_repeat"):
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].isin([True, "true", "True", 1])
    if "repeat_of_trial_number" in df.columns:
        df["repeat_of_trial_number"] = pd.to_numeric(
            df["repeat_of_trial_number"], errors="coerce"
        )

    for col in ("final_locations", "init_locations", "moves"):
        if col in df.columns:
            df[col] = df[col].apply(lambda s: _normalise_locations(s, canvas_w, canvas_h))

    keep = [c for c in _SESSION_KEEP + ["trial_number", "n_moves"] if c in df.columns]
    df = df[keep].copy()

    df["session_file"] = session_path.stem
    demo_cols = {k: participant[k] for k in participant.index if k not in ("prolific_status",)}
    for col, val in demo_cols.items():
        df[col] = val

    return df.reset_index(drop=True)


def _load_session_catch_trials(session_path: Path, participant: pd.Series) -> pd.DataFrame:
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

    df["sort_area_width"] = canvas_w
    df["sort_area_height"] = canvas_h
    df["session_file"] = session_path.stem
    demo_cols = {k: participant[k] for k in participant.index if k not in ("prolific_status",)}
    for col, val in demo_cols.items():
        df[col] = val

    return df.reset_index(drop=True)
