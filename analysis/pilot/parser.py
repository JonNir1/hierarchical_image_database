"""
Loader for SpAM pilot data.

Usage (from repo root):
    from analysis.pilot.parser import load_pilot_data
    data = load_pilot_data("data/pilot")
    df_trials = data["trials"]   # one row per experimental trial, completed subjects only
    df_status = data["status"]   # one row per participant with completion_status
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

# analysis/pilot/parser.py → analysis/pilot/ → analysis/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEMOGRAPHICS_FILE = "participant_demographics.csv"
_EXPERIMENTAL_TRIAL_TYPES = {f"trial_{i}" for i in range(1, 11)}

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
]

_PROLIFIC_SENTINEL = "CONSENT_REVOKED"
_EXPIRED_SENTINEL = "DATA_EXPIRED"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pilot_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load SpAM pilot data from *data_dir*.

    Returns a dict with two keys:
      "trials"  -- one row per experimental trial per completed participant
      "status"  -- one row per participant with a completion_status column

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

    demo_path = data_dir / _DEMOGRAPHICS_FILE
    if not demo_path.exists():
        raise FileNotFoundError(f"Demographics file not found: {demo_path}")

    df_demo = _load_demographics(demo_path)
    session_files = _index_session_files(data_dir)

    status_rows: list[dict] = []
    trial_rows: list[pd.DataFrame] = []

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

        session_path = session_files.get(pid)
        if session_path is None:
            warnings.warn(
                f"Participant {pid}: APPROVED but no session file found (erroneous completion), excluded from trials.",
                UserWarning,
                stacklevel=2,
            )
            status_rows.append({**participant.to_dict(), "completion_status": "erroneous_completion"})
            continue

        df_trials = _load_session_trials(session_path, participant)
        trial_rows.append(df_trials)
        status_rows.append({**participant.to_dict(), "completion_status": "completed"})

    df_status = pd.DataFrame(status_rows).reset_index(drop=True)

    if trial_rows:
        df_trials_all = pd.concat(trial_rows, ignore_index=True)
    else:
        df_trials_all = pd.DataFrame()

    return {"trials": df_trials_all, "status": df_status}


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


def _index_session_files(data_dir: Path) -> dict[str, Path]:
    """Return {participant_id: path} for all session CSVs in data_dir.

    Participant IDs are read from the participant_id column inside each CSV
    because the filenames use literal placeholder tokens rather than actual IDs.
    """
    index: dict[str, Path] = {}
    for csv_path in data_dir.glob("*.csv"):
        if csv_path.name == _DEMOGRAPHICS_FILE:
            continue
        try:
            # Read only the first data row to extract participant_id cheaply
            df_head = pd.read_csv(csv_path, usecols=["participant_id"], nrows=1)
            pid = str(df_head["participant_id"].iloc[0])
            index[pid] = csv_path
        except (KeyError, IndexError, pd.errors.ParserError, ValueError):
            continue
    return index


def _load_session_trials(session_path: Path, participant: pd.Series) -> pd.DataFrame:
    df = pd.read_csv(session_path)

    # Read canvas dimensions before filtering (present on every row)
    canvas_w = int(df["sort_area_width"].iloc[0])
    canvas_h = int(df["sort_area_height"].iloc[0])

    df = df[df["trial_type"].isin(_EXPERIMENTAL_TRIAL_TYPES)].copy()

    # Derived columns
    df["trial_number"] = df["trial_type"].str.extract(r"trial_(\d+)").astype(int)
    df["n_moves"] = df["moves"].apply(_count_moves)
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
    # Keep task_version as a string — "1.0" would otherwise parse as float 1.0,
    # which would corrupt versions like "1.10" → 1.1.
    if "task_version" in df.columns:
        df["task_version"] = df["task_version"].astype(str)

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
