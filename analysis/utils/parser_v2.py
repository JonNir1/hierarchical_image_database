"""
Loader for SpAM session data -- flat data/ directory, cohort determined from
each file's own content (deployment_mode), not from which directory it lives in.

Usage (from repo root):
    from analysis.utils.parser_v2 import load_data
    data = load_data("data/")
    df_participants = data["participants"]  # one row per participant_id
    df_trials       = data["trials"]         # one row per real SpAM/catch trial;
                                              # join to df_participants on participant_id

This is a parser-only refactor; analysis/pilot/figures.py and analysis.ipynb keep
using analysis.utils.parser.load_pilot_data. See the plan this was built from for
the full rationale and verification strategy.
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from analysis.utils.parser import (
    _DEMOGRAPHICS_DROP,
    _DEMOGRAPHICS_RENAME,
    _EXPIRED_SENTINEL,
    _PROLIFIC_SENTINEL,
    _count_moves,
    _normalise_locations,
    parse_pairwise_distances,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MAIN_TRIAL_TYPE_RE = re.compile(r"^trial_(\d+)$")
_CATCH_TRIAL_TYPE_RE = re.compile(r"^catch_(\d+)$")
_SCREENING_EVAL_TRIAL_TYPE = "screening_eval"

# Session-level fields expected constant across every row of one session file.
_CONSTANT_FIELDS = ["deployment_mode", "shine_variant", "sort_area_width", "sort_area_height", "task_version"]

# Matches the timestamp embedded in a session filename, e.g.
# "..._2026-07-24_20h55.35.371.csv" -> ("2026-07-24", "20", "55", "35", "371")
_FILENAME_TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})\.(\d{2})\.(\d+)")

_TRIALS_COLUMNS = [
    "participant_id", "trial_id", "is_catch", "block_type",
    "rt", "num_moves", "init_locations", "moves", "final_locations",
    "pairwise_distances", "qc_flag", "repeat_of_trial", "reliability",
    "catch_trial_target_location", "centroid_x", "centroid_y", "cluster_mean_distance",
]

_PARTICIPANTS_COLUMNS = [
    "participant_id", "file_name", "num_associated_files", "cohort", "status",
    "shine_variant", "sort_area_width", "sort_area_height", "task_version",
    "submission_id", "prolific_status", "prolific_duration_s",
    "age", "sex", "ethnicity", "country_of_birth", "country_of_residence",
    "nationality", "language", "student_status", "employment_status",
    "reasons", "move_ratio_fail_rate", "distance_sd_fail_rate",
    "min_reliability", "median_reliability",
]

# screening_eval diagnostic fields, restored onto df_participants (v4.0+ only;
# NaN/None for pre-v4 sessions or v4 sessions that never reached the screening
# evaluation) -- needed to keep screening-threshold analyses possible directly
# from load_data() output, without a separate table.
_SCREENING_EVAL_DIAGNOSTIC_FIELDS = [
    "reasons", "move_ratio_fail_rate", "distance_sd_fail_rate",
    "min_reliability", "median_reliability",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load SpAM data from a flat *data_dir* (no pilot/prod split).

    Returns a dict with two keys:
      "participants" -- one row per participant_id
      "trials"       -- one row per real SpAM/catch trial (practice excluded);
                        join to "participants" on participant_id

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist, or no demographics file is found.
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

    demo_paths = sorted(p for p in data_dir.glob("*.csv") if "demographics" in p.name.lower())
    if not demo_paths:
        raise FileNotFoundError(f"No demographics file (matching *demographics*.csv) found in: {data_dir}")

    df_demo = pd.concat([_load_demographics(p) for p in demo_paths], ignore_index=True)
    dup_pids = df_demo["participant_id"][df_demo["participant_id"].duplicated()]
    if not dup_pids.empty:
        warnings.warn(
            f"Duplicate participant_id(s) across demographics files: {sorted(set(dup_pids))}",
            UserWarning,
            stacklevel=2,
        )

    demo_path_set = set(demo_paths)
    session_files = _index_session_files(data_dir, demo_path_set)

    participant_rows: list[dict] = []
    trial_rows: list[pd.DataFrame] = []

    for _, participant in df_demo.iterrows():
        pid = participant["participant_id"]
        candidates = session_files.get(pid, [])
        resolved = _resolve_session_file(pid, candidates)

        base = _participant_metadata(pid, resolved)
        status = _determine_status(participant["prolific_status"], resolved, base)
        row = {**participant.to_dict(), **base, "status": status}
        participant_rows.append(row)

        if status in {"full data", "screened out"}:
            trial_rows.append(_load_trials_for_participant(pid, resolved["path"]))

    df_participants = pd.DataFrame(participant_rows)
    df_participants = df_participants[[c for c in _PARTICIPANTS_COLUMNS if c in df_participants.columns]]

    if trial_rows:
        df_trials = pd.concat(trial_rows, ignore_index=True)
    else:
        df_trials = pd.DataFrame(columns=_TRIALS_COLUMNS)

    _validate_trial_repeat_image_sets_v2(df_trials, warn_on_mismatch=True)

    return {"participants": df_participants, "trials": df_trials}


def _validate_trial_repeat_image_sets_v2(df_trials: pd.DataFrame, warn_on_mismatch: bool = False) -> pd.DataFrame:
    """
    Sanity-check the repeat_of_trial mechanism: for every trial flagged as a repeat,
    verify it shares the same image set as its original (grouped by participant_id
    only -- each participant contributes exactly one resolved session file).

    Returns one row per repeat trial: participant_id, trial_id, repeat_of_trial,
    images_match, n_images_original, n_images_repeat. Never raises on a mismatch --
    inspect images_match, or pass warn_on_mismatch=True to also emit a UserWarning.
    """
    required = {"participant_id", "trial_id", "repeat_of_trial", "pairwise_distances"}
    missing = required - set(df_trials.columns)
    if missing:
        raise KeyError(f"_validate_trial_repeat_image_sets_v2: df_trials missing column(s): {sorted(missing)}")

    rows = []
    for pid, group in df_trials.groupby("participant_id"):
        images_by_id = {
            int(row["trial_id"]): _trial_image_set(row["pairwise_distances"])
            for _, row in group.iterrows()
        }
        repeats = group[group["repeat_of_trial"].notna()]
        for _, row in repeats.iterrows():
            trial_id = int(row["trial_id"])
            orig_id = int(row["repeat_of_trial"])
            orig_images = images_by_id.get(orig_id)
            rep_images = images_by_id.get(trial_id)
            rows.append({
                "participant_id": pid,
                "trial_id": trial_id,
                "repeat_of_trial": orig_id,
                "images_match": orig_images is not None and orig_images == rep_images,
                "n_images_original": len(orig_images) if orig_images is not None else 0,
                "n_images_repeat": len(rep_images) if rep_images is not None else 0,
            })

    report = pd.DataFrame(rows, columns=[
        "participant_id", "trial_id", "repeat_of_trial",
        "images_match", "n_images_original", "n_images_repeat",
    ])
    if warn_on_mismatch and not report.empty:
        bad = report[~report["images_match"]]
        if not bad.empty:
            warnings.warn(
                f"{len(bad)} repeat trial(s) do not share the same image set as their "
                f"original: {bad[['participant_id', 'trial_id']].to_dict('records')}",
                UserWarning,
                stacklevel=2,
            )
    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in _DEMOGRAPHICS_DROP if c in df.columns])
    df = df.rename(columns={k: v for k, v in _DEMOGRAPHICS_RENAME.items() if k in df.columns})
    df = df.replace({_PROLIFIC_SENTINEL: pd.NA, _EXPIRED_SENTINEL: pd.NA})
    for col in ("age", "prolific_duration_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _index_session_files(data_dir: Path, demo_paths: set[Path]) -> dict[str, list[Path]]:
    """Return {participant_id: [session paths]}, excluding demographics files."""
    index: dict[str, list[Path]] = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        if csv_path in demo_paths:
            continue
        try:
            df_head = pd.read_csv(csv_path, usecols=["participant_id"], nrows=1)
            pid = str(df_head["participant_id"].iloc[0])
            index.setdefault(pid, []).append(csv_path)
        except (KeyError, IndexError, pd.errors.ParserError, ValueError):
            warnings.warn(f"Could not read participant_id from {csv_path.name}; skipping.", UserWarning, stacklevel=2)
            continue
    return index


def _real_trial_row_count(session_path: Path) -> int:
    """Number of trial_N/catch_N rows in a session CSV (practice/consent/etc excluded)."""
    trial_types = pd.read_csv(session_path, usecols=["trial_type"])["trial_type"].astype(str)
    return int(
        trial_types.str.match(_MAIN_TRIAL_TYPE_RE).sum()
        + trial_types.str.match(_CATCH_TRIAL_TYPE_RE).sum()
    )


def _parse_filename_timestamp(path: Path) -> datetime:
    m = _FILENAME_TIMESTAMP_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse a timestamp from filename for recency tie-break: {path.name}")
    date_str, hh, mm, ss, frac = m.groups()
    return datetime.strptime(f"{date_str}_{hh}h{mm}.{ss}.{frac}", "%Y-%m-%d_%Hh%M.%S.%f")


def _resolve_session_file(pid: str, candidates: list[Path]) -> dict:
    """
    Pick one session file for *pid* out of *candidates* (possibly empty).

    Returns {"path": Path | None, "num_associated_files": int}. Ties in real-trial-row
    count are broken by most-recent filename timestamp; raises ValueError if recency
    can't be determined (unparseable timestamp, or two candidates parse identically).
    """
    if not candidates:
        return {"path": None, "num_associated_files": 0}
    if len(candidates) == 1:
        return {"path": candidates[0], "num_associated_files": 1}

    counts = {p: _real_trial_row_count(p) for p in candidates}
    max_count = max(counts.values())
    tied = [p for p, c in counts.items() if c == max_count]

    if len(tied) == 1:
        chosen = tied[0]
    else:
        timestamps = {p: _parse_filename_timestamp(p) for p in tied}
        max_ts = max(timestamps.values())
        most_recent = [p for p, ts in timestamps.items() if ts == max_ts]
        if len(most_recent) > 1:
            raise ValueError(
                f"Participant {pid}: cannot resolve which session file is authoritative -- "
                f"{len(tied)} files tie on real-trial-row count ({max_count}) and "
                f"{len(most_recent)} of those also tie on parsed timestamp: "
                f"{[p.name for p in most_recent]}"
            )
        chosen = most_recent[0]

    return {"path": chosen, "num_associated_files": len(candidates)}


def _cohort_from_prefix(path: Path) -> str | None:
    name = path.name
    if name.startswith("pilot_"):
        return "pilot"
    if name.startswith("prod_"):
        return "production"
    return None


def _assert_constant(df: pd.DataFrame, col: str, path: Path) -> object:
    """Return the single distinct non-null value of *col* in *df*, raising if
    there's more than one (session-level fields are expected constant per file)."""
    if col not in df.columns:
        return None
    values = df[col].dropna().unique()
    if len(values) > 1:
        raise ValueError(f"{path.name}: expected a single constant value for '{col}', found {list(values)}")
    return values[0] if len(values) == 1 else None


def _participant_metadata(pid: str, resolved: dict) -> dict:
    path = resolved["path"]
    out = {
        "file_name": path.name if path is not None else None,
        "num_associated_files": resolved["num_associated_files"],
        "cohort": None,
        "shine_variant": None,
        "sort_area_width": None,
        "sort_area_height": None,
        "task_version": None,
        **{k: None for k in _SCREENING_EVAL_DIAGNOSTIC_FIELDS},
    }
    if path is None:
        return out

    df = pd.read_csv(path)
    if "task_version" in df.columns:
        df["task_version"] = pd.to_numeric(df["task_version"], errors="coerce")

    deployment_mode = _assert_constant(df, "deployment_mode", path)
    if deployment_mode is None:
        deployment_mode = _cohort_from_prefix(path)
        if deployment_mode is None:
            raise ValueError(
                f"{path.name}: no deployment_mode value in the file and filename has no "
                f"recognizable pilot_/prod_ prefix -- cannot determine cohort for {pid}."
            )

    out["cohort"] = deployment_mode
    out["shine_variant"] = _assert_constant(df, "shine_variant", path)
    out["sort_area_width"] = _assert_constant(df, "sort_area_width", path)
    out["sort_area_height"] = _assert_constant(df, "sort_area_height", path)
    out["task_version"] = _assert_constant(df, "task_version", path)
    out.update(_screening_eval_diagnostics(path))
    return out


def _has_screening_fail(path: Path) -> bool:
    df = pd.read_csv(path)
    rows = df[df["trial_type"] == _SCREENING_EVAL_TRIAL_TYPE]
    if rows.empty:
        return False
    passed = rows.iloc[0]["pass"]
    return not bool(passed) if not pd.isna(passed) else False


def _screening_eval_diagnostics(path: Path) -> dict:
    """Read the screening_eval row's diagnostic fields, if present. Returns an
    all-None dict if there's no screening_eval row (pre-v4 session, or a v4
    session that never reached the screening evaluation)."""
    df = pd.read_csv(path)
    rows = df[df["trial_type"] == _SCREENING_EVAL_TRIAL_TYPE]
    if rows.empty:
        return {k: None for k in _SCREENING_EVAL_DIAGNOSTIC_FIELDS}
    row = rows.iloc[0]
    return {k: (row[k] if k in row.index and not pd.isna(row[k]) else None)
            for k in _SCREENING_EVAL_DIAGNOSTIC_FIELDS}


def _determine_status(prolific_status: str, resolved: dict, base: dict) -> str:
    if prolific_status in {"RETURNED", "REJECTED"}:
        return "revoked consent"

    path = resolved["path"]
    if path is None:
        return "missing data"

    if _real_trial_row_count(path) == 0:
        return "missing data"

    if _has_screening_fail(path):
        return "screened out"

    return "full data"


def _trial_image_set(pw_json: str) -> frozenset[str]:
    images: set[str] = set()
    for a, b in parse_pairwise_distances(pw_json):
        images.add(a)
        images.add(b)
    return frozenset(images)


def _safe_pair_spearman_r(orig_row: pd.Series, repeat_row: pd.Series, pid: str, trial_id: int) -> float | None:
    """Like spearman_r.pair_spearman_r, but degrades to None + a UserWarning on
    malformed pairwise_distances JSON instead of raising (real corrupted-file case
    observed in prod data)."""
    try:
        d_orig = _distance_dict(orig_row["pairwise_distances"])
        d_repeat = _distance_dict(repeat_row["pairwise_distances"])
    except json.JSONDecodeError:
        warnings.warn(
            f"Participant {pid}, trial_id {trial_id}: malformed pairwise_distances JSON, "
            f"reliability set to NaN.",
            UserWarning,
            stacklevel=2,
        )
        return None
    keys = d_orig.keys() & d_repeat.keys()
    if len(keys) < 2:
        return None
    keys = sorted(keys, key=lambda k: sorted(k))
    r, _p = spearmanr([d_orig[k] for k in keys], [d_repeat[k] for k in keys])
    return r


def _distance_dict(pw_json: str) -> dict[frozenset, float]:
    if pd.isna(pw_json) or pw_json == "":
        return {}
    items = json.loads(pw_json)
    return {frozenset((d["src1"], d["src2"])): d["distance"] for d in items}


def _load_trials_for_participant(pid: str, session_path: Path) -> pd.DataFrame:
    df = pd.read_csv(session_path)
    df["task_version"] = pd.to_numeric(df["task_version"], errors="coerce")

    canvas_w = int(_assert_constant(df, "sort_area_width", session_path))
    canvas_h = int(_assert_constant(df, "sort_area_height", session_path))

    trial_type_all = df["trial_type"].astype(str)
    main_mask = trial_type_all.str.match(_MAIN_TRIAL_TYPE_RE)
    catch_mask = trial_type_all.str.match(_CATCH_TRIAL_TYPE_RE)
    df = df[main_mask | catch_mask].copy()
    df = df.reset_index(drop=True)

    df["trial_id"] = range(1, len(df) + 1)
    df["is_catch"] = df["trial_type"].astype(str).str.match(_CATCH_TRIAL_TYPE_RE)

    if "block" in df.columns:
        df["block_type"] = df["block"].where(df["block"].notna(), "experimental")
    else:
        df["block_type"] = "experimental"

    df["qc_flag"] = df["qc_flag"].isin([True, "true", "True", 1])
    df["num_moves"] = df["moves"].apply(_count_moves).astype(int)
    df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

    for col in ("final_locations", "init_locations", "moves"):
        if col in df.columns:
            df[col] = df[col].apply(lambda s: _normalise_locations(s, canvas_w, canvas_h))

    for row_idx, row in df.iterrows():
        pw = row.get("pairwise_distances")
        if pd.isna(pw) or pw == "":
            continue
        try:
            json.loads(pw)
        except json.JSONDecodeError:
            warnings.warn(
                f"Participant {pid}, trial_id {row['trial_id']}: pairwise_distances failed to parse.",
                UserWarning,
                stacklevel=2,
            )

    # old trial-number (trial_N) -> new trial_id map, main trials only -- the space
    # repeat_of_trial_number references.
    trial_number_to_id: dict[int, int] = {}
    for _, row in df[~df["is_catch"]].iterrows():
        m = _MAIN_TRIAL_TYPE_RE.match(str(row["trial_type"]))
        if m:
            trial_number_to_id[int(m.group(1))] = int(row["trial_id"])

    repeat_of_trial = pd.Series([None] * len(df), index=df.index, dtype="object")
    reliability = pd.Series([None] * len(df), index=df.index, dtype="object")

    if "is_trial_repeat" in df.columns and "repeat_of_trial_number" in df.columns:
        is_repeat = df["is_trial_repeat"].isin([True, "true", "True", 1])
        repeat_number = pd.to_numeric(df["repeat_of_trial_number"], errors="coerce")
        for idx in df.index[is_repeat & repeat_number.notna()]:
            old_number = int(repeat_number.loc[idx])
            orig_trial_id = trial_number_to_id.get(old_number)
            if orig_trial_id is None:
                warnings.warn(
                    f"Participant {pid}, trial_id {df.loc[idx, 'trial_id']}: repeat_of_trial_number "
                    f"{old_number} does not resolve to a known trial in this session.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            repeat_of_trial.loc[idx] = orig_trial_id
            orig_row = df.loc[df["trial_id"] == orig_trial_id].iloc[0]
            reliability.loc[idx] = _safe_pair_spearman_r(orig_row, df.loc[idx], pid, int(df.loc[idx, "trial_id"]))

    df["repeat_of_trial"] = pd.to_numeric(repeat_of_trial, errors="coerce")
    df["reliability"] = pd.to_numeric(reliability, errors="coerce")
    df["participant_id"] = pid

    keep = [c for c in _TRIALS_COLUMNS if c in df.columns]
    return df[keep].copy()
