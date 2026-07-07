from pathlib import Path

import pandas as pd

_DEMOGRAPHICS_COLUMNS = [
    "Submission id", "Participant id", "Status", "Time taken", "Age", "Sex",
    "Ethnicity simplified", "Country of birth", "Country of residence",
    "Nationality", "Language", "Student status", "Employment status",
]

_SESSION_COLUMNS = [
    "participant_id", "trial_type", "sort_area_width", "sort_area_height",
    "moves", "rt", "qc_flag", "shine_variant", "task_version", "deployment_mode",
    "pairwise_distances", "final_locations", "init_locations",
    "is_trial_repeat", "repeat_of_trial_number",
    "catch_trial_target_location", "centroid_x", "centroid_y", "cluster_mean_distance",
]


def write_demographics_csv(data_dir: Path, rows: list[dict], filename: str = "participant_demographics.csv") -> Path:
    """Write a Prolific-style demographics CSV into *data_dir*. Missing columns default to ''."""
    df = pd.DataFrame([{col: row.get(col, "") for col in _DEMOGRAPHICS_COLUMNS} for row in rows])
    path = data_dir / filename
    df.to_csv(path, index=False)
    return path


def write_session_csv(data_dir: Path, rows: list[dict], filename: str) -> Path:
    """Write a task-session CSV (one row per trial) into *data_dir*. Missing columns default to ''."""
    df = pd.DataFrame([{col: row.get(col, "") for col in _SESSION_COLUMNS} for row in rows])
    path = data_dir / filename
    df.to_csv(path, index=False)
    return path
