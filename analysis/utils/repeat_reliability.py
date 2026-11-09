"""
Test-retest reliability by repeat index: one Spearman r per verbatim trial
repeat, ordered within each subject's session. Shared by
analysis/pilot/figures.py (per task-version group) and
analysis/behavioral/data_quality.py (per SHINE cohort).

Usage (from repo root):
    from analysis.utils.repeat_reliability import repeat_reliability_by_index
"""
from __future__ import annotations

import pandas as pd


def repeat_reliability_by_index(df_trials: pd.DataFrame) -> pd.DataFrame:
    """Per-subject, per-repeat-ordinal Spearman r for trials that verbatim
    repeat an earlier trial (`repeat_of_trial` notna). `repeat_index` is 1-based,
    ordered by ascending `trial_id` within each subject's session -- repeat #1
    is the earliest repeat trial encountered, etc. Reliability values are read
    from the trials' precomputed `reliability` column (analysis/utils/parser.py
    computes Spearman r between each repeat and its original), never
    recomputed; repeats with a missing reliability are dropped before ranking.

    No version or cohort filtering -- the caller passes the trials it wants.
    Returns participant_id, repeat_index, spearman_r, sorted by participant_id
    then repeat_index."""
    repeats = (
        df_trials[df_trials["repeat_of_trial"].notna() & df_trials["reliability"].notna()]
        .sort_values(["participant_id", "trial_id"])
    )
    return pd.DataFrame({
        "participant_id": repeats["participant_id"].to_numpy(),
        "repeat_index": repeats.groupby("participant_id").cumcount().to_numpy() + 1,
        "spearman_r": repeats["reliability"].astype(float).to_numpy(),
    })
