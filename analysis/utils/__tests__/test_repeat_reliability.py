"""Tests for analysis.utils.repeat_reliability.repeat_reliability_by_index."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.utils.repeat_reliability import repeat_reliability_by_index


def _trials(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["participant_id", "trial_id", "repeat_of_trial", "reliability"])


def test_repeat_index_is_ordered_by_trial_id_within_subject():
    df = _trials([
        ("p1", 9, 2, 0.3), ("p1", 1, np.nan, np.nan), ("p1", 5, 1, 0.1), ("p1", 7, 3, 0.2),
        ("p2", 4, 1, 0.5), ("p2", 8, 2, 0.6),
    ])
    result = repeat_reliability_by_index(df)
    assert result["participant_id"].tolist() == ["p1", "p1", "p1", "p2", "p2"]
    assert result["repeat_index"].tolist() == [1, 2, 3, 1, 2]
    assert result["spearman_r"].tolist() == [0.1, 0.2, 0.3, 0.5, 0.6]


def test_repeats_with_missing_reliability_are_dropped_before_ranking():
    df = _trials([("p1", 3, 1, np.nan), ("p1", 6, 2, 0.4), ("p1", 8, 3, 0.7)])
    result = repeat_reliability_by_index(df)
    assert result["repeat_index"].tolist() == [1, 2]
    assert result["spearman_r"].tolist() == [0.4, 0.7]


def test_no_repeats_returns_empty_frame_with_columns():
    df = _trials([("p1", 1, np.nan, np.nan)])
    result = repeat_reliability_by_index(df)
    assert result.empty
    assert list(result.columns) == ["participant_id", "repeat_index", "spearman_r"]
