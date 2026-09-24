"""Tests for analysis.behavioral.cohort_quality's pure data helpers (the fig_*
builders follow this repo's convention of being checked by notebook execution)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.behavioral.cohort_quality import subject_repeat_slope


def _repeat_df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["participant_id", "repeat_index", "spearman_r"])


def test_subject_repeat_slope_recovers_linear_trend_per_subject():
    df = _repeat_df(
        [("p1", i, 0.2 + 0.05 * i) for i in (1, 2, 3, 4)]
        + [("p2", i, 0.5 - 0.1 * i) for i in (1, 2, 3, 4)]
    )
    slopes = subject_repeat_slope(df)
    assert slopes["p1"] == pytest.approx(0.05)
    assert slopes["p2"] == pytest.approx(-0.1)


def test_subject_repeat_slope_drops_subjects_with_too_few_repeats():
    df = _repeat_df([("p1", 1, 0.3), ("p1", 2, 0.4), ("p2", 1, 0.1), ("p2", 2, 0.2), ("p2", 3, 0.3)])
    slopes = subject_repeat_slope(df, min_repeats=3)
    assert list(slopes.index) == ["p2"]


def test_subject_repeat_slope_flat_reliability_is_zero():
    df = _repeat_df([("p1", i, 0.3) for i in (1, 2, 3, 4)])
    assert subject_repeat_slope(df)["p1"] == pytest.approx(0.0, abs=1e-12)
