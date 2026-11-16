"""Tests for analysis.pre_vs_post_shine.data_quality.subject_repeat_slope (the fig_*
builders follow this repo's convention of being checked by notebook execution)."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.pre_vs_post_shine.data_quality import subject_repeat_slope


def test_subject_repeat_slope_recovers_trends_drops_short_series_and_is_zero_when_flat():
    rows = (
        [("p1", i, 0.2 + 0.05 * i) for i in (1, 2, 3, 4)]
        + [("p2", i, 0.5 - 0.1 * i) for i in (1, 2, 3, 4)]
        + [("flat", i, 0.3) for i in (1, 2, 3, 4)]
        + [("short", 1, 0.3), ("short", 2, 0.4)]
    )
    slopes = subject_repeat_slope(pd.DataFrame(rows, columns=["participant_id", "repeat_index", "spearman_r"]))
    assert slopes["p1"] == pytest.approx(0.05) and slopes["p2"] == pytest.approx(-0.1)
    assert slopes["flat"] == pytest.approx(0.0, abs=1e-12)
    assert "short" not in slopes.index  # fewer than min_repeats repeats
