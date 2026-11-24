"""Tests for analysis.s2_semantic_hierarchy.depth_gradient: the NaN-aware depth profile, rank slope and figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.s2_semantic_hierarchy import common as c
from analysis.s2_semantic_hierarchy import depth_gradient as dg
from analysis.utils.hierarchy import path_parts

B = chr(92)
_TOY_PATHS = [
    f"animate{B}bird{B}duck1.png",
    f"animate{B}bird{B}duck2.png",
    f"animate{B}mammal{B}cat{B}cat1.png",
    f"animate{B}mammal{B}cat{B}cat2.png",
    f"animate{B}mammal{B}dog{B}dog1.png",
    f"inanimate{B}tool{B}hammer1.png",
]


@pytest.fixture()
def toy_tree(monkeypatch):
    parts = path_parts(_TOY_PATHS)
    monkeypatch.setattr(c, "tree_parts", lambda: parts)
    return parts


class TestDepthGradient:
    def test_falling_distance_gives_falling_profile_and_positive_slope_ignoring_nans(self, toy_tree):
        tb = dg.tree_bins(toy_tree)
        distance = 10.0 - tb.levels[tb.bin_index]
        gappy = distance.copy()
        gappy[::4] = np.nan
        for x in (distance, gappy):
            profile = dg.normalised_profile(x, tb.bin_index, tb.n_bins)
            assert dg.is_monotone_falling(profile)
            assert profile[0] == pytest.approx(1.0)
            assert dg.rank_slope(x, tb.bin_index) > 0.8
        table = dg.rank_slope_table({"sem_km": distance}, n_boot=20, cache_dir=None, parts=toy_tree)
        assert table.loc["sem_km", "rank_slope"] > 0.8


class TestFigures:
    def test_depth_figure_skips_the_count_column(self):
        profile = pd.DataFrame(
            {"n_pairs": [10, 5], "clip_pre": [1.0, 0.5], "sens_pre": [1.0, 0.9]},
            index=pd.Index(["0", "1"], name="lca_bin"),
        )
        fig = dg.plot_depth_profile(profile)
        assert {t.name for t in fig.data} == {"CLIP pre", "pixels pre"}
