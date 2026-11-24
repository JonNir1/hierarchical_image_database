"""Tests for analysis.s2_semantic_hierarchy.semantic_vs_lowlevel: partial correlation, incremental R^2, commonality."""
from __future__ import annotations

import numpy as np
import pytest

import analysis.s2_semantic_hierarchy.semantic_vs_lowlevel as svl


class TestPartialCorrelation:
    def test_controlling_for_a_copy_removes_correlation(self):
        rng = np.random.default_rng(0)
        c = rng.random(500)
        a = c + 0.01 * rng.random(500)
        b = c + 0.01 * rng.random(500)
        assert svl.partial_correlation(a, b, c) < 0.5
        assert np.corrcoef(a, b)[0, 1] > 0.9

    def test_independent_control_leaves_correlation(self):
        rng = np.random.default_rng(0)
        a = rng.random(500)
        b = a + 0.05 * rng.random(500)
        control = rng.random(500)
        assert svl.partial_correlation(a, b, control) > 0.8


class TestIncrementalR2:
    def test_reports_each_predictor_joint_and_sum(self):
        rng = np.random.default_rng(0)
        t = rng.random(300)
        out = svl.incremental_r2(t, {"p1": t + 0.1 * rng.random(300), "p2": rng.random(300)})
        assert list(out["predictors"]) == ["p1", "p2", "p1 + p2", "(sum if independent)"]
        joint = out.loc[out["predictors"] == "p1 + p2", "r2"].iloc[0]
        assert 0.0 <= joint <= 1.0
        # joint must be at least as good as either predictor alone
        assert joint >= out.loc[out["predictors"] == "p1", "r2"].iloc[0] - 1e-9


class TestCommonality:
    def test_components_sum_to_full_r2_and_a_copy_has_no_unique_part(self):
        rng = np.random.default_rng(0)
        target = rng.random(400)
        p1 = target + 0.5 * rng.random(400)
        p2 = rng.random(400) + 0.3 * target
        predictors = {"p1": p1, "p1_copy": p1.copy(), "p2": p2}
        target[::7] = np.nan
        out = svl.commonality(target, predictors)
        assert len(out) == 2 ** 3 - 1
        assert out["value"].sum() == pytest.approx(out.attrs["full_r2"])
        unique = out.loc[out["kind"] == "unique"].set_index("component")["value"]
        assert unique["p1"] == pytest.approx(0.0, abs=1e-9)
        assert unique["p1_copy"] == pytest.approx(0.0, abs=1e-9)
