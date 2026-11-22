"""Tests for analysis.s2_semantic_hierarchy.rdm_vs_hierarchy: the statistical helpers and figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis.s2_semantic_hierarchy.rdm_vs_hierarchy as hc
from analysis.utils.hierarchy import lca_depth_condensed, path_parts

B = chr(92)  # backslash, matching the manifest's Windows-style curated_path

# A deliberately ragged toy tree: leaves at depth 2 and depth 3.
_TOY_PATHS = [
    f"animate{B}bird{B}duck1.png",
    f"animate{B}bird{B}duck2.png",
    f"animate{B}mammal{B}cat{B}cat1.png",
    f"animate{B}mammal{B}cat{B}cat2.png",
    f"animate{B}mammal{B}dog{B}dog1.png",
    f"inanimate{B}tool{B}hammer1.png",
]
_TOY_N = len(_TOY_PATHS)


@pytest.fixture()
def toy_tree(monkeypatch):
    """Point rdm_vs_hierarchy at the toy tree instead of the real manifest."""
    parts = path_parts(_TOY_PATHS)
    monkeypatch.setattr(hc, "manifest_parts", lambda: parts)
    return parts


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    def test_diagonal_is_one_and_symmetric(self):
        a = np.array([1.0, 2, 3, 4, 5, 6])
        rdms = {"a": a, "b": a[::-1].copy(), "c": np.array([1.0, 1, 2, 2, 3, 3])}
        corr = hc.correlation_matrix(rdms=rdms)
        assert np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr.to_numpy(), corr.to_numpy().T)

    def test_reversed_vector_is_perfectly_anticorrelated(self):
        a = np.array([1.0, 2, 3, 4, 5, 6])
        corr = hc.correlation_matrix(rdms={"a": a, "rev": a[::-1].copy()})
        assert corr.loc["a", "rev"] == pytest.approx(-1.0)


class TestPartialCorrelation:
    def test_controlling_for_a_copy_removes_correlation(self):
        rng = np.random.default_rng(0)
        c = rng.random(500)
        a = c + 0.01 * rng.random(500)
        b = c + 0.01 * rng.random(500)
        assert hc.partial_correlation(a, b, c) < 0.5
        assert np.corrcoef(a, b)[0, 1] > 0.9

    def test_independent_control_leaves_correlation(self):
        rng = np.random.default_rng(0)
        a = rng.random(500)
        b = a + 0.05 * rng.random(500)
        control = rng.random(500)
        assert hc.partial_correlation(a, b, control) > 0.8


class TestIncrementalR2:
    def test_reports_each_predictor_joint_and_sum(self):
        rng = np.random.default_rng(0)
        t = rng.random(300)
        out = hc.incremental_r2(t, {"p1": t + 0.1 * rng.random(300), "p2": rng.random(300)})
        assert list(out["predictors"]) == ["p1", "p2", "p1 + p2", "(sum if independent)"]
        joint = out.loc[out["predictors"] == "p1 + p2", "r2"].iloc[0]
        assert 0.0 <= joint <= 1.0
        # joint must be at least as good as either predictor alone
        assert joint >= out.loc[out["predictors"] == "p1", "r2"].iloc[0] - 1e-9


class TestLevelSeparation:
    def test_between_category_exceeds_within_for_a_tree_distance(self, toy_tree):
        d = 8.0 - lca_depth_condensed(toy_tree)
        out = hc.level_separation(1, rdms={"sem_km": d})
        assert out.loc["sem_km", "between"] > out.loc["sem_km", "within"]
        assert out.loc["sem_km", "cohens_d"] > 0

    def test_constant_rdm_gives_zero_separation(self, toy_tree):
        flat = np.ones(_TOY_N * (_TOY_N - 1) // 2)
        out = hc.level_separation(1, rdms={"flat": flat})
        assert out.loc["flat", "cohens_d"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Figures: two questions, two figures
# ---------------------------------------------------------------------------

class TestFigures:
    def test_depth_figure_skips_the_count_column(self, toy_tree):
        profile = pd.DataFrame(
            {"n_pairs": [10, 5], "a": [1.0, 0.5], "b": [1.0, 0.9]},
            index=pd.Index([0, 1], name="lca_depth"),
        )
        fig = hc.plot_depth_profile(profile)
        assert {t.name for t in fig.data} == {"a", "b"}
