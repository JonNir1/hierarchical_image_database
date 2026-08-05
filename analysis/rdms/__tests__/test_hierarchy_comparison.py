"""Tests for analysis.rdms.hierarchy_comparison: tree derivation from manifest
paths, the KM reconstruction identity, and the statistical helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import squareform

import analysis.rdms.common as common
import analysis.rdms.hierarchy_comparison as hc
import analysis.rdms.semantic_km as km

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


def _toy_manifest(paths: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "curated_path": paths,
        "curated_filename": [p.split(B)[-1] for p in paths],
        "category": ["X"] * len(paths),
        "wn_synset_name": ["dog.n.01"] * len(paths),
    })


@pytest.fixture()
def toy_tree(monkeypatch):
    """Point hierarchy_comparison at the toy tree instead of the real manifest."""
    df = _toy_manifest(_TOY_PATHS)
    monkeypatch.setattr(hc, "load_manifest", lambda: df)
    monkeypatch.setattr(hc, "_EXPECTED_N", _TOY_N)
    return df


# ---------------------------------------------------------------------------
# Tree derivation
# ---------------------------------------------------------------------------

class TestTreeDerivation:
    def test_path_parts_strips_filename(self, toy_tree):
        assert hc.path_parts()[0] == ("animate", "bird")
        assert hc.path_parts()[2] == ("animate", "mammal", "cat")

    def test_path_depths(self, toy_tree):
        assert list(hc.path_depths()) == [2, 2, 3, 3, 3, 2]

    def test_ragged_detected(self, toy_tree):
        assert hc.depth_is_ragged() is True

    def test_uniform_tree_not_ragged(self, monkeypatch):
        paths = [f"a{B}b{B}x1.png", f"a{B}c{B}x2.png"]
        monkeypatch.setattr(hc, "load_manifest", lambda: _toy_manifest(paths))
        assert hc.depth_is_ragged() is False

    def test_lca_depth_values(self, toy_tree):
        lca = hc.lca_depth_matrix()
        # the two ducks share animate/bird
        assert lca[0, 1] == 2
        # duck vs cat share only animate
        assert lca[0, 2] == 1
        # cat vs dog share animate/mammal
        assert lca[2, 4] == 2
        # animate vs inanimate share nothing
        assert lca[0, 5] == 0

    def test_lca_matrix_symmetric_zero_diagonal(self, toy_tree):
        lca = hc.lca_depth_matrix()
        assert np.array_equal(lca, lca.T)
        assert np.all(np.diag(lca) == 0)

    def test_condensed_matches_matrix(self, toy_tree):
        assert np.array_equal(
            hc.lca_depth_condensed(), squareform(hc.lca_depth_matrix(), checks=False)
        )


# ---------------------------------------------------------------------------
# The reconstruction identity
# ---------------------------------------------------------------------------

class TestReconstruction:
    def test_matches_semantic_km_builder(self, toy_tree, monkeypatch, tmp_path):
        """The independent reimplementation must agree with the real builder.

        This is the check the notebook relies on. hierarchy_comparison keeps its
        own LCA implementation precisely so this comparison has teeth, so the two
        must be verified against each other here.
        """
        monkeypatch.setattr(km, "load_manifest", lambda: _toy_manifest(_TOY_PATHS))
        monkeypatch.setattr(common, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(common, "_EXPECTED_N", _TOY_N)
        monkeypatch.setattr(common, "_EXPECTED_LEN", _TOY_N * (_TOY_N - 1) // 2)
        assert np.array_equal(hc.reconstruct_km_from_hierarchy(), km.build_km_rdm())

    def test_off_diagonal_floored_at_one(self, toy_tree):
        # the two ducks sit in the same leaf folder: raw path length 0, floored to 1
        assert squareform(hc.reconstruct_km_from_hierarchy())[0, 1] == 1.0

    def test_distance_grows_with_separation(self, toy_tree):
        sq = squareform(hc.reconstruct_km_from_hierarchy())
        assert sq[0, 1] < sq[0, 2] < sq[0, 5]


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


class TestLabelShuffleTest:
    def test_identical_rdms_hit_significance_floor(self, toy_tree):
        d = squareform(np.array([
            [0, 1, 2, 3, 4, 5], [1, 0, 6, 7, 8, 9], [2, 6, 0, 1, 2, 3],
            [3, 7, 1, 0, 4, 5], [4, 8, 2, 4, 0, 6], [5, 9, 3, 5, 6, 0],
        ], dtype=float), checks=False)
        res = hc.label_shuffle_test(d, d, n_perm=99)
        assert res.observed == pytest.approx(1.0)
        assert res.p_value == pytest.approx(1 / 100)

    def test_null_is_centred_near_zero(self, toy_tree):
        rng = np.random.default_rng(1)
        a = squareform(squareform(rng.random(15)), checks=False)
        b = squareform(squareform(rng.random(15)), checks=False)
        res = hc.label_shuffle_test(a, b, n_perm=200)
        assert abs(res.null_mean) < 0.15

    def test_z_property(self):
        r = hc.ShuffleResult(observed=0.5, null_mean=0.1, null_sd=0.2, p_value=0.01)
        assert r.z == pytest.approx(2.0)


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
    def test_between_category_exceeds_within_for_km(self, toy_tree):
        d = hc.reconstruct_km_from_hierarchy()
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
