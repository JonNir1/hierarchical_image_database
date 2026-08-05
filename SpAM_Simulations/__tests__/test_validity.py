"""Tests for the empirical-validity checks.

Hand-built hierarchies with known LCA depths, plus an agreement test against
``analysis.rdms.semantic_km`` (skipped when that module cannot be imported, since it drags in the
broken images/manifest.csv loader).

No R needed.
"""
import numpy as np
import pytest
from scipy.spatial.distance import squareform

from SpAM_Simulations import validity as val

# Two categories, two subcategories each, two images per leaf. LCA depths are checkable by eye.
PATHS = [
    "animate/animal/bird/a1.png",
    "animate/animal/bird/a2.png",
    "animate/animal/mammal/b1.png",
    "animate/human/face/c1.png",
    "inanimate/handmade/tool/d1.png",
    "inanimate/handmade/tool/d2.png",
]


def test_hierarchy_levels_match_hand_computed_lca_depths():
    levels = squareform(val.hierarchy_levels(PATHS))
    assert levels[0, 1] == 3      # same leaf: animate/animal/bird
    assert levels[0, 2] == 2      # same category: animate/animal
    assert levels[0, 3] == 1      # same top level: animate
    assert levels[0, 4] == 0      # nothing shared
    assert levels[4, 5] == 3      # same leaf: inanimate/handmade/tool


def test_hierarchy_levels_is_condensed_and_the_right_length():
    lv = val.hierarchy_levels(PATHS)
    assert lv.shape == (len(PATHS) * (len(PATHS) - 1) // 2,)


def test_hierarchy_levels_handles_ragged_depths():
    """Real manifest paths are 3, 4 or 5 segments deep, so zip-based LCA must not overrun."""
    ragged = ["a/b/x.png", "a/b/c/d/y.png", "a/z.png"]
    lv = squareform(val.hierarchy_levels(ragged))
    assert lv[0, 1] == 2 and lv[0, 2] == 1


# --------------------------------------------------------------------- gradient

def _graded(levels, per_level):
    """Distances that decrease with LCA depth by construction."""
    return np.array([per_level[l] for l in levels], dtype=float)


def test_gradient_table_recovers_a_planted_gradient():
    levels = val.hierarchy_levels(PATHS)
    d = _graded(levels, {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2})
    table = val.gradient_table(d, levels)
    assert list(table["level"]) == [0, 1, 2, 3]
    assert list(table["mean_distance"]) == pytest.approx([1.0, 0.8, 0.5, 0.2])
    assert val.gradient_is_monotone(table)
    assert table["std_gap"].iloc[0] == pytest.approx(0.0)
    assert (table["std_gap"].diff().dropna() > 0).all(), "gaps must grow with relatedness"


def test_gradient_flags_a_non_monotone_space():
    levels = val.hierarchy_levels(PATHS)
    scrambled = _graded(levels, {0: 0.2, 1: 0.9, 2: 0.4, 3: 0.7})
    assert not val.gradient_is_monotone(val.gradient_table(scrambled, levels))


def test_std_gap_is_scale_free():
    """The gradient must be comparable between simulated and pilot distances despite unit mismatch."""
    levels = val.hierarchy_levels(PATHS)
    d = _graded(levels, {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2})
    a = val.gradient_table(d, levels)["std_gap"].to_numpy()
    b = val.gradient_table(d * 37.0, levels)["std_gap"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_gradient_table_ignores_unobserved_pairs():
    levels = val.hierarchy_levels(PATHS)
    d = _graded(levels, {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2})
    d[0] = np.nan
    table = val.gradient_table(d, levels)
    assert table["n_pairs"].sum() == np.isfinite(d).sum()


# --------------------------------------------------------------------- distributions

def test_identical_distributions_have_zero_wasserstein():
    x = np.random.default_rng(0).random(500) + 0.5
    out = val.distribution_comparison(x, x)
    assert out["wasserstein"] == pytest.approx(0.0)
    assert out["cv_ratio"] == pytest.approx(1.0)


def test_median_rescaling_absorbs_a_pure_unit_difference():
    """Simulated distances are in GT units, pilot ones normalised to [0,1]; only shape should count."""
    x = np.random.default_rng(0).random(500) + 0.5
    out = val.distribution_comparison(x * 100.0, x)
    assert out["wasserstein"] == pytest.approx(0.0, abs=1e-9)
    assert out["cv_ratio"] == pytest.approx(1.0)


def test_without_rescaling_a_unit_difference_dominates():
    x = np.random.default_rng(0).random(500) + 0.5
    assert val.distribution_comparison(x * 100.0, x, rescale="none")["wasserstein"] > 10


def test_shape_differences_survive_rescaling():
    rng = np.random.default_rng(0)
    tight = rng.normal(1.0, 0.05, 5000)
    broad = rng.normal(1.0, 0.40, 5000)
    out = val.distribution_comparison(broad, tight)
    assert out["cv_ratio"] > 3
    assert out["wasserstein"] > 0.05


def test_distribution_comparison_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one finite"):
        val.distribution_comparison(np.array([np.nan]), np.arange(5.0))


def test_unknown_rescale_is_rejected():
    with pytest.raises(ValueError, match="rescale"):
        val.distribution_comparison(np.arange(5.0), np.arange(5.0), rescale="nonsense")


# --------------------------------------------------------------------- end to end

def test_compare_to_pilot_reports_both_checks():
    levels = val.hierarchy_levels(PATHS)
    pilot = _graded(levels, {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2})
    sim = pilot * 12.0                      # same structure, different units
    out = val.compare_to_pilot(sim, pilot, PATHS)
    assert out["sim_gradient_monotone"] and out["pilot_gradient_monotone"]
    assert out["max_abs_std_gap_diff"] == pytest.approx(0.0, abs=1e-9)
    assert out["distribution"]["wasserstein"] == pytest.approx(0.0, abs=1e-9)


def test_compare_to_pilot_detects_a_structurally_wrong_simulation():
    """A simulation with no semantic gradient must be caught even if its marginal looks fine."""
    levels = val.hierarchy_levels(PATHS)
    pilot = _graded(levels, {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2})
    sim = np.random.default_rng(0).permutation(pilot)   # identical marginal, structure destroyed
    out = val.compare_to_pilot(sim, pilot, PATHS)
    assert out["distribution"]["wasserstein"] == pytest.approx(0.0, abs=1e-9), "marginal matches"
    assert out["max_abs_std_gap_diff"] > 0.1, "but the gradient must expose it"


# --------------------------------------------------------------------- cross-check

def test_lca_agrees_with_semantic_km():
    """The reimplementation must match analysis.rdms.semantic_km, which it deliberately avoids importing."""
    km = pytest.importorskip("analysis.rdms.semantic_km",
                             reason="analysis.rdms unavailable (it reads the broken images/manifest.csv)")
    for a in PATHS:
        for b in PATHS:
            mine = val._lca_depth(val._dir_parts(a), val._dir_parts(b))
            theirs = km._lca_depth(km._dir_parts(a), km._dir_parts(b))
            assert mine == theirs, f"{a} vs {b}"


def test_runs_on_the_real_manifest():
    """725 images -> 262,450 pairs, with every level populated."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    manifest = root / "SpAM_Task" / "stimuli_manifest.json"
    if not manifest.exists():
        pytest.skip("stimuli_manifest.json not present")
    images = json.loads(manifest.read_text())["images"]
    levels = val.hierarchy_levels(images)
    assert levels.shape == (len(images) * (len(images) - 1) // 2,)
    counts = dict(zip(*np.unique(levels, return_counts=True)))
    assert set(counts) >= {1, 2, 3}, f"expected several hierarchy levels, got {counts}"
