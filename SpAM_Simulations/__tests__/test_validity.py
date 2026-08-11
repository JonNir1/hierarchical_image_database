"""Tests for the empirical-validity checks.

Hand-built hierarchies with known LCA depths, plus an agreement test against
``analysis.rdms.semantic_km`` (skipped when that module cannot be imported, since it drags in the
broken images/manifest.csv loader).

No R needed.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import squareform

from SpAM_Simulations.measures import validity as val

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


# --------------------------------------------------------------------- noise vs distance
# The empirical curve is an inverted U: clearly-similar and clearly-dissimilar pairs are judged
# consistently, the ambiguous middle is not. The two flanks are NOT equally strong tests, and the
# tests below keep them apart on purpose - see the module comment.


class _RetestSubject:
    def __init__(self, pairs):
        self.retest_pairs = pairs


def test_repeat_pairs_pools_every_subjects_retest_vectors():
    subs = [_RetestSubject([(np.array([0.1, 0.2]), np.array([0.15, 0.25]))]),
            _RetestSubject([(np.array([0.3]), np.array([0.4])),
                            (np.array([0.5, 0.6]), np.array([0.5, 0.7]))])]
    o, r = val.repeat_pairs(subs)
    assert o.size == r.size == 5
    np.testing.assert_allclose(o, [0.1, 0.2, 0.3, 0.5, 0.6])


def test_repeat_pairs_drops_non_finite_and_survives_a_subject_with_no_repeats():
    subs = [_RetestSubject([]),
            _RetestSubject([(np.array([0.1, np.nan]), np.array([0.2, 0.3]))])]
    o, r = val.repeat_pairs(subs)
    np.testing.assert_allclose(o, [0.1])
    np.testing.assert_allclose(r, [0.2])
    assert val.repeat_pairs([_RetestSubject([])]) [0].size == 0


def test_noise_vs_distance_rmse_is_hand_computable():
    """Two bins, exact arithmetic, no rescaling: rmse == sqrt(mean((o-r)^2)) per bin."""
    o = np.array([1.0, 1.0, 10.0, 10.0])
    r = np.array([1.0, 2.0, 12.0, 10.0])       # sq diffs 0, 1, 4, 0
    t = val.noise_vs_distance(o, r, n_bins=2, binning="fixed", rescale="none")
    assert len(t) == 2
    np.testing.assert_allclose(t["rmse"].to_numpy(), [np.sqrt(0.5), np.sqrt(2.0)])
    assert list(t["n_pairs"]) == [2, 2]


def test_noise_vs_distance_matches_the_pilot_figures_definition():
    """Same quantity as analysis/pilot/figures.py: sqrt(mean(sq_diff)) over a bin of pair_mean."""
    rng = np.random.default_rng(0)
    o, r = rng.random(500), rng.random(500)
    t = val.noise_vs_distance(o, r, n_bins=4, binning="fixed", rescale="none")
    pair_mean, sq_diff = 0.5 * (o + r), (o - r) ** 2
    for _, row in t.iterrows():
        sel = (pair_mean >= row["bin_low"]) & (pair_mean <= row["bin_high"])
        assert row["rmse"] == pytest.approx(np.sqrt(sq_diff[sel].mean()), rel=1e-9)


def test_median_rescaling_makes_the_curve_scale_free():
    """Simulated and pilot distances are in different units; the shape must survive that."""
    rng = np.random.default_rng(1)
    o, r = rng.random(2000) + 0.1, rng.random(2000) + 0.1
    a = val.noise_vs_distance(o, r, n_bins=5)
    b = val.noise_vs_distance(o * 37.0, r * 37.0, n_bins=5)
    np.testing.assert_allclose(a["rmse"].to_numpy(), b["rmse"].to_numpy(), rtol=1e-9)


def test_a_planted_inverted_u_is_detected():
    rng = np.random.default_rng(2)
    pair_mean = rng.uniform(0, 1, 20000)
    # noise smallest at both ends, largest in the middle
    sd = 0.02 + 0.30 * np.sin(np.pi * pair_mean)
    delta = rng.normal(0, sd)
    o, r = pair_mean + delta / 2, pair_mean - delta / 2
    shape = val.noise_curve_shape(val.noise_vs_distance(o, r, n_bins=9))
    assert shape["has_low_floor"] and shape["turns_over"]
    assert shape["is_inverted_u"]
    assert 0.3 < shape["peak_bin_frac"] < 0.7


def test_a_monotonically_rising_curve_rises_but_never_turns_over():
    """The shape an unbounded generative model produces: a floor, but no ceiling."""
    rng = np.random.default_rng(3)
    pair_mean = rng.uniform(0, 1, 20000)
    delta = rng.normal(0, 0.02 + 0.30 * pair_mean)
    o, r = pair_mean + delta / 2, pair_mean - delta / 2
    shape = val.noise_curve_shape(val.noise_vs_distance(o, r, n_bins=9))
    assert shape["has_low_floor"]
    assert not shape["turns_over"]
    assert shape["drop_from_peak"] == pytest.approx(0.0, abs=1e-9)
    assert not shape["is_inverted_u"]
    assert shape["peak_bin_frac"] == 1.0


def test_the_descriptor_is_peak_relative_not_thirds_relative():
    """Regression on a real defect: the thirds summary called the PILOT not an inverted U.

    The empirical turnover is confined to the top bin and the peak sits at ~78% of the range, so
    averaging a high third smears the drop away (the pilot scores high_over_mid = 1.30). Anything
    that reads the verdict off thirds is wrong for this curve.
    """
    # A long rise to a late peak, then one sharp drop - the empirical shape in miniature.
    rmse = np.array([0.06, 0.12, 0.16, 0.18, 0.20, 0.21, 0.27, 0.32, 0.31, 0.20])
    table = pd.DataFrame({"rmse": rmse, "n_pairs": np.full(len(rmse), 100)})
    shape = val.noise_curve_shape(table)
    assert shape["peak_bin_frac"] == pytest.approx(7 / 9)
    assert shape["drop_from_peak"] == pytest.approx(1 - 0.20 / 0.32)
    assert shape["rise_from_first"] == pytest.approx(1 - 0.06 / 0.32)
    assert shape["turns_over"] and shape["has_low_floor"] and shape["is_inverted_u"]
    # The thirds are still reported, and on this curve they would have said the opposite.
    assert shape["rmse_high"] > shape["rmse_mid"]


def test_noise_curve_rejects_degenerate_input():
    with pytest.raises(ValueError, match="match in length"):
        val.noise_vs_distance(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="no finite"):
        val.noise_vs_distance(np.array([np.nan]), np.array([np.nan]))
    with pytest.raises(ValueError, match="rescale must be"):
        val.noise_vs_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0]), rescale="zscore")
    with pytest.raises(ValueError, match="binning must be"):
        val.noise_vs_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0]), binning="log")
    with pytest.raises(ValueError, match="empty curve"):
        val.noise_curve_shape(pd.DataFrame())


def test_compare_noise_vs_distance_reports_both_sources_and_the_flank_verdicts():
    rng = np.random.default_rng(4)
    pm = rng.uniform(0, 1, 8000)
    inv = rng.normal(0, 0.02 + 0.30 * np.sin(np.pi * pm))       # pilot-like inverted U
    rise = rng.normal(0, 0.02 + 0.30 * pm)                      # model-like rising curve
    out = val.compare_noise_vs_distance((pm + rise / 2, pm - rise / 2),
                                             (pm + inv / 2, pm - inv / 2), n_bins=9)
    assert set(out["curves"]["source"]) == {"sim", "pilot"}
    assert list(out["shape"]["source"]) == ["sim", "pilot"]
    assert out["pilot_is_inverted_u"] and not out["sim_is_inverted_u"]
    assert out["low_end_matches"]        # both rise off the near-forced floor
    assert not out["turnover_matches"]   # only the pilot has the ceiling


def test_the_generative_model_reproduces_the_low_flank_but_not_the_high_one():
    """A prediction, recorded as a test so a future model change is noticed.

    `task_v3_experiment` places points on an UNBOUNDED plane - its docstring says a fixed-canvas
    ceiling is "a documented future refinement, not modelled". Real subjects work on a bounded
    canvas, where a pair already at opposite corners cannot move much further apart, which is what
    produces the empirical high-distance flank. So the model should get the low flank (forced by
    distances being non-negative) and miss the high one. If this test starts failing because
    `high_flank_quieter` became True, a ceiling was added and FINDINGS.md should say so.
    """
    coords = np.random.default_rng(0).normal(size=(120, 5)).astype(np.float32)
    o, r = val.simulate_repeat_pairs(coords, subjects_noise_scale=0.35, n_subjects=15,
                                          trials_per_subject=4, images_per_trial=16, seed=1)
    shape = val.noise_curve_shape(val.noise_vs_distance(o, r, n_bins=10))
    assert shape["has_low_floor"], shape
    assert not shape["turns_over"], shape


def test_simulate_repeat_pairs_returns_matched_vectors_and_checks_the_trial_size():
    coords = np.random.default_rng(0).normal(size=(40, 4)).astype(np.float32)
    o, r = val.simulate_repeat_pairs(coords, 0.3, n_subjects=3, trials_per_subject=2,
                                          images_per_trial=10, seed=0)
    assert o.shape == r.shape == (3 * 2 * (10 * 9 // 2),)
    assert np.isfinite(o).all() and np.isfinite(r).all()
    # A repeat is a fresh draw, not a copy: the two must differ.
    assert not np.allclose(o, r)
    with pytest.raises(ValueError, match="exceeds n_images"):
        val.simulate_repeat_pairs(coords, 0.3, images_per_trial=100)


# --------------------------------------------------------------------- support-aware gradient gate

def test_min_pairs_ignores_an_underpowered_level():
    """A level with almost no pairs must not be able to condemn the whole gradient.

    This is the failure that produced a spurious "do not trust the arm comparison" banner on a
    finished 15-hour run: `depth_5` holds 348 of 262,450 pairs, and its inversion flipped a boolean
    that gated every downstream conclusion.
    """
    table = pd.DataFrame({
        "level": [0, 1, 2, 3],
        "n_pairs": [10000, 5000, 2000, 4],       # the last level is the underpowered one
        "mean_distance": [1.0, 0.8, 0.5, 0.9],   # ... and it inverts
    })
    assert not val.gradient_is_monotone(table)
    assert val.gradient_is_monotone(table, min_pairs=100)


def test_min_pairs_does_not_rescue_a_genuinely_broken_gradient():
    table = pd.DataFrame({
        "level": [0, 1, 2, 3],
        "n_pairs": [10000, 5000, 2000, 1000],
        "mean_distance": [1.0, 0.4, 0.9, 0.2],   # a well-supported level inverts
    })
    assert not val.gradient_is_monotone(table, min_pairs=100)


def test_compare_to_pilot_tests_the_same_levels_on_both_sides():
    """Support is judged jointly: a level thin on either side is skipped on both."""
    levels = val.hierarchy_levels(PATHS)
    d = _graded(levels, {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2})
    rep = val.compare_to_pilot(d, d, PATHS)
    tested, skipped = rep["gradient_levels_tested"], rep["gradient_levels_skipped"]
    assert set(tested).isdisjoint(skipped)
    assert sorted(tested + skipped) == [0, 1, 2, 3]
    # This toy manifest has far fewer than MIN_GRADIENT_PAIRS per level, so everything is skipped -
    # and the report must say so rather than quietly returning a vacuous True.
    assert tested == [] and rep["sim_gradient_monotone_all_levels"] is True


# --------------------------------------------------------------------- trial_simulator seam

def test_simulate_repeat_pairs_uses_the_injected_simulator():
    """The seam exists so the noise curve describes the model the sweep runs.

    Without it this function silently measured task-v3's unbounded placement while the sweep ran
    task-v5's bounded canvas - and since the turnover being measured is *caused* by the canvas, the
    check would have been reporting on a model nobody was running. It crashed a 15-hour stage-2 run
    with a TypeError because nothing exercised the argument.
    """
    calls = []

    def fake_simulator(trial, rows, cols, n_images, gt, weights, noise, obs, n_obs, rng):
        calls.append(noise)
        return None, np.full(rows.shape[0], 0.25), None

    gt = np.random.default_rng(0).random((12, 3)).astype(np.float32)
    orig, repeat = val.simulate_repeat_pairs(gt, subjects_noise_scale=0.3, n_subjects=2,
                                             trials_per_subject=2, images_per_trial=4, seed=1,
                                             trial_simulator=fake_simulator)
    assert calls, "the injected simulator was never called"
    assert np.all(orig == 0.25) and np.all(repeat == 0.25)
    assert orig.size == repeat.size == 2 * 2 * (4 * 3 // 2)


def test_simulate_repeat_pairs_defaults_to_the_v3_simulator():
    """Omitting the seam must keep the v3/v4 behaviour its bit-exactness tests depend on."""
    gt = np.random.default_rng(2).random((15, 3)).astype(np.float32)
    kwargs = dict(subjects_noise_scale=0.2, n_subjects=2, trials_per_subject=2,
                  images_per_trial=5, seed=7)
    a = val.simulate_repeat_pairs(gt, **kwargs)
    b = val.simulate_repeat_pairs(gt, trial_simulator=None, **kwargs)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


# --------------------------------------------------------------------- gradient across cells

def _cells(arm, monotone_flags, gaps=None):
    gaps = [0.3] * len(monotone_flags) if gaps is None else gaps
    return pd.DataFrame({"arm": arm, "monotone": monotone_flags, "max_abs_std_gap_diff": gaps})


def test_summarise_gradients_reports_a_fraction_not_a_verdict():
    """The case a boolean cannot express: the gradient depends on a lever the sweep varies."""
    summary = val.summarise_gradients(_cells("random", [True, True, False, True]))
    row = summary.iloc[0]
    assert row["n_cells"] == 4 and row["n_monotone"] == 3
    assert row["monotone_frac"] == pytest.approx(0.75)


def test_summarise_gradients_separates_the_arms():
    per_cell = pd.concat([_cells("random", [True, True]), _cells("designed", [False, False])],
                         ignore_index=True)
    summary = val.summarise_gradients(per_cell).set_index("arm")
    assert summary.loc["random", "monotone_frac"] == 1.0
    assert summary.loc["designed", "monotone_frac"] == 0.0


def test_summarise_gradients_carries_the_gap_spread():
    summary = val.summarise_gradients(_cells("random", [True] * 3, gaps=[0.1, 0.3, 0.5]))
    row = summary.iloc[0]
    assert row["gap_min"] == pytest.approx(0.1) and row["gap_max"] == pytest.approx(0.5)
    assert row["gap_mean"] == pytest.approx(0.3)
    assert row["gap_sd"] > 0, "a single number would hide how much the cells differ"


def test_summarise_gradients_survives_no_cells():
    """TABLES_ONLY reruns have no cohorts in memory, and that must not raise."""
    summary = val.summarise_gradients(pd.DataFrame())
    assert summary.empty and "monotone_frac" in summary.columns
