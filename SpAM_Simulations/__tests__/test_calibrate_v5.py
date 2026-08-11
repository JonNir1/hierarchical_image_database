"""Tests for the task-v5 calibration and its cache.

The real fits take about ten minutes, which is the entire reason the cache exists, so they are
stubbed here. What is under test is the *decision* logic - when a cached calibration may be reused,
when it must not be, and when a fit must abort rather than return a number that looks fine.
"""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from SpAM_Simulations.empirical import calibrate_v5 as cal

N_PAIRS = 45


@pytest.fixture
def stub_fits(monkeypatch):
    """Replace the three fitters with counters, so reuse is observable rather than inferred."""
    calls = {"noise_population": 0, "dispersion": 0, "test_retest": 0}

    def fake_noise_population(coords, reliability, **kwargs):
        calls["noise_population"] += 1
        return {
            "best": {"family": "lognormal", "shape": 0.35, "noise_scale": 0.22, "distance": 0.0275,
                     "cv": 0.5, "at_shape_boundary": False, "at_noise_boundary": False,
                     "sim_median": 0.243, "empirical_median": 0.243, "median_gap": 0.0,
                     "noise_grid_min": 0.02, "noise_grid_max": 0.80},
            "grid": pd.DataFrame({"noise_scale": [0.22], "distance": [0.0275]}),
        }

    def fake_dispersion(coords, agreement, **kwargs):
        calls["dispersion"] += 1
        return 0.25, 0.1446

    def fake_test_retest(coords, target, **kwargs):
        calls["test_retest"] += 1
        return 0.30, 0.176

    monkeypatch.setattr(cal, "fit_noise_population", fake_noise_population)
    monkeypatch.setattr(cal, "fit_dispersion_for_agreement", fake_dispersion)
    monkeypatch.setattr(cal, "fit_noise_for_test_retest", fake_test_retest)
    monkeypatch.setattr(cal, "subject_reliability_sample",
                        lambda subjects: np.linspace(0.1, 0.4, len(subjects)))
    monkeypatch.setattr(cal, "between_subject_agreement",
                        lambda arr, **kw: {"mean_agreement": 0.1423})
    return calls


def _subjects(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return [SimpleNamespace(distances=rng.random(N_PAIRS),
                            n_obs=np.ones(N_PAIRS, dtype=np.int64)) for _ in range(n)]


def _calibrate(tmp_path, coords, subjects, **kwargs):
    return cal.calibrate(coords, subjects, images_per_trial=20, reps=3, cal_dir=tmp_path,
                         trial_simulator=None, softness=4.0, gt_file="gt.npy", n_dims=8,
                         verbose=False, **kwargs)


# --------------------------------------------------------------------- the fits

def test_calibrate_returns_the_three_constants(tmp_path, stub_fits):
    result = _calibrate(tmp_path, np.zeros((10, 3)), _subjects())
    assert result["subjects_noise_scale"] == 0.30
    assert result["dispersion"] == 0.25
    assert result["noise_family"] == "lognormal"
    assert result["noise_lognormal_sigma"] == 0.35
    assert result["noise_df"] == 5, "a lognormal family must not carry a meaningful t df"
    assert (tmp_path / "calibration.json").is_file()


def test_a_noise_scale_on_the_grid_edge_aborts(tmp_path, stub_fits, monkeypatch):
    """A grid that could not reach the data mis-calibrates everything downstream.

    This is not hypothetical: a stage-2 run used a grid whose floor sat above the optimum, pinned
    there, and produced an achieved reliability of 0.11 against a target of 0.243.
    """
    def pinned(coords, reliability, **kwargs):
        return {
            "best": {"family": "lognormal", "shape": 0.35, "noise_scale": 0.02, "distance": 0.4,
                     "cv": 0.5, "at_shape_boundary": False, "at_noise_boundary": True,
                     "sim_median": 0.11, "empirical_median": 0.243, "median_gap": -0.133,
                     "noise_grid_min": 0.02, "noise_grid_max": 0.80},
            "grid": pd.DataFrame({"noise_scale": [0.02]}),
        }

    monkeypatch.setattr(cal, "fit_noise_population", pinned)
    with pytest.raises(cal.CalibrationError, match="edge of the search grid"):
        _calibrate(tmp_path, np.zeros((10, 3)), _subjects())


def test_dispersion_sweep_clamps_to_the_fitters_search_range():
    assert cal.dispersion_sweep(0.25) == [0.1, 0.25, 0.4]
    assert min(cal.dispersion_sweep(0.05)) == 0.0, "dispersion cannot go negative"
    assert max(cal.dispersion_sweep(cal.DISP_MAX)) == cal.DISP_MAX


# --------------------------------------------------------------------- the cache

def test_an_unchanged_input_reuses_the_cache(tmp_path, stub_fits):
    coords, subjects = np.zeros((10, 3)), _subjects()
    first = _calibrate(tmp_path, coords, subjects)
    assert stub_fits["noise_population"] == 1
    second = _calibrate(tmp_path, coords, subjects)
    assert stub_fits["noise_population"] == 1, "the fits must not run again"
    assert second["subjects_noise_scale"] == first["subjects_noise_scale"]


def test_reuse_false_forces_a_refit(tmp_path, stub_fits):
    coords, subjects = np.zeros((10, 3)), _subjects()
    _calibrate(tmp_path, coords, subjects)
    _calibrate(tmp_path, coords, subjects, reuse=False)
    assert stub_fits["noise_population"] == 2


def test_a_rebuilt_gt_at_the_same_filename_invalidates_the_cache(tmp_path, stub_fits):
    """The failure a filename check would miss, and which already cost two aborted runs.

    `gt_file` is unchanged here; only the coordinates differ. Fingerprinting the contents is what
    makes that detectable.
    """
    subjects = _subjects()
    _calibrate(tmp_path, np.zeros((10, 3)), subjects)
    _calibrate(tmp_path, np.ones((10, 8)), subjects)
    assert stub_fits["noise_population"] == 2


def test_a_changed_subject_sample_invalidates_the_cache(tmp_path, stub_fits):
    coords = np.zeros((10, 3))
    _calibrate(tmp_path, coords, _subjects(n=6))
    _calibrate(tmp_path, coords, _subjects(n=7))
    assert stub_fits["noise_population"] == 2


def test_a_changed_noise_grid_invalidates_the_cache(tmp_path, stub_fits):
    coords, subjects = np.zeros((10, 3)), _subjects()
    _calibrate(tmp_path, coords, subjects)
    _calibrate(tmp_path, coords, subjects, noise_grid=(0.02, 0.04, 0.06))
    assert stub_fits["noise_population"] == 2


def test_a_changed_softness_invalidates_the_cache(tmp_path, stub_fits):
    coords, subjects = np.zeros((10, 3)), _subjects()
    _calibrate(tmp_path, coords, subjects)
    cal.calibrate(coords, subjects, images_per_trial=20, reps=3, cal_dir=tmp_path,
                  trial_simulator=None, softness=8.0, verbose=False)
    assert stub_fits["noise_population"] == 2


def test_a_corrupt_cache_is_ignored_rather_than_fatal(tmp_path, stub_fits):
    coords, subjects = np.zeros((10, 3)), _subjects()
    _calibrate(tmp_path, coords, subjects)
    (tmp_path / "calibration.json").write_text("{not json")
    _calibrate(tmp_path, coords, subjects)
    assert stub_fits["noise_population"] == 2


def test_a_calibration_without_a_fingerprint_is_never_reused(tmp_path, stub_fits):
    """Files written before the cache existed carry no fingerprint and must not be trusted."""
    (tmp_path / "calibration.json").write_text(json.dumps({"subjects_noise_scale": 0.3}))
    _calibrate(tmp_path, np.zeros((10, 3)), _subjects())
    assert stub_fits["noise_population"] == 1


def test_write_false_leaves_no_cache_behind(tmp_path, stub_fits):
    _calibrate(tmp_path, np.zeros((10, 3)), _subjects(), write=False)
    assert not (tmp_path / "calibration.json").exists()
