import json

import numpy as np
import pytest
from scipy.stats import ks_2samp

from analysis.utils.null_distances import simulate, trial_ks_distance


def _pw_json(distances: list[float]) -> str:
    return json.dumps([
        {"src1": f"a{i}", "src2": f"b{i}", "distance": d} for i, d in enumerate(distances)
    ])


def test_simulate_shape_range_and_seed():
    # Shape is T * C(K, 2), values are diagonal-normalised (<= 1), same seed reproduces.
    out = simulate(num_dots=6, num_trials=50, seed=3)
    assert out.shape == (50 * 15,)
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_array_equal(out, simulate(num_dots=6, num_trials=50, seed=3))
    assert not np.array_equal(out, simulate(num_dots=6, num_trials=50, seed=4))


def test_simulate_mean_matches_unit_square_expectation():
    # Mean distance between two uniform points in the unit square is ~0.5214, divided by sqrt(2).
    out = simulate(num_dots=20, num_trials=2000, seed=0)
    assert out.mean() == pytest.approx(0.5214 / np.sqrt(2), abs=0.005)


def test_trial_ks_distance_matches_scipy():
    # Delegates to ks_2samp on the trial's parsed distances; far-from-null trials score near 1.
    null = simulate(num_dots=20, num_trials=200, seed=1)
    dists = [0.01, 0.02, 0.03, 0.04, 0.05]
    expected, _ = ks_2samp(dists, null)
    assert trial_ks_distance(_pw_json(dists), null) == pytest.approx(expected)
    assert trial_ks_distance(_pw_json(dists), null) > 0.9


def test_trial_ks_distance_nan_for_fewer_than_two_distances():
    # A trial with 0 or 1 parsable distances has no distribution to compare.
    null = simulate(num_dots=10, num_trials=20, seed=1)
    assert np.isnan(trial_ks_distance(_pw_json([0.3]), null))
    assert np.isnan(trial_ks_distance("", null))
