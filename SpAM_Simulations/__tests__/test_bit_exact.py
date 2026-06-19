"""Bit-exact validation: the refactored code must reproduce the pre-refactor
``simulate_experiment`` output byte-for-byte.

The reference fixture (``fixtures/golden_experiment.npz``) was generated against the
pre-refactor code by ``generate_golden_fixtures.py``. Here we rerun the identical
entries (same global seed, same Generator seed, same ground-truth distances) and
assert exact equality of dtype, shape, and values (NaN-aware).
"""
from pathlib import Path

import numpy as np
import pytest

import _golden_config as gc
from SpAM_Simulations.experiment import simulate_experiment

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_experiment.npz"


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"golden fixture missing: {FIXTURE_PATH} (run generate_golden_fixtures.py)")
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _assert_bit_exact(actual: np.ndarray, expected: np.ndarray, label: str):
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    assert np.array_equal(actual, expected, equal_nan=np.issubdtype(actual.dtype, np.floating)), (
        f"{label}: values differ"
    )


@pytest.mark.parametrize("combo_idx,rep,params,global_seed,rng_seed", list(gc.entries()))
def test_simulate_experiment_matches_golden(
    golden, combo_idx, rep, params, global_seed, rng_seed
):
    gt_distances = golden["gt_distances"]

    np.random.seed(global_seed)
    rng = np.random.default_rng(rng_seed)
    _, res = simulate_experiment(params, gt_distances, rng, verbose=False)

    key = gc.entry_key(combo_idx, rep)
    _assert_bit_exact(res.distances, golden[f"{key}_distances"], f"{key} distances")
    _assert_bit_exact(res.num_obs, golden[f"{key}_num_obs"], f"{key} num_obs")
    _assert_bit_exact(res.subject_noises, golden[f"{key}_subject_noises"], f"{key} subject_noises")
