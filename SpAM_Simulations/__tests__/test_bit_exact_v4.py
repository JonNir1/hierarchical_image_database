"""Bit-exact validation for the task-v4 model: the default allocation path must not move.

`test_bit_exact.py` pins `experiment.simulate_experiment` (task-v0.1) only, so nothing guarded the
task-v4 RNG stream even though every run in `sim_results/` came from it. Introducing an
`allocation_mode` lever is exactly the kind of change that can silently reorder `rng` calls and
invalidate comparability with those runs while leaving every other test green.

The reference fixture (`fixtures/golden_task_v4.npz`) was recorded by
`generate_golden_v4_fixtures.py` against the code as it stood before the allocation refactor. Here
we rerun the identical entries and assert exact equality.

A failure here is not a flaky test: it means the random-allocation arm no longer reproduces the
published runs, and either the change must be reworked or the existing results must be regenerated.
"""
from pathlib import Path

import numpy as np
import pytest

import _golden_v4_config as gc
from SpAM_Simulations.task_v4_experiment import simulate_task_v4_experiment

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_task_v4.npz"


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"golden fixture missing: {FIXTURE_PATH} (run generate_golden_v4_fixtures.py)")
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _assert_bit_exact(actual: np.ndarray, expected: np.ndarray, label: str):
    actual = np.asarray(actual)
    assert actual.dtype == expected.dtype, f"{label}: dtype {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    assert np.array_equal(actual, expected, equal_nan=np.issubdtype(actual.dtype, np.floating)), (
        f"{label}: values differ"
    )


@pytest.mark.parametrize("combo_idx,rep,params,global_seed,rng_seed", list(gc.entries()))
def test_simulate_task_v4_matches_golden(golden, combo_idx, rep, params, global_seed, rng_seed):
    gt = golden["gt_embeddings"]

    np.random.seed(global_seed)
    rng = np.random.default_rng(rng_seed)
    _, res = simulate_task_v4_experiment(params, gt, rng, verbose=False)

    key = gc.entry_key(combo_idx, rep)
    _assert_bit_exact(res.distances, golden[f"{key}_distances"], f"{key} distances")
    _assert_bit_exact(res.num_obs, golden[f"{key}_num_obs"], f"{key} num_obs")
    _assert_bit_exact(res.subject_noises, golden[f"{key}_subject_noises"], f"{key} subject_noises")
    _assert_bit_exact(res.subject_test_retest, golden[f"{key}_test_retest"], f"{key} test_retest")
    # The screening retry loop draws extra noise values, so the candidate count witnesses the
    # stream independently of the arrays above.
    _assert_bit_exact(np.asarray(res.n_candidates_screened),
                      golden[f"{key}_n_candidates"], f"{key} n_candidates")
