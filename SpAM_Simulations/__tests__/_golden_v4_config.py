"""Shared specification for the task-v4 bit-exact golden-fixture validation.

The existing `_golden_config.py` / `test_bit_exact.py` pair pins `experiment.simulate_experiment`
(the task-v0.1 model) and was recorded against the pre-vectorisation code. Nothing guards the
task-v4 RNG stream, yet every result in `sim_results/` was produced by it, so any reordering of
`rng` calls silently breaks comparability with those runs while every test still passes.

This module is the v4 analogue, recorded against the code as it stood *before* the allocation
refactor. Its job is narrow: prove that the default (random) allocation path is bit-for-bit
unchanged after `allocation_mode` is introduced.

Both the generator (`generate_golden_v4_fixtures.py`) and the checker (`test_bit_exact_v4.py`)
import this module so the two cannot drift apart.

The grid deliberately spans the branch points that matter:
* `subjects_noise_scale` 0.0 exercises the zero-noise short-circuit in the noise draw;
* `(screening_trials, screening_repeats)` of (0, 0) is the v3-equivalent path where the image pool
  is drawn inside the single-subject function, while (2, 1) takes the partitioned two-stage path;
* `screening_min_reliability` -1.0 admits everyone, 0.0 actually rejects candidates and so
  exercises the retry loop that consumes extra RNG draws;
* `subjects_noise_lognormal_sigma` 0.0 vs 0.3 selects the |t(df)| and lognormal noise populations.
"""
from itertools import product

from SpAM_Simulations.allocation import RANDOM
from SpAM_Simulations.task_v4_experiment import TaskV4ExperimentParameters

# Kept tiny so the fixture stays small and the suite stays fast.
N_IMAGES = 40
N_DIMS = 4
GT_SEED = 42

NUM_SUBJECTS = [3]
TRIALS_PER_SUBJECT = [4]        # -> t_distinct 3, n_repeats 1 at frac_trials_repeated 0.25
IMAGES_PER_TRIAL = [5]          # main stage needs 15 images, screening 5, of 40 available
SUBJECTS_NOISE_SCALE = [0.0, 0.5]
SUBJECTS_NOISE_DF = [5]
FRAC_TRIALS_REPEATED = [0.25]
PERSPECTIVE_DISPERSION = [0.2]
# (screening_trials, screening_repeats): the invalid pairings are excluded by construction, since
# `screening_repeats > screening_trials - 1` is rejected by the config validator.
SCREENING = [(0, 0), (2, 1)]
SCREENING_MIN_RELIABILITY = [-1.0, 0.0]
SUBJECTS_NOISE_LOGNORMAL_SIGMA = [0.0, 0.3]
REPS = 2

_GLOBAL_SEED_BASE = 3000
_RNG_SEED_BASE = 4000
_SEED_STRIDE = 7


def param_combos():
    """All TaskV4ExperimentParameters combinations in a deterministic order."""
    combos = []
    for (ns, tps, ipt, noise, df, ftr, disp, (st, sr), minrel, sigma) in product(
        NUM_SUBJECTS, TRIALS_PER_SUBJECT, IMAGES_PER_TRIAL, SUBJECTS_NOISE_SCALE,
        SUBJECTS_NOISE_DF, FRAC_TRIALS_REPEATED, PERSPECTIVE_DISPERSION, SCREENING,
        SCREENING_MIN_RELIABILITY, SUBJECTS_NOISE_LOGNORMAL_SIGMA,
    ):
        # allocation_mode is pinned to RANDOM: the fixture was recorded before that field existed,
        # so it is only meaningful as a guard on the default arm. The whole point is that adding
        # the field must NOT change these arrays - the fixture is never regenerated for it.
        combos.append(TaskV4ExperimentParameters(
            ns, tps, ipt, noise, df, ftr, disp, st, sr, minrel, sigma, RANDOM))
    return combos


def entries():
    """Yield (combo_idx, rep, params, global_seed, rng_seed) for every fixture entry."""
    for combo_idx, params in enumerate(param_combos()):
        for rep in range(REPS):
            offset = _SEED_STRIDE * combo_idx + rep
            yield combo_idx, rep, params, _GLOBAL_SEED_BASE + offset, _RNG_SEED_BASE + offset


def entry_key(combo_idx: int, rep: int) -> str:
    """Stable npz key prefix for a (combo, rep) entry."""
    return f"v4_c{combo_idx}_r{rep}"
