"""Shared specification for the bit-exact golden-fixture validation.

Both the fixture generator (`generate_golden_fixtures.py`, run once against the
*pre-refactor* code) and the bit-exact test (`test_bit_exact.py`, run against the
*post-refactor* code) import this module so the two cannot drift apart.

Each (config, rep) entry is made independently reproducible by a fresh seeded
``np.random.default_rng(rng_seed)`` Generator, which (after the reproducibility fix)
drives BOTH the per-trial image selection (``rng.choice``) and the Gaussian
measurement noise (``rng.normal``). ``np.random.seed(global_seed)`` is still called
before each entry for backwards compatibility, but no longer affects the results.

Isolating each entry this way means the guarantee is tested at the level of
``simulate_experiment`` alone, independent of how the orchestration layer happens to
iterate over configurations.
"""
from itertools import product

from SpAM_Simulations.experiment import ExperimentParameters

# Ground-truth embedding (kept tiny so the fixture stays small and fast).
N_IMAGES = 30
N_DIMS = 4
GT_SEED = 42

# Parameter grid for the validation. Includes a zero-noise config (exercises the
# `mu_noise == 0` branch in `_draw_subject_noises`) and a heavy-tailed noisy config.
NUM_SUBJECTS = [3]
TRIALS_PER_SUBJECT = [4]
IMAGES_PER_TRIAL = [6]
SUBJECTS_NOISE_SCALE = [0.0, 0.5]
SUBJECTS_NOISE_DF = [1]
REPS = 2

# Per-entry seed bases; the stride keeps each (combo, rep) on a distinct seed.
_GLOBAL_SEED_BASE = 1000
_RNG_SEED_BASE = 2000
_SEED_STRIDE = 7


def param_combos():
    """All ExperimentParameters combinations in a deterministic order."""
    return [
        ExperimentParameters(*p)
        for p in product(
            NUM_SUBJECTS,
            TRIALS_PER_SUBJECT,
            IMAGES_PER_TRIAL,
            SUBJECTS_NOISE_SCALE,
            SUBJECTS_NOISE_DF,
        )
    ]


def entries():
    """Yield (combo_idx, rep, params, global_seed, rng_seed) for every fixture entry."""
    for combo_idx, params in enumerate(param_combos()):
        for rep in range(REPS):
            offset = _SEED_STRIDE * combo_idx + rep
            yield (
                combo_idx,
                rep,
                params,
                _GLOBAL_SEED_BASE + offset,
                _RNG_SEED_BASE + offset,
            )


def entry_key(combo_idx: int, rep: int) -> str:
    """Stable npz key prefix for a (combo, rep) entry."""
    return f"c{combo_idx}_r{rep}"
