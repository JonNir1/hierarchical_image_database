"""Tests for the ground-truth diagnostics.

These answer "is this GT a faithful summary of the data it was fitted on?", so the fixtures plant a
known answer and check the tables report it. No R and no MDS: every function here works on
coordinates that are handed in.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from SpAM_Simulations import gt_diagnostics as gd
from SpAM_Simulations import validity as val

PATHS = [
    "animate/animal/bird/a1.png",
    "animate/animal/bird/a2.png",
    "animate/animal/mammal/b1.png",
    "animate/human/face/c1.png",
    "inanimate/handmade/tool/d1.png",
    "inanimate/handmade/tool/d2.png",
]
N_PAIRS = len(PATHS) * (len(PATHS) - 1) // 2


def _subject(distances, n_obs):
    return SimpleNamespace(distances=np.asarray(distances, dtype=float),
                           n_obs=np.asarray(n_obs, dtype=np.int64))


def _cohort(n, rng, observed_frac=1.0, signal=None):
    """`n` subjects who judge a random `observed_frac` of pairs, optionally around a shared signal."""
    subs = []
    for _ in range(n):
        seen = (rng.random(N_PAIRS) < observed_frac).astype(np.int64)
        base = rng.random(N_PAIRS) if signal is None else signal + rng.normal(0, 0.01, N_PAIRS)
        subs.append(_subject(base, seen))
    return subs


# --------------------------------------------------------------------- coverage

def test_level_coverage_counts_observed_pairs_per_level():
    rng = np.random.default_rng(0)
    levels = val.hierarchy_levels(PATHS)
    # One subject who saw everything: observed_frac must be 1.0 at every level.
    table = gd.level_coverage([_subject(rng.random(N_PAIRS), np.ones(N_PAIRS))], levels)
    assert (table["observed_frac"] == 1.0).all()
    assert table["n_pairs"].sum() == N_PAIRS


def test_level_coverage_reports_a_level_nobody_judged():
    levels = val.hierarchy_levels(PATHS)
    n_obs = np.ones(N_PAIRS, dtype=np.int64)
    n_obs[levels == 3] = 0            # nobody judged any same-leaf pair
    table = gd.level_coverage([_subject(np.random.default_rng(1).random(N_PAIRS), n_obs)], levels)
    leaf = table[table["level"] == 3].iloc[0]
    assert leaf["n_observed"] == 0 and leaf["observed_frac"] == 0.0
    assert leaf["n_pairs"] > 0, "the pairs exist, they are merely unjudged"


# --------------------------------------------------------------------- gradient

def test_gt_gradient_reads_the_embedding_not_the_subjects():
    """A GT whose geometry encodes the hierarchy is monotone regardless of who judged what."""
    levels = val.hierarchy_levels(PATHS)
    # Place each leaf's images together and push the categories apart along one axis.
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [5.0, 0.0], [20.0, 0.0], [20.1, 0.0]])
    table = gd.gt_gradient(coords, levels)
    assert val.gradient_is_monotone(table)


# --------------------------------------------------------------------- ceiling

def test_noise_ceiling_is_high_when_subjects_agree():
    rng = np.random.default_rng(3)
    levels = val.hierarchy_levels(PATHS)
    shared = rng.random(N_PAIRS)
    table = gd.raw_noise_ceiling(_cohort(12, rng, signal=shared), levels, n_splits=5,
                                 rng=np.random.default_rng(4))
    assert (table["ceiling_half"] > 0.8).all(), "near-identical subjects must agree with themselves"


def test_noise_ceiling_is_near_zero_when_subjects_are_independent():
    rng = np.random.default_rng(5)
    levels = val.hierarchy_levels(PATHS)
    table = gd.raw_noise_ceiling(_cohort(16, rng), levels, n_splits=8, rng=np.random.default_rng(6))
    assert table["ceiling_half"].abs().max() < 0.6, "unrelated judgements cannot self-agree strongly"


def test_spearman_brown_lifts_the_half_split_estimate():
    rng = np.random.default_rng(7)
    levels = val.hierarchy_levels(PATHS)
    shared = rng.random(N_PAIRS)
    table = gd.raw_noise_ceiling(_cohort(12, rng, signal=shared), levels, n_splits=4,
                                 rng=np.random.default_rng(8))
    positive = table[table["ceiling_half"] > 0]
    assert (positive["ceiling_full"] >= positive["ceiling_half"]).all()


def test_negative_half_split_is_not_projected_into_a_confident_number():
    """Spearman-Brown on a negative r produces nonsense, so it must be passed through instead."""
    table = gd.raw_noise_ceiling(_cohort(8, np.random.default_rng(9)),
                                 val.hierarchy_levels(PATHS), n_splits=6,
                                 rng=np.random.default_rng(10))
    negative = table[table["ceiling_half"] < 0]
    assert (negative["ceiling_full"] == negative["ceiling_half"]).all()


def test_noise_ceiling_needs_enough_subjects_to_split():
    with pytest.raises(ValueError, match="at least 4 subjects"):
        gd.raw_noise_ceiling(_cohort(2, np.random.default_rng(11)), val.hierarchy_levels(PATHS))


# --------------------------------------------------------------------- diagnose

def test_diagnose_flags_a_gt_that_fits_beyond_the_ceiling():
    """The signature of an overfitted GT: agreement with the aggregate above the data's own.

    This is the case that matters. A 725xD embedding fitted to a sparsely observed matrix can
    reproduce variance that the data cannot reproduce in itself, and reading `gt_vs_raw` alone would
    call that success.
    """
    rng = np.random.default_rng(12)
    levels = val.hierarchy_levels(PATHS)
    subjects = _cohort(10, rng, observed_frac=0.5)        # independent judgements: no real signal
    mean, _ = gd.aggregate_subjects(subjects)
    # A "GT" built to match the aggregate exactly - the extreme of fitting the noise.
    from scipy.spatial.distance import squareform
    coords = np.linalg.eigh(squareform(mean))[1][:, -2:]
    tables = gd.diagnose(coords, subjects, PATHS, n_splits=6, rng=np.random.default_rng(13))
    assert set(tables) == {"level_coverage", "gt_gradient", "gt_vs_raw", "noise_ceiling"}
    assert "frac_of_ceiling" in tables["gt_vs_raw"].columns


def test_diagnose_rejects_a_manifest_that_does_not_match_the_gt():
    rng = np.random.default_rng(14)
    with pytest.raises(ValueError, match="hierarchy levels would be misaligned"):
        gd.diagnose(rng.random((len(PATHS) + 1, 2)), _cohort(6, rng), PATHS)
