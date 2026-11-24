"""Tests for analysis.s0_build_rdms.behavior.noise_ceiling's data-generation functions.

Uses a small synthetic multi-participant SpAM dataset (mirroring
analysis/s0_build_rdms/__tests__/test_spam_ratings.py's synthetic-fixture pattern, scaled up
to enough participants for leave-one-out/split-half to be meaningful) so this
suite needs neither a real 725-image dataset nor real SpAM session data.

Scoped to noise_ceiling.py's data-generation functions only (loo_reliability,
split_half_reliability, make_splits) -- the pairwise Wilcoxon comparisons and
the plotly figure live in aggregation_comparison.ipynb and are not unit-tested,
consistent with how this repo's other analysis notebooks aren't either.
Masked-correlation coverage (formerly this module's own _kendall_spearman)
moved to analysis/utils/__tests__/test_stats.py's masked_rank_correlation
tests, alongside the shared function noise_ceiling.py now calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analysis.s0_build_rdms.behavior.noise_ceiling as noise_ceiling

# 6 synthetic images -> C(6,2) = 15 pairs
_IMAGES = ["img0.png", "img1.png", "img2.png", "img3.png", "img4.png", "img5.png"]
_N_PARTICIPANTS = 8


def _image_to_idx() -> dict[str, int]:
    return {img: i for i, img in enumerate(_IMAGES)}


def _trial_dict(images: list[str], base: float, rng: np.random.Generator) -> dict:
    pairs = {}
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs[(images[i], images[j])] = float(base + rng.uniform(0, 0.05))
    return pairs


def _synthetic_trials_by_participant() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """8 synthetic participants, each with 3 trials of 4 overlapping images, parsed
    directly into noise_ceiling.py's (indices, values) sparse-trial format (bypassing
    the JSON/DataFrame layer, since this suite targets the LOO/split-half math, not
    the trial-parsing wiring already covered by test_spam_ratings.py)."""
    from analysis.s0_build_rdms.behavior.spam_ratings import _condensed_index

    n_conds = len(_IMAGES)
    img2idx = _image_to_idx()
    rng = np.random.default_rng(0)
    # Each participant's 3 trials cycle through overlapping 4-image windows so
    # cross-participant pair overlap is ample (unlike the real task's own design).
    windows = [
        ["img0.png", "img1.png", "img2.png", "img3.png"],
        ["img1.png", "img2.png", "img3.png", "img4.png"],
        ["img2.png", "img3.png", "img4.png", "img5.png"],
    ]
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for p in range(_N_PARTICIPANTS):
        pid = f"p{p}"
        trials = []
        for images in windows:
            pairs = _trial_dict(images, base=0.1 + 0.01 * p, rng=rng)
            pair_i = np.array([img2idx[a] for (a, b) in pairs], dtype=np.int64)
            pair_j = np.array([img2idx[b] for (a, b) in pairs], dtype=np.int64)
            values = np.array(list(pairs.values()), dtype=np.float64)
            lo = np.minimum(pair_i, pair_j)
            hi = np.maximum(pair_i, pair_j)
            indices = _condensed_index(lo, hi, n_conds)
            trials.append((indices, values))
        out[pid] = trials
    return out


@pytest.fixture(scope="module")
def trials_by_participant():
    return _synthetic_trials_by_participant()


# ---------------------------------------------------------------------------
# loo_reliability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aggregation", ["mean", "simple", "evidence"])
def test_loo_reliability_returns_exactly_n_rows(trials_by_participant, aggregation):
    n_conds = len(_IMAGES)
    result = noise_ceiling.loo_reliability(
        "pre_shine", aggregation, trials_by_participant=trials_by_participant,
        n_conds=n_conds, min_overlap=3,
    )
    assert len(result) == _N_PARTICIPANTS
    assert set(result["fold_id"]) == set(trials_by_participant.keys())
    assert not result["underpowered"].any()


def test_nili_upper_counts_each_participants_share_and_is_not_below_the_lower_bound(trials_by_participant):
    """Every synthetic participant rates the same pairs equally often, so each owns 1/8 of every entry."""
    n_conds = len(_IMAGES)
    upper = noise_ceiling.nili_upper_reliability(
        "mean", trials_by_participant=trials_by_participant, n_conds=n_conds, min_overlap=3,
    )
    lower = noise_ceiling.loo_reliability(
        "pre_shine", "mean", trials_by_participant=trials_by_participant, n_conds=n_conds, min_overlap=3,
    )
    assert np.allclose(upper["self_share"], 1 / _N_PARTICIPANTS)
    assert upper["spearman"].mean() >= lower["spearman"].mean()


def test_group_size_reliability_draws_disjoint_groups_per_size(trials_by_participant):
    n_conds = len(_IMAGES)
    table = noise_ceiling.group_size_reliability(
        "mean", trials_by_participant=trials_by_participant, n_conds=n_conds, sizes=(2, 4), n_splits=3, min_overlap=3,
    )
    assert list(table["group_size"]) == [2, 2, 2, 4, 4, 4]
    assert {"kendall", "kendall_a", "spearman"} <= set(table.columns)
    with pytest.raises(ValueError, match="exceed half"):
        noise_ceiling.group_size_reliability(
            "mean", trials_by_participant=trials_by_participant, n_conds=n_conds, sizes=(5,), n_splits=1,
        )


# ---------------------------------------------------------------------------
# _spearman_brown / _add_ceiling_full
# ---------------------------------------------------------------------------

def test_spearman_brown_matches_closed_form():
    assert noise_ceiling._spearman_brown(0.2) == pytest.approx(2 * 0.2 / 1.2)
    assert noise_ceiling._spearman_brown(0.19) == pytest.approx(2 * 0.19 / 1.19)


@pytest.mark.parametrize("r", [0.0, -0.1, -1.0, np.nan])
def test_spearman_brown_passes_through_non_positive_or_nonfinite(r):
    """Mirrors gt_diagnostics.raw_noise_ceiling()'s guard: a non-positive half-split
    correlation makes the projection meaningless, so it's passed through unchanged."""
    result = noise_ceiling._spearman_brown(r)
    if np.isnan(r):
        assert np.isnan(result)
    else:
        assert result == r


def test_add_ceiling_full_only_populated_for_upper_spearman():
    long = pd.DataFrame([
        {"bound": "lower", "metric": "kendall", "value": 0.13},
        {"bound": "lower", "metric": "spearman", "value": 0.19},
        {"bound": "upper", "metric": "kendall", "value": 0.13},
        {"bound": "upper", "metric": "spearman", "value": 0.19},
    ])
    result = noise_ceiling._add_ceiling_full(long)

    assert np.isnan(result.loc[0, "ceiling_full"])  # lower/kendall
    assert np.isnan(result.loc[1, "ceiling_full"])  # lower/spearman
    assert np.isnan(result.loc[2, "ceiling_full"])  # upper/kendall (not well-justified)
    assert result.loc[3, "ceiling_full"] == pytest.approx(2 * 0.19 / 1.19)  # upper/spearman


# ---------------------------------------------------------------------------
# make_splits / split_half_reliability
# ---------------------------------------------------------------------------

def test_make_splits_is_deterministic_given_a_seed():
    ids = [f"p{i}" for i in range(_N_PARTICIPANTS)]
    splits_a = noise_ceiling.make_splits(ids, n_splits=10, seed=42)
    splits_b = noise_ceiling.make_splits(ids, n_splits=10, seed=42)
    assert splits_a == splits_b


def test_make_splits_halves_are_disjoint_and_cover_all_participants():
    ids = [f"p{i}" for i in range(_N_PARTICIPANTS)]
    splits = noise_ceiling.make_splits(ids, n_splits=5, seed=0)
    for half_a, half_b in splits:
        assert set(half_a) & set(half_b) == set()
        assert set(half_a) | set(half_b) == set(ids)
        assert len(half_a) == _N_PARTICIPANTS // 2


@pytest.mark.parametrize("aggregation", ["mean", "simple", "evidence"])
def test_split_half_reliability_returns_exactly_n_splits_rows(trials_by_participant, aggregation):
    n_conds = len(_IMAGES)
    ids = list(trials_by_participant.keys())
    splits = noise_ceiling.make_splits(ids, n_splits=7, seed=1)
    result = noise_ceiling.split_half_reliability(
        "pre_shine", aggregation, trials_by_participant=trials_by_participant,
        n_conds=n_conds, splits=splits, min_overlap=3,
    )
    assert len(result) == 7
    assert list(result["fold_id"]) == list(range(7))


def test_split_half_reliability_reuses_identical_splits_across_aggregations(trials_by_participant):
    """The same `splits` argument passed to two different aggregation methods must
    produce results derived from identical participant-id partitions -- the paired
    design §6.2 of the plan relies on this, not just on both calls happening to use
    the same seed."""
    n_conds = len(_IMAGES)
    ids = list(trials_by_participant.keys())
    splits = noise_ceiling.make_splits(ids, n_splits=5, seed=3)

    result_mean = noise_ceiling.split_half_reliability(
        "pre_shine", "mean", trials_by_participant=trials_by_participant,
        n_conds=n_conds, splits=splits, min_overlap=3,
    )
    result_evidence = noise_ceiling.split_half_reliability(
        "pre_shine", "evidence", trials_by_participant=trials_by_participant,
        n_conds=n_conds, splits=splits, min_overlap=3,
    )
    # Same splits object -> same fold_id -> split-partition correspondence for both calls.
    assert list(result_mean["fold_id"]) == list(result_evidence["fold_id"])
    # Re-deriving the splits with the same seed must match what was actually used.
    assert splits == noise_ceiling.make_splits(ids, n_splits=5, seed=3)
