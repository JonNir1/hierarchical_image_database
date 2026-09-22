"""Tests for analysis.behavioral.reliability's data-generation functions.

Uses a small synthetic multi-participant SpAM dataset (mirroring
analysis/rdms/__tests__/test_behavioral.py's synthetic-fixture pattern, scaled up
to enough participants for leave-one-out/split-half to be meaningful) so this
suite needs neither a real 725-image dataset nor real SpAM session data.

Scoped to reliability.py's data-generation functions only (_kendall_spearman,
loo_reliability, split_half_reliability, make_splits) -- the pairwise Wilcoxon
comparisons and the plotly figure live in noise_ceiling.ipynb and are not
unit-tested, consistent with how this repo's other analysis notebooks aren't either.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, spearmanr

import analysis.behavioral.reliability as reliability

# 6 synthetic images -> C(6,2) = 15 pairs
_IMAGES = ["img0.png", "img1.png", "img2.png", "img3.png", "img4.png", "img5.png"]
_N_PARTICIPANTS = 8


def _pairwise_json(pairs: dict[tuple[str, str], float]) -> str:
    items = [
        {"src1": f"./images/pre_shine/{a}", "src2": f"./images/pre_shine/{b}", "distance": d}
        for (a, b), d in pairs.items()
    ]
    return json.dumps(items)


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
    directly into reliability.py's (indices, values) sparse-trial format (bypassing
    the JSON/DataFrame layer, since this suite targets the LOO/split-half math, not
    the trial-parsing wiring already covered by test_behavioral.py)."""
    from analysis.rdms.behavioral import _condensed_index

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


@pytest.fixture()
def trials_by_participant():
    return _synthetic_trials_by_participant()


# ---------------------------------------------------------------------------
# _kendall_spearman
# ---------------------------------------------------------------------------

def test_kendall_spearman_masks_to_jointly_covered_pairs():
    rdm_a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    rdm_b = np.array([1.0, np.nan, 3.0, 4.0, 6.0])
    result = reliability._kendall_spearman(rdm_a, rdm_b, min_overlap=2)

    mask = np.array([True, False, False, True, True])
    assert result["n_overlap"] == 3
    assert not result["underpowered"]
    expected_tau = kendalltau(rdm_a[mask], rdm_b[mask]).statistic
    expected_rho = spearmanr(rdm_a[mask], rdm_b[mask]).statistic
    assert result["kendall"] == pytest.approx(expected_tau)
    assert result["spearman"] == pytest.approx(expected_rho)


def test_kendall_spearman_flags_underpowered_folds():
    rdm_a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    rdm_b = np.array([1.0, np.nan, 3.0, 4.0, 6.0])
    result = reliability._kendall_spearman(rdm_a, rdm_b, min_overlap=5)  # only 3 jointly covered

    assert result["n_overlap"] == 3
    assert result["underpowered"]
    assert np.isnan(result["kendall"])
    assert np.isnan(result["spearman"])


# ---------------------------------------------------------------------------
# loo_reliability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aggregation", ["mean", "simple", "evidence"])
def test_loo_reliability_returns_exactly_n_rows(trials_by_participant, aggregation):
    n_conds = len(_IMAGES)
    result = reliability.loo_reliability(
        "pre_shine", aggregation, trials_by_participant=trials_by_participant,
        n_conds=n_conds, min_overlap=3,
    )
    assert len(result) == _N_PARTICIPANTS
    assert set(result["fold_id"]) == set(trials_by_participant.keys())
    assert not result["underpowered"].any()


# ---------------------------------------------------------------------------
# make_splits / split_half_reliability
# ---------------------------------------------------------------------------

def test_make_splits_is_deterministic_given_a_seed():
    ids = [f"p{i}" for i in range(_N_PARTICIPANTS)]
    splits_a = reliability.make_splits(ids, n_splits=10, seed=42)
    splits_b = reliability.make_splits(ids, n_splits=10, seed=42)
    assert splits_a == splits_b


def test_make_splits_halves_are_disjoint_and_cover_all_participants():
    ids = [f"p{i}" for i in range(_N_PARTICIPANTS)]
    splits = reliability.make_splits(ids, n_splits=5, seed=0)
    for half_a, half_b in splits:
        assert set(half_a) & set(half_b) == set()
        assert set(half_a) | set(half_b) == set(ids)
        assert len(half_a) == _N_PARTICIPANTS // 2


@pytest.mark.parametrize("aggregation", ["mean", "simple", "evidence"])
def test_split_half_reliability_returns_exactly_n_splits_rows(trials_by_participant, aggregation):
    n_conds = len(_IMAGES)
    ids = list(trials_by_participant.keys())
    splits = reliability.make_splits(ids, n_splits=7, seed=1)
    result = reliability.split_half_reliability(
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
    splits = reliability.make_splits(ids, n_splits=5, seed=3)

    result_mean = reliability.split_half_reliability(
        "pre_shine", "mean", trials_by_participant=trials_by_participant,
        n_conds=n_conds, splits=splits, min_overlap=3,
    )
    result_evidence = reliability.split_half_reliability(
        "pre_shine", "evidence", trials_by_participant=trials_by_participant,
        n_conds=n_conds, splits=splits, min_overlap=3,
    )
    # Same splits object -> same fold_id -> split-partition correspondence for both calls.
    assert list(result_mean["fold_id"]) == list(result_evidence["fold_id"])
    # Re-deriving the splits with the same seed must match what was actually used.
    assert splits == reliability.make_splits(ids, n_splits=5, seed=3)
