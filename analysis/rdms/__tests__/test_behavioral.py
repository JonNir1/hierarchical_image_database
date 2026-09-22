"""Tests for analysis.rdms.behavioral: the SpAM multi-arrangement RDM builder.

Uses a tiny synthetic 4-image manifest and hand-built partial trials (patching
load_manifest/load_data/save_rdm in the module's own namespace, mirroring the
sensory/semantic_km test patterns in test_common.py) so this suite needs
neither a real 725-image dataset nor real SpAM session data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import rsatoolbox
from rsatoolbox.rdm.combine import rescale as rsatoolbox_rescale
from rsatoolbox.rdm.combine import from_partials
from scipy.spatial.distance import squareform

import analysis.rdms.behavioral as behavioral

# 4 synthetic images -> C(4,2) = 6 pairs
_IMAGES = ["img0.png", "img1.png", "img2.png", "img3.png"]


def _pairwise_json(pairs: dict[tuple[str, str], float], variant_dir: str) -> str:
    items = [
        {"src1": f"./images/{variant_dir}/{a}", "src2": f"./images/{variant_dir}/{b}", "distance": d}
        for (a, b), d in pairs.items()
    ]
    return json.dumps(items)


def _synthetic_manifest() -> pd.DataFrame:
    return pd.DataFrame({"curated_path": _IMAGES})


def _synthetic_participants() -> pd.DataFrame:
    return pd.DataFrame([
        {"participant_id": "p1", "status": "full data", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "p2", "status": "full data", "shine_variant": "pre", "task_version": 4.01},
        {"participant_id": "p3", "status": "screened out", "shine_variant": "pre", "task_version": 4.0},
        {"participant_id": "p4", "status": "full data", "shine_variant": "post", "task_version": 4.0},
        {"participant_id": "p5", "status": "full data", "shine_variant": "pre", "task_version": 3.06},
    ])


def _synthetic_trials() -> pd.DataFrame:
    rows = [
        # p1, screening block: img0-img1-img2 triangle
        {
            "participant_id": "p1", "is_catch": False, "block_type": "screening",
            "pairwise_distances": _pairwise_json(
                {("img0.png", "img1.png"): 0.1, ("img0.png", "img2.png"): 0.2,
                 ("img1.png", "img2.png"): 0.3},
                "pre_shine",
            ),
        },
        # p1, experimental block: img1-img2-img3 triangle (overlaps p1's own screening trial on img1/img2)
        {
            "participant_id": "p1", "is_catch": False, "block_type": "experimental",
            "pairwise_distances": _pairwise_json(
                {("img1.png", "img2.png"): 0.32, ("img1.png", "img3.png"): 0.4,
                 ("img2.png", "img3.png"): 0.5},
                "pre_shine",
            ),
        },
        # p1 catch trial: must be excluded regardless of its content
        {
            "participant_id": "p1", "is_catch": True, "block_type": "experimental",
            "pairwise_distances": _pairwise_json({("img0.png", "img1.png"): 0.99}, "pre_shine"),
        },
        # p2, experimental block: same img1-img2-img3 triangle, independent judgement
        {
            "participant_id": "p2", "is_catch": False, "block_type": "experimental",
            "pairwise_distances": _pairwise_json(
                {("img1.png", "img2.png"): 0.28, ("img1.png", "img3.png"): 0.42,
                 ("img2.png", "img3.png"): 0.48},
                "pre_shine",
            ),
        },
        # p3 (screened out) -- must be excluded even though it has trial data
        {
            "participant_id": "p3", "is_catch": False, "block_type": "screening",
            "pairwise_distances": _pairwise_json({("img0.png", "img3.png"): 0.6}, "pre_shine"),
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def patched_behavioral(monkeypatch):
    monkeypatch.setattr(behavioral, "load_manifest", _synthetic_manifest)
    monkeypatch.setattr(
        behavioral, "load_data",
        lambda cache_dir, prod_only=False: {
            "participants": _synthetic_participants(), "trials": _synthetic_trials(),
        },
    )
    saved = {}

    def fake_save_rdm(name, condensed, *, metric, source, extra=None):
        saved["name"] = name
        saved["condensed"] = condensed
        saved["extra"] = extra or {}

    monkeypatch.setattr(behavioral, "save_rdm", fake_save_rdm)
    return saved


# ---------------------------------------------------------------------------
# _curated_path_from_src
# ---------------------------------------------------------------------------

def test_curated_path_from_src_strips_prefix():
    src = "./images/pre_shine/inanimate/handmade/kitchen/bottle1.png"
    assert behavioral._curated_path_from_src(src, "pre_shine") == "inanimate/handmade/kitchen/bottle1.png"


def test_curated_path_from_src_missing_marker_raises():
    with pytest.raises(ValueError, match="Could not find"):
        behavioral._curated_path_from_src("./images/post_shine/x.png", "pre_shine")


# ---------------------------------------------------------------------------
# _select_participants
# ---------------------------------------------------------------------------

def test_select_participants_filters_status_variant_and_version():
    selected = behavioral._select_participants(_synthetic_participants(), "pre_shine")
    assert sorted(selected["participant_id"]) == ["p1", "p2"]


def test_select_participants_rejects_variant_without_shine_suffix():
    with pytest.raises(ValueError, match="must end with '_shine'"):
        behavioral._select_participants(_synthetic_participants(), "pre")


# ---------------------------------------------------------------------------
# _condensed_index
# ---------------------------------------------------------------------------

def test_condensed_index_matches_squareform():
    """_condensed_index must agree with scipy squareform's own condensed ordering."""
    n = 6
    sq = squareform(np.arange(n * (n - 1) // 2), checks=False)
    for i in range(n):
        for j in range(i + 1, n):
            idx = behavioral._condensed_index(np.array([i]), np.array([j]), n)[0]
            assert sq[i, j] == idx


# ---------------------------------------------------------------------------
# _trial_sparse
# ---------------------------------------------------------------------------

def test_trial_sparse_none_for_empty_pairwise():
    image_to_idx = {img: i for i, img in enumerate(_IMAGES)}
    assert behavioral._trial_sparse("", "pre_shine", image_to_idx, 4) is None
    assert behavioral._trial_sparse(None, "pre_shine", image_to_idx, 4) is None


def test_trial_sparse_indices_and_values():
    image_to_idx = {img: i for i, img in enumerate(_IMAGES)}
    pw = _pairwise_json({("img0.png", "img1.png"): 0.1, ("img0.png", "img2.png"): 0.2,
                          ("img1.png", "img2.png"): 0.3}, "pre_shine")
    indices, values = behavioral._trial_sparse(pw, "pre_shine", image_to_idx, 4)
    got = dict(zip(indices.tolist(), values.tolist()))
    # condensed order for n=4: (0,1)->0, (0,2)->1, (0,3)->2, (1,2)->3, (1,3)->4, (2,3)->5
    assert got == {0: 0.1, 1: 0.2, 3: 0.3}


# ---------------------------------------------------------------------------
# _sparse_combine vs. rsatoolbox's own dense from_partials()/rescale()/mean()
#
# _sparse_combine() reimplements that pipeline to avoid materializing a dense
# (n_trials, n_pairs) array (see the module docstring for why). This is the
# equivalence check that justifies doing so: it must reproduce rsatoolbox's own
# reference output, up to floating-point precision, on genuinely overlapping
# partial arrangements (not just the trivial single-trial case).
# ---------------------------------------------------------------------------

def _rsatoolbox_dense_combine(trial_dicts, images, method):
    rdms = []
    for d in trial_dicts:
        imgs = sorted({im for pair in d for im in pair})
        idx = {im: i for i, im in enumerate(imgs)}
        n = len(imgs)
        # NaN-fill (not zero-fill): a trial dict here may legitimately reference an
        # image without giving every pair among that trial's images (see the "extra
        # overlapping observation" trial below, which touches 3 images via only 2
        # pairs) -- zero-filling would silently fabricate a 0.0 "observation" for
        # the missing (b, c) pair instead of leaving it genuinely unobserved.
        mat = np.full((n, n), np.nan)
        np.fill_diagonal(mat, 0.0)
        for (a, b), dist in d.items():
            i, j = idx[a], idx[b]
            mat[i, j] = mat[j, i] = dist
        rdms.append(rsatoolbox.rdm.RDMs(dissimilarities=mat[np.newaxis], pattern_descriptors={"conds": imgs}))
    combined = from_partials(rdms, all_patterns=images, descriptor="conds")
    aligned = rsatoolbox_rescale(combined, method=method)
    final = aligned.mean(weights="rescalingWeights")
    return final.dissimilarities[0]


@pytest.mark.parametrize("method", ["evidence", "setsize", "simple"])
def test_sparse_combine_matches_rsatoolbox_dense(method):
    images = ["a", "b", "c", "d", "e"]
    image_to_idx = {im: i for i, im in enumerate(images)}
    trial_dicts = [
        {("a", "b"): 0.10, ("a", "c"): 0.40, ("b", "c"): 0.70},
        {("b", "c"): 0.35, ("b", "d"): 0.50, ("c", "d"): 0.65},
        {("c", "d"): 0.60, ("c", "e"): 0.20, ("d", "e"): 0.80},
        {("a", "b"): 0.15, ("a", "c"): 0.42},  # extra overlapping observation
    ]
    n_conds = len(images)

    sparse_trials = []
    for d in trial_dicts:
        pair_i, pair_j, values = [], [], []
        for (a, b), dist in d.items():
            ia, ib = image_to_idx[a], image_to_idx[b]
            i, j = (ia, ib) if ia < ib else (ib, ia)
            pair_i.append(i)
            pair_j.append(j)
            values.append(dist)
        indices = behavioral._condensed_index(np.array(pair_i), np.array(pair_j), n_conds)
        sparse_trials.append((indices, np.array(values, dtype=np.float64)))

    sparse_result, _ = behavioral._sparse_combine(sparse_trials, n_conds, method=method)
    dense_result = _rsatoolbox_dense_combine(trial_dicts, images, method)

    np.testing.assert_allclose(sparse_result, dense_result, rtol=1e-6, atol=1e-8, equal_nan=True)


# ---------------------------------------------------------------------------
# build_behavioral_rdm (end-to-end on synthetic data)
# ---------------------------------------------------------------------------

def test_build_behavioral_rdm_condensed_length(patched_behavioral):
    condensed = behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused")
    assert len(condensed) == 4 * 3 // 2  # 6 pairs


def test_build_behavioral_rdm_excludes_uncovered_pair(patched_behavioral):
    """img0-img3 is never jointly observed in any p1/p2 trial (only p3's excluded
    screened-out trial touches it) -- must come out NaN, not silently zero or dropped."""
    condensed = behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused")
    sq = squareform(condensed)
    i0, i3 = _IMAGES.index("img0.png"), _IMAGES.index("img3.png")
    assert np.isnan(sq[i0, i3])


def test_build_behavioral_rdm_covered_pairs_are_finite_and_nonneg(patched_behavioral):
    condensed = behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused")
    sq = squareform(condensed)
    i1, i2 = _IMAGES.index("img1.png"), _IMAGES.index("img2.png")
    assert np.isfinite(sq[i1, i2])
    assert sq[i1, i2] >= 0


def test_build_behavioral_rdm_symmetric(patched_behavioral):
    condensed = behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused")
    sq = squareform(condensed)
    assert np.allclose(sq, sq.T, equal_nan=True)


def test_build_behavioral_rdm_excludes_screened_out_and_catch_and_other_variant(patched_behavioral):
    """n_participants must be 2 (p1, p2 only) -- p3 is screened out, p4 is the
    wrong shine_variant, p5 is below the min task_version, and p1's catch trial
    must not count toward n_trials."""
    behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused")
    extra = patched_behavioral["extra"]
    assert extra["n_participants"] == 2
    assert extra["n_trials"] == 3  # p1 screening + p1 experimental + p2 experimental


def test_build_behavioral_rdm_saves_with_expected_name(patched_behavioral):
    behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused")
    assert patched_behavioral["name"] == "behav_pre_evidence"


def test_build_behavioral_rdm_name_encodes_aggregation(patched_behavioral):
    behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused", aggregation="mean")
    assert patched_behavioral["name"] == "behav_pre_mean"


def test_build_behavioral_rdm_rejects_unknown_aggregation(patched_behavioral):
    with pytest.raises(ValueError, match="aggregation"):
        behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused", aggregation="bogus")


# ---------------------------------------------------------------------------
# Regression (golden-output) tests, one per aggregation method -- same fixture,
# same variant, only `aggregation` varies, so a future change to any one
# method's math is caught without touching the other two's expected values.
#
# Expected values below are hand-verified for 'mean' against _synthetic_trials():
# pair (1,2) [img1-img2] is observed in all 3 trials (0.3, 0.32, 0.28) -> mean 0.3;
# pair (1,3) in 2 trials (0.4, 0.42) -> mean 0.41; pair (2,3) in 2 trials (0.5, 0.48)
# -> mean 0.49; (0,1)/(0,2) each observed once (0.1/0.2); (0,3) never observed -> NaN.
# 'evidence'/'simple' values are pinned from the current implementation's own output
# (cross-checked elsewhere against rsatoolbox's reference -- see
# test_sparse_combine_matches_rsatoolbox_dense above); this test's job is to catch
# future *drift*, not to re-derive them by hand.
# ---------------------------------------------------------------------------

def test_build_behavioral_rdm_mean_matches_expected(patched_behavioral):
    condensed = behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused", aggregation="mean")
    expected = [0.1, 0.2, np.nan, 0.3, 0.41, 0.49]
    np.testing.assert_allclose(condensed, expected, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_build_behavioral_rdm_evidence_matches_expected(patched_behavioral):
    condensed = behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused", aggregation="evidence")
    expected = [0.135387114, 0.270774228, np.nan, 0.4070538003, 0.554730821, 0.6615233623]
    np.testing.assert_allclose(condensed, expected, rtol=1e-8, atol=1e-10, equal_nan=True)


def test_build_behavioral_rdm_simple_matches_expected(patched_behavioral):
    condensed = behavioral.build_behavioral_rdm("pre_shine", cache_dir="unused", aggregation="simple")
    expected = [0.1350432326, 0.2700864653, np.nan, 0.4048221759, 0.5537341126, 0.6613756875]
    np.testing.assert_allclose(condensed, expected, rtol=1e-8, atol=1e-10, equal_nan=True)


def test_build_behavioral_rdm_post_shine_uses_own_prefix(patched_behavioral, monkeypatch):
    """Swap in post_shine-prefixed trial data and confirm the post-SHINE URL prefix
    is what gets stripped (not a hardcoded pre_shine assumption)."""
    post_participants = pd.DataFrame([
        {"participant_id": "q1", "status": "full data", "shine_variant": "post", "task_version": 4.0},
    ])
    post_trials = pd.DataFrame([{
        "participant_id": "q1", "is_catch": False, "block_type": "experimental",
        "pairwise_distances": _pairwise_json(
            {("img0.png", "img1.png"): 0.15, ("img0.png", "img2.png"): 0.25,
             ("img1.png", "img2.png"): 0.35},
            "post_shine",
        ),
    }])
    monkeypatch.setattr(
        behavioral, "load_data",
        lambda cache_dir, prod_only=False: {"participants": post_participants, "trials": post_trials},
    )
    condensed = behavioral.build_behavioral_rdm("post_shine", cache_dir="unused")
    assert len(condensed) == 6
    assert patched_behavioral["name"] == "behav_post_evidence"


# ---------------------------------------------------------------------------
# _mean_combine
# ---------------------------------------------------------------------------

def test_mean_combine_matches_total_over_count():
    """Plain unweighted total/count, matching SpAM_Simulations' mean_from_sum_and_count()
    exactly -- no iterative rescaling, unlike _sparse_combine()."""
    n_conds = 4
    trial_a = (np.array([0, 3]), np.array([1.0, 2.0]))   # pair 0=(0,1), pair 3=(1,2)
    trial_b = (np.array([0, 3]), np.array([3.0, 4.0]))
    condensed, n_obs = behavioral._mean_combine([trial_a, trial_b], n_conds)
    assert condensed[0] == pytest.approx(2.0)   # mean(1.0, 3.0)
    assert condensed[3] == pytest.approx(3.0)   # mean(2.0, 4.0)
    assert n_obs[0] == 2
    assert n_obs[3] == 2
    assert np.isnan(condensed[1]) and np.isnan(condensed[2]) and np.isnan(condensed[4]) and np.isnan(condensed[5])


def test_mean_combine_single_observation_is_unchanged():
    n_conds = 3
    trial = (np.array([1]), np.array([0.42]))
    condensed, n_obs = behavioral._mean_combine([trial], n_conds)
    assert condensed[1] == pytest.approx(0.42)
    assert n_obs[1] == 1
