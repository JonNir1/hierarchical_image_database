"""Tests for ground-truth construction and dimensionality selection.

Everything except the SMACOF path runs under ``method="classical"`` on a small synthetic pilot, so
the suite needs no R. The one R-dependent test uses the module-level skipif idiom.

Note ``method="classical"`` mean-imputes and is a smoke-test path only - these tests exercise
plumbing and selection logic, not the scientific claim that a particular dimensionality is right.
"""
import numpy as np
import pandas as pd
import pytest

from SpAM_Simulations.empirical import gt_construction as gtc

try:
    from SpAM_Simulations.core.multi_dimensional_scaling import run_mds  # noqa: F401
    _MDS_AVAILABLE, _MDS_SKIP = True, ""
except (ImportError, IndexError, RuntimeError, OSError, AttributeError) as e:
    _MDS_AVAILABLE, _MDS_SKIP = False, f"R/rpy2/smacof unavailable: {type(e).__name__}: {e}"


class _Subject:
    """Minimal Subject stand-in: the module only needs distances/n_obs/task_version."""

    def __init__(self, distances, n_obs, task_version=3.0, shine_variant="pre"):
        self.distances = np.asarray(distances, dtype=np.float32)
        self.n_obs = np.asarray(n_obs, dtype=np.int32)
        self.task_version = task_version
        self.shine_variant = shine_variant
        self.participant_id = "P"


def _cohort(n_subjects=12, n_images=12, seed=0, observed_frac=1.0):
    """Subjects whose distances come from a shared latent space, each observing a random subset."""
    rng = np.random.default_rng(seed)
    n_pairs = n_images * (n_images - 1) // 2
    from scipy.spatial.distance import pdist
    truth = pdist(rng.normal(size=(n_images, 3)))
    subs = []
    for _ in range(n_subjects):
        obs = (rng.random(n_pairs) < observed_frac).astype(np.int32)
        d = truth + rng.normal(0, 0.05, n_pairs)
        subs.append(_Subject(np.where(obs > 0, d, np.nan), obs))
    return subs


# --------------------------------------------------------------------- aggregation

def test_aggregate_weights_mark_observed_pairs():
    subs = _cohort(observed_frac=0.5, seed=1)
    dists, weights = gtc.aggregate_subjects(subs)
    assert set(np.unique(weights)) <= {0.0, 1.0}
    assert dists.shape == weights.shape


def test_aggregate_does_not_raise_on_a_disconnected_graph():
    """Unlike aggregate: the split search must be able to *test* subsets cheaply."""
    n_pairs = 6
    lonely = _Subject(np.full(n_pairs, np.nan), np.zeros(n_pairs))
    lonely.n_obs[0] = 1
    lonely.distances[0] = 0.5
    gtc.aggregate_subjects([lonely])          # must not raise
    assert not gtc.is_connected([lonely])


def test_fully_observed_cohort_is_connected():
    assert gtc.is_connected(_cohort(observed_frac=1.0))
    assert gtc.coverage_of(_cohort(observed_frac=1.0)) == 1.0


# --------------------------------------------------------------------- split search

def test_splits_are_disjoint_and_cover_the_cohort():
    subs = _cohort(n_subjects=12)
    splits, diag = gtc.draw_valid_splits(subs, n_draws=5, rng=np.random.default_rng(0))
    assert len(splits) == 5
    for a, b in splits:
        assert not set(a.tolist()) & set(b.tolist())
        assert len(a) == diag["half_size"]


def test_discarded_draws_are_counted_and_replaced(monkeypatch):
    """Reject the first few draws and assert we still get exactly n_draws, with the rate recorded."""
    subs = _cohort(n_subjects=12)
    calls = {"n": 0}
    real = gtc.is_connected

    def flaky(s):
        calls["n"] += 1
        return False if calls["n"] <= 4 else real(s)

    monkeypatch.setattr(gtc, "is_connected", flaky)
    splits, diag = gtc.draw_valid_splits(subs, n_draws=3, rng=np.random.default_rng(0))
    assert len(splits) == 3
    assert diag["n_discarded"] > 0
    assert 0 < diag["discard_rate"] < 1
    assert diag["n_attempts"] == len(splits) + diag["n_discarded"]


def test_split_search_gives_up_rather_than_looping_forever(monkeypatch):
    monkeypatch.setattr(gtc, "is_connected", lambda s: False)
    with pytest.raises(RuntimeError, match="too sparse to split"):
        gtc.draw_valid_splits(_cohort(), n_draws=2, rng=np.random.default_rng(0))


def test_split_search_rejects_a_tiny_cohort():
    with pytest.raises(ValueError, match="at least 4 subjects"):
        gtc.draw_valid_splits(_cohort(n_subjects=3), n_draws=1, rng=np.random.default_rng(0))


# --------------------------------------------------------------------- scoring

def test_identical_configurations_score_perfectly():
    coords = np.random.default_rng(0).normal(size=(20, 3))
    s = gtc.split_half_scores(coords, coords)
    assert s["spearman"] == pytest.approx(1.0)
    assert s["procrustes_m2"] == pytest.approx(0.0, abs=1e-10)
    assert s["topk_jaccard"] == pytest.approx(1.0)


def test_scores_degrade_as_configurations_diverge():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 3))
    near = gtc.split_half_scores(base, base + rng.normal(0, 0.05, base.shape))
    far = gtc.split_half_scores(base, base + rng.normal(0, 2.0, base.shape))
    assert near["spearman"] > far["spearman"]
    assert near["procrustes_m2"] < far["procrustes_m2"]


def test_procrustes_is_invariant_to_rotation_and_scale():
    """MDS solutions are only defined up to these, so the metric must not see them."""
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 2))
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert gtc.split_half_scores(coords, 3.0 * coords @ rot)["procrustes_m2"] == pytest.approx(0.0, abs=1e-10)


# --------------------------------------------------------------------- selection

def _scan(means, sems=0.01, n=20):
    """Synthetic scan whose per-ndim means are exactly `means`."""
    rows = []
    rng = np.random.default_rng(0)
    for ndim, m in means.items():
        vals = m + rng.normal(0, sems, n)
        vals = m + (vals - vals.mean())          # force the mean exactly
        for draw, v in enumerate(vals):
            rows.append({"ndim": ndim, "draw": draw, "spearman": v,
                         "procrustes_m2": 1 - v, "topk_jaccard": v})
    return pd.DataFrame(rows)


def test_one_se_prefers_the_smaller_of_two_indistinguishable_dims():
    """The whole point: a flat curve must not drift to high ndim on noise."""
    scan = _scan({2: 0.80, 5: 0.805, 10: 0.806}, sems=0.05)
    assert gtc.select_ndim(scan, rule="one_se") == 2
    assert gtc.select_ndim(scan, rule="argmax") == 10


def test_one_se_rejects_a_smaller_dim_that_is_genuinely_worse():
    """One-SE is a tie-break, not a bias toward small: ndim 2 is far below and must lose.

    5 and 10 are within one SE of each other here (SEM of the mean is 0.05/sqrt(20) ~ 0.011, and
    they differ by 0.01), so parsimony picks 5; 2 is 0.40 below and never qualifies.
    """
    scan = _scan({2: 0.40, 5: 0.80, 10: 0.81}, sems=0.05)
    assert gtc.select_ndim(scan, rule="one_se") == 5


def test_one_se_defers_to_argmax_when_the_gap_is_real():
    """With a tight SEM, 0.80 vs 0.81 is distinguishable and the larger dim genuinely wins."""
    scan = _scan({2: 0.40, 5: 0.80, 10: 0.81}, sems=0.005)
    assert gtc.select_ndim(scan, rule="one_se") == 10


def test_selection_respects_metric_direction():
    """Procrustes M^2 is lower-is-better; selecting on it must not invert the answer."""
    scan = _scan({2: 0.40, 5: 0.80, 10: 0.81}, sems=0.005)
    assert gtc.select_ndim(scan, criterion="procrustes_m2", rule="argmax") == 10
    assert gtc.select_ndim(scan, criterion="spearman", rule="argmax") == 10


def test_unknown_criterion_or_rule_is_rejected():
    scan = _scan({2: 0.5, 5: 0.6})
    with pytest.raises(ValueError, match="criterion"):
        gtc.select_ndim(scan, criterion="nonsense")
    with pytest.raises(ValueError, match="rule"):
        gtc.select_ndim(scan, rule="nonsense")


def test_summarise_scan_reports_mean_and_sem_per_ndim():
    summary = gtc.summarise_scan(_scan({2: 0.5, 5: 0.7}))
    assert list(summary["ndim"]) == [2, 5]
    assert summary["spearman_mean"].tolist() == pytest.approx([0.5, 0.7])


# --------------------------------------------------------------------- scan / CV plumbing

def test_dimensionality_scan_reuses_one_set_of_splits():
    """Paired comparison: every ndim must be scored on the SAME splits, else draw noise dominates."""
    subs = _cohort(n_subjects=10, n_images=10)
    scan, diag = gtc.dimensionality_scan(subs, ndims=[2, 3], n_draws=3, method="classical",
                                         verbose=False)
    assert set(scan["ndim"]) == {2, 3}
    assert len(scan) == 6
    for ndim in (2, 3):
        assert sorted(scan[scan["ndim"] == ndim]["draw"]) == [0, 1, 2]
    assert diag["n_draws"] == 3


def test_scan_accepts_precomputed_splits():
    subs = _cohort(n_subjects=10, n_images=10)
    splits, _ = gtc.draw_valid_splits(subs, n_draws=2, rng=np.random.default_rng(0))
    scan, diag = gtc.dimensionality_scan(subs, ndims=[2], method="classical", verbose=False,
                                         splits=splits)
    assert len(scan) == 2 and diag == {}


def test_leave_k_out_folds_are_the_right_size_and_within_range():
    folds = gtc.leave_k_out_folds(20, k=5, n_folds=7, rng=np.random.default_rng(0))
    assert len(folds) == 7
    for f in folds:
        assert len(np.unique(f)) == 5 and f.min() >= 0 and f.max() < 20


def test_leave_k_out_rejects_impossible_k():
    with pytest.raises(ValueError, match="k must be in"):
        gtc.leave_k_out_folds(5, k=5)


def test_cross_validate_scores_every_ndim_and_fold():
    subs = _cohort(n_subjects=10, n_images=10)
    cv = gtc.cross_validate_ndim(subs, ndims=[2, 3], k=2, n_folds=3, method="classical",
                                 verbose=False)
    assert len(cv) == 6
    assert cv["spearman"].notna().all()


def test_cross_validation_recovers_a_known_dimensionality_ordering():
    """A 2-D latent space should not score better at 2 dims than a 3-D one does - sanity only."""
    subs = _cohort(n_subjects=12, n_images=14, seed=3)
    cv = gtc.cross_validate_ndim(subs, ndims=[1, 3], k=3, n_folds=5, method="classical",
                                 verbose=False)
    means = cv.groupby("ndim")["spearman"].mean()
    assert means[3] > means[1], "more dimensions must fit a 3-D latent space better than one does"


# --------------------------------------------------------------------- build_gt

def test_build_gt_returns_coords_and_provenance():
    subs = _cohort(n_subjects=8, n_images=10)
    coords, info = gtc.build_gt(subs, ndim=3, method="classical")
    assert coords.shape == (10, 3)
    assert info["n_dims"] == 3 and info["n_subjects"] == 8
    assert info["variants"] == ["pre"]
    assert info["observed_frac"] == pytest.approx(1.0)


def test_build_gt_requires_a_positive_ndim():
    with pytest.raises(ValueError, match="must be positive"):
        gtc.build_gt(_cohort(), ndim=0, method="classical")


def test_build_gt_refuses_a_disconnected_graph():
    n_pairs = 6
    lonely = _Subject(np.full(n_pairs, np.nan), np.zeros(n_pairs))
    lonely.n_obs[0] = 1
    lonely.distances[0] = 0.5
    with pytest.raises(RuntimeError, match="connected components"):
        gtc.build_gt([lonely], ndim=2, method="classical")


def test_build_gt_from_pilot_delegates(monkeypatch):
    """subjects.build_gt_from_pilot is now a thin wrapper; n_dims must be required."""
    from SpAM_Simulations.empirical import subjects
    seen = {}
    monkeypatch.setattr(subjects, "build_gt", lambda s, n, method="smacof": (seen.update(
        dict(n=n, method=method)) or (np.zeros((3, n), dtype=np.float32), {"n_dims": n})))
    subjects.build_gt_from_pilot(_cohort(n_subjects=4), 7, method="classical")
    assert seen == {"n": 7, "method": "classical"}


def test_choose_n_dims_is_gone():
    """The imputed-eigenspectrum rule must not come back: it manufactured rank on sparse data."""
    from SpAM_Simulations.empirical import subjects
    assert not hasattr(subjects, "_choose_n_dims")


@pytest.mark.skipif(not _MDS_AVAILABLE, reason=_MDS_SKIP)
def test_smacof_embedding_matches_the_requested_shape():
    subs = _cohort(n_subjects=6, n_images=12)
    coords = gtc.embed_subset(subs, ndim=3, method="smacof", max_iters=100)
    assert coords.shape == (12, 3)
    assert np.isfinite(coords).all()


# --------------------------------------------------------------------- parallel execution
# `_embed_payload` is the only R-dependent piece, so it is stubbed here and the surrounding
# payload-building / scoring / bookkeeping is what these tests actually cover.

def _stub_embed(monkeypatch, ndim_coords, status="success"):
    """Replace `_embed_payload` with a deterministic fake that echoes the payload's key."""
    calls = []

    def fake(payload):
        key, dists, weights, ndim, *_ = payload
        calls.append({"key": key, "ndim": ndim, "n_pairs": int(np.size(dists)),
                      "n_observed": int(np.count_nonzero(weights))})
        rng = np.random.default_rng(abs(hash((str(key), ndim))) % (2 ** 32))
        coords = ndim_coords + rng.normal(0, 1e-3, ndim_coords.shape)
        return key, coords.astype(np.float32), {"status": status, "niter": 7.0, "stress": 0.1}

    monkeypatch.setattr(gtc, "_embed_payload", fake)
    return calls


def test_split_aggregates_pools_each_half_once():
    subs = _cohort(n_subjects=10, n_images=10, observed_frac=0.6)
    splits, _ = gtc.draw_valid_splits(subs, n_draws=3, rng=np.random.default_rng(0))
    aggs = gtc.split_aggregates(subs, splits)
    assert len(aggs) == 3
    for (da, wa), (db, wb) in aggs:
        assert da.shape == wa.shape == db.shape == wb.shape == (45,)
        assert set(np.unique(wa)) <= {0.0, 1.0}
    # The pooled halves must match a direct aggregation of the same subject indices.
    (ia, _ib) = splits[0]
    expect = gtc.aggregate_subjects([subs[i] for i in ia])
    np.testing.assert_allclose(aggs[0][0][0], expect[0])


def test_scan_ndim_parallel_returns_one_row_per_draw_with_solver_diagnostics(monkeypatch):
    subs = _cohort(n_subjects=10, n_images=10, observed_frac=0.6)
    splits, _ = gtc.draw_valid_splits(subs, n_draws=4, rng=np.random.default_rng(0))
    aggs = gtc.split_aggregates(subs, splits)
    calls = _stub_embed(monkeypatch, np.random.default_rng(1).normal(size=(10, 3)))

    out = gtc.scan_ndim_parallel(aggs, ndim=3, verbose=False)
    assert len(out) == 4
    assert list(out["draw"]) == [0, 1, 2, 3]
    assert (out["ndim"] == 3).all()
    # Two fits per draw, and both halves' solver status is carried through rather than discarded:
    # a scan where every fit hit max_iters is measuring the stopping rule, not the data.
    assert len(calls) == 8
    assert set(out["status_a"]) == {"success"} and set(out["status_b"]) == {"success"}
    assert (out["niter_a"] == 7.0).all()
    for col in ("spearman", "procrustes_m2", "topk_jaccard"):
        assert out[col].notna().all()


def test_scan_ndim_parallel_records_nan_scores_for_a_failed_fit(monkeypatch):
    """One unusable draw must not abort a multi-hour scan, nor be silently scored."""
    subs = _cohort(n_subjects=8, n_images=10, observed_frac=0.6)
    splits, _ = gtc.draw_valid_splits(subs, n_draws=2, rng=np.random.default_rng(0))
    aggs = gtc.split_aggregates(subs, splits)

    def fake(payload):
        key = payload[0]
        if key == ("a", 1):
            return key, None, {"status": "error", "niter": np.nan, "stress": np.nan}
        return key, np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32), \
            {"status": "success", "niter": 3.0, "stress": 0.1}

    monkeypatch.setattr(gtc, "_embed_payload", fake)
    out = gtc.scan_ndim_parallel(aggs, ndim=3, verbose=False).set_index("draw")
    assert out.loc[1, "status_a"] == "error"
    assert np.isnan(out.loc[1, "spearman"])
    assert np.isfinite(out.loc[0, "spearman"])


def test_scan_ndim_parallel_agrees_with_the_serial_scan(monkeypatch):
    """The parallel driver must be a scheduling change only, not a different computation."""
    subs = _cohort(n_subjects=10, n_images=10, observed_frac=0.7)
    splits, _ = gtc.draw_valid_splits(subs, n_draws=3, rng=np.random.default_rng(2))

    # A stub keyed only on the subject subset, so serial and parallel see identical coordinates.
    def coords_for(weights):
        rng = np.random.default_rng(int(np.count_nonzero(weights)))
        return rng.normal(size=(10, 3)).astype(np.float32)

    monkeypatch.setattr(gtc, "_embed_payload", lambda p: (
        p[0], coords_for(p[2]), {"status": "success", "niter": 1.0, "stress": 0.0}))
    monkeypatch.setattr(gtc, "embed_subset",
                        lambda s, ndim, method="smacof", **kw: coords_for(gtc.observed_mask(s)))

    serial, _ = gtc.dimensionality_scan(subs, ndims=[3], splits=splits, verbose=False)
    parallel = gtc.scan_ndim_parallel(gtc.split_aggregates(subs, splits), ndim=3, verbose=False)
    for col in ("spearman", "procrustes_m2", "topk_jaccard"):
        np.testing.assert_allclose(serial[col].to_numpy(), parallel[col].to_numpy(), rtol=1e-6)


def test_cross_validate_ndim_parallel_holds_out_the_named_folds(monkeypatch):
    subs = _cohort(n_subjects=12, n_images=10, observed_frac=0.8)
    folds = gtc.leave_k_out_folds(12, k=3, n_folds=4, rng=np.random.default_rng(0))
    calls = _stub_embed(monkeypatch, np.random.default_rng(1).normal(size=(10, 3)))

    out = gtc.cross_validate_ndim_parallel(subs, ndim=3, folds=folds, verbose=False)
    assert len(out) == 4 and (out["ndim"] == 3).all()
    assert list(out["fold"]) == [0, 1, 2, 3]
    assert out["spearman"].notna().all()
    # One fit per fold, each trained on the 9 subjects not held out.
    assert len(calls) == 4


def test_cross_validate_ndim_parallel_matches_the_serial_version(monkeypatch):
    subs = _cohort(n_subjects=12, n_images=10, observed_frac=0.8)
    folds = gtc.leave_k_out_folds(12, k=3, n_folds=3, rng=np.random.default_rng(0))

    def coords_for(weights):
        rng = np.random.default_rng(int(np.count_nonzero(weights)))
        return rng.normal(size=(10, 3)).astype(np.float32)

    monkeypatch.setattr(gtc, "_embed_payload", lambda p: (
        p[0], coords_for(p[2]), {"status": "success", "niter": 1.0, "stress": 0.0}))
    monkeypatch.setattr(gtc, "embed_subset",
                        lambda s, ndim, method="smacof", **kw: coords_for(gtc.observed_mask(s)))
    monkeypatch.setattr(gtc, "leave_k_out_folds", lambda *a, **k: folds)

    serial = gtc.cross_validate_ndim(subs, ndims=[3], k=3, n_folds=3, verbose=False)
    parallel = gtc.cross_validate_ndim_parallel(subs, ndim=3, folds=folds, verbose=False)
    np.testing.assert_allclose(serial["spearman"].to_numpy(), parallel["spearman"].to_numpy(),
                               rtol=1e-6)


# --------------------------------------------------------------------- split diagnostics
# Regression on a real defect: the coverage diagnostic used to be `0.5 * (cov_a + cov_b)`, the
# average of the TWO COMPLEMENTARY halves. A well-covered half forces a poorly-covered partner, so
# that quantity is near-invariant across permutations - measured on the pilot it had sd 0.0001 and
# reported 0.172 for kept and discarded alike. It could not have detected a gap of any size.

def test_the_coverage_diagnostic_is_the_binding_half_not_the_average():
    """The average of two complementary halves is a near-constant and detects nothing."""
    subs = _cohort(n_subjects=16, n_images=14, observed_frac=0.25, seed=5)
    _, diag = gtc.draw_valid_splits(subs, n_draws=8, rng=np.random.default_rng(0))
    assert "mean_binding_coverage_kept" in diag
    assert "mean_binding_coverage_discarded" in diag
    assert "coverage_gap" in diag and "coverage_gap_frac" in diag
    # The retired keys must not come back: they carried a statistic with no power.
    assert "mean_coverage_kept" not in diag
    assert "mean_coverage_discarded" not in diag


def test_the_binding_coverage_is_the_minimum_of_the_two_halves(monkeypatch):
    """Pin the definition: a split's recorded coverage is its WORSE half, since that is the one
    that gets it discarded."""
    subs = _cohort(n_subjects=10, n_images=12, observed_frac=0.5, seed=2)
    splits, diag = gtc.draw_valid_splits(subs, n_draws=4, rng=np.random.default_rng(1))
    expected = [min(gtc.coverage_of([subs[i] for i in a]),
                    gtc.coverage_of([subs[i] for i in b])) for a, b in splits]
    assert diag["mean_binding_coverage_kept"] == pytest.approx(float(np.mean(expected)))
    # and it is strictly below the average of the two halves, which is what made the old one blind
    avg = [0.5 * (gtc.coverage_of([subs[i] for i in a]) + gtc.coverage_of([subs[i] for i in b]))
           for a, b in splits]
    assert float(np.mean(expected)) < float(np.mean(avg))


def test_the_gap_is_reported_both_absolutely_and_relatively():
    """A threshold must read the relative gap: 0.001 means very different things at 0.17 and 0.9."""
    subs = _cohort(n_subjects=16, n_images=14, observed_frac=0.25, seed=7)
    _, diag = gtc.draw_valid_splits(subs, n_draws=6, rng=np.random.default_rng(3))
    if np.isfinite(diag["coverage_gap"]) and diag["mean_binding_coverage_kept"]:
        assert diag["coverage_gap_frac"] == pytest.approx(
            diag["coverage_gap"] / diag["mean_binding_coverage_kept"])
    assert diag["coverage_gap"] == pytest.approx(
        diag["mean_binding_coverage_kept"] - diag["mean_binding_coverage_discarded"], nan_ok=True)


def test_diagnostics_survive_a_run_with_no_discards():
    """Nothing discarded means no gap to report, not a crash."""
    subs = _cohort(n_subjects=12, n_images=10, observed_frac=1.0, seed=0)
    _, diag = gtc.draw_valid_splits(subs, n_draws=3, rng=np.random.default_rng(0))
    assert diag["n_discarded"] == 0
    assert np.isnan(diag["mean_binding_coverage_discarded"])
    assert np.isnan(diag["coverage_gap"])
