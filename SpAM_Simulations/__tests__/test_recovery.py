"""Tests for recovery-vs-ground-truth statistics.

Anchored on cases whose answers are known analytically - perfect recovery, chance recovery, and
monotone degradation under injected noise - rather than on regression values, since the point of
these statistics is that their scale is interpretable.

No R needed.
"""
import numpy as np
import pytest

from SpAM_Simulations import recovery as rec


def _gt(n=2000, seed=0):
    return np.random.default_rng(seed).random(n)


# --------------------------------------------------------------------- topk_mask

def test_topk_mask_selects_the_smallest_entries():
    d = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    assert rec.topk_mask(d, 0.4).tolist() == [False, True, False, True, False]


def test_topk_mask_always_selects_at_least_one():
    assert rec.topk_mask(np.arange(1000.0), 1e-9).sum() == 1


def test_topk_mask_rejects_out_of_range_fracs():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="frac must be in"):
            rec.topk_mask(np.arange(10.0), bad)


# --------------------------------------------------------------------- perfect / chance

def test_perfect_recovery_scores_at_ceiling():
    gt = _gt()
    assert rec.recall_at_frac(gt, gt, 0.05) == 1.0
    assert rec.auc_near_pairs(gt, gt, 0.05) == pytest.approx(1.0)
    assert rec.separation_dprime(gt, gt, 0.05) > 0
    assert np.isfinite(rec.dprime_at_frac(gt, gt, 0.05)), "loglinear correction must keep it finite"


def test_a_shuffled_recovery_scores_at_chance():
    gt = _gt(n=5000)
    shuffled = np.random.default_rng(1).permutation(gt)
    frac = 0.05
    assert rec.recall_at_frac(shuffled, gt, frac) == pytest.approx(frac, abs=0.03)
    assert rec.auc_near_pairs(shuffled, gt, frac) == pytest.approx(0.5, abs=0.05)
    assert rec.dprime_at_frac(shuffled, gt, frac) == pytest.approx(0.0, abs=0.25)
    assert rec.separation_dprime(shuffled, gt, frac) == pytest.approx(0.0, abs=0.15)


def test_inverted_recovery_scores_below_chance():
    """Flipping the ordering must push AUC below 0.5, not merely away from 1."""
    gt = _gt()
    assert rec.auc_near_pairs(-gt, gt, 0.05) < 0.1


# --------------------------------------------------------------------- monotonicity

@pytest.mark.parametrize("metric", ["recall_at_frac", "auc_near_pairs", "separation_dprime",
                                    "dprime_at_frac"])
def test_every_metric_degrades_monotonically_with_noise(metric):
    gt = _gt(n=3000)
    rng = np.random.default_rng(2)
    fn = getattr(rec, metric)
    scores = [fn(gt + rng.normal(0, s, gt.shape), gt, 0.05) for s in (0.001, 0.05, 0.2, 1.0)]
    assert scores == sorted(scores, reverse=True), f"{metric} not monotone: {scores}"


def test_recall_equals_precision_at_matched_fractions():
    """Documented rationale for reporting recall alone: the two sets are the same size."""
    gt = _gt()
    rng = np.random.default_rng(3)
    recovered = gt + rng.normal(0, 0.1, gt.shape)
    g = rec.topk_mask(gt, 0.05)
    r = rec.topk_mask(recovered, 0.05)
    recall = np.count_nonzero(g & r) / g.sum()
    precision = np.count_nonzero(g & r) / r.sum()
    assert recall == pytest.approx(precision)


# --------------------------------------------------------------------- d-prime arithmetic

def test_dprime_matches_a_hand_computed_sdt_table():
    from scipy.stats import norm
    # 100 pairs, top 10% -> 10 GT-positives; construct exactly 6 hits.
    gt = np.arange(100.0)
    recovered = gt.copy()
    recovered[[0, 1, 2, 3]] = 99.0          # 4 GT-near pairs pushed out of the recovered top-10
    d = rec.dprime_at_frac(recovered, gt, 0.10)
    hits, n_pos, fa, n_neg = 6, 10, 4, 90
    expected = norm.ppf((hits + 0.5) / (n_pos + 1)) - norm.ppf((fa + 0.5) / (n_neg + 1))
    assert d == pytest.approx(expected)


def test_uncorrected_dprime_is_infinite_at_ceiling():
    """Why loglinear is the default."""
    gt = _gt()
    assert np.isinf(rec.dprime_at_frac(gt, gt, 0.05, correction="none"))
    assert np.isfinite(rec.dprime_at_frac(gt, gt, 0.05, correction="loglinear"))


def test_unknown_correction_is_rejected():
    gt = _gt()
    with pytest.raises(ValueError, match="correction"):
        rec.dprime_at_frac(gt, gt, 0.05, correction="nonsense")


# --------------------------------------------------------------------- scale invariance

def test_rank_based_metrics_ignore_monotone_rescaling():
    """MDS output has no natural scale, so the threshold-free metrics must not see one."""
    gt = _gt()
    recovered = gt + np.random.default_rng(4).normal(0, 0.1, gt.shape)
    for scale in (0.01, 100.0):
        assert rec.auc_near_pairs(recovered * scale, gt, 0.05) == pytest.approx(
            rec.auc_near_pairs(recovered, gt, 0.05))
        assert rec.recall_at_frac(recovered * scale, gt, 0.05) == pytest.approx(
            rec.recall_at_frac(recovered, gt, 0.05))


# --------------------------------------------------------------------- summary

def test_recovery_summary_covers_every_frac():
    gt = _gt()
    out = rec.recovery_summary(gt, gt, fracs=(0.01, 0.05))
    assert [r["top_frac"] for r in out] == [0.01, 0.05]
    assert all(set(r) == {"top_frac", "recall", "dprime", "separation_dprime", "auc"} for r in out)


def test_recovery_summary_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        rec.recovery_summary(np.arange(10.0), np.arange(20.0))


# --------------------------------------------------------------------- pipeline integration

def test_compute_recovery_vs_gt_groups_and_averages(tmp_path):
    from SpAM_Simulations import pipeline
    from SpAM_Simulations.storage import ResultStore

    n_pairs = 500
    gt = _gt(n=n_pairs)
    store = ResultStore.create(tmp_path / "store", confdist_len=n_pairs,
                               meta_columns=["num_subjects", "rep", "ndim", "niter", "stress",
                                             "status"])
    rng = np.random.default_rng(5)
    for num_subjects, sigma in ((10, 1.0), (100, 0.01)):   # noisy cohort vs near-perfect cohort
        for rep in range(3):
            store.append({"num_subjects": num_subjects, "rep": rep, "ndim": 5, "niter": 10,
                          "stress": 0.1, "status": "success"},
                         confdist=(gt + rng.normal(0, sigma, n_pairs)).astype(np.float32))
    store.close()

    out = pipeline.compute_recovery_vs_gt(ResultStore.open(tmp_path / "store"), gt,
                                          fracs=(0.05,), verbose=False)
    assert set(out["num_subjects"]) == {10, 100}
    assert (out["n_reps"] == 3).all()
    good = out[out.num_subjects == 100]["mean_recall"].iloc[0]
    bad = out[out.num_subjects == 10]["mean_recall"].iloc[0]
    assert good > bad, "the near-perfect cohort must recover more of the GT's closest pairs"


def test_compute_recovery_vs_gt_rejects_a_mismatched_ground_truth(tmp_path):
    from SpAM_Simulations import pipeline
    from SpAM_Simulations.storage import ResultStore

    store = ResultStore.create(tmp_path / "s", confdist_len=100,
                               meta_columns=["rep", "ndim", "niter", "stress", "status"])
    store.append({"rep": 0, "ndim": 3, "niter": 5, "stress": 0.1, "status": "success"},
                 confdist=np.zeros(100, dtype=np.float32))
    store.close()
    with pytest.raises(ValueError, match="must index the same image set"):
        pipeline.compute_recovery_vs_gt(ResultStore.open(tmp_path / "s"), _gt(n=50), verbose=False)
