"""Tests for between-cohort cluster-structure agreement.

Anchored on cases with known answers: identical partitions, chance-level partitions, a planted
three-blob structure, and a continuum with no clusters at all. The continuum case is the important
one, since the design has to be able to *report* that no stable granularity exists rather than
silently returning one.

No R needed anywhere.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from SpAM_Simulations import cluster_stability as cs


# --------------------------------------------------------------------- factories

def _blobs(n_per=20, k=3, sd=0.25, seed=0, ndim=3):
    """`k` well-separated Gaussian blobs; the planted structure the metrics should recover."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 10, (k, ndim))
    return np.vstack([c + rng.normal(0, sd, (n_per, ndim)) for c in centres])


def _continuum(n=60, ndim=3, seed=0):
    """Uniform points in a box: no cluster structure to find, at any granularity."""
    return np.random.default_rng(seed).random((n, ndim))


def _noisy(coords, sd, seed):
    return coords + np.random.default_rng(seed).normal(0, sd, coords.shape)


def _labels(n, k, seed):
    return np.random.default_rng(seed).integers(0, k, n)


# --------------------------------------------------------------------- VI is a metric
#
# These are load-bearing: the whole reason VI is primary over ARI is that it obeys the triangle
# inequality, which is what lets two separately-measured agreements be chained.

class TestVariationOfInformationIsAMetric:
    def test_identical_partitions_give_exactly_zero(self):
        a = _labels(200, 8, 0)
        assert cs.variation_of_information(a, a.copy(), normalise=False) == 0.0
        assert cs.variation_of_information(a, a.copy()) == 0.0

    def test_symmetry(self):
        a, b = _labels(200, 6, 1), _labels(200, 9, 2)
        assert cs.variation_of_information(a, b) == pytest.approx(cs.variation_of_information(b, a))

    def test_non_negativity_over_random_pairs(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            a, b = _labels(120, rng.integers(2, 12), rng.integers(0, 9999)), \
                   _labels(120, rng.integers(2, 12), rng.integers(0, 9999))
            assert cs.variation_of_information(a, b) >= 0.0

    def test_triangle_inequality_over_random_triples(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            a, b, c = (_labels(120, rng.integers(2, 12), rng.integers(0, 9999)) for _ in range(3))
            ac = cs.variation_of_information(a, c, normalise=False)
            ab = cs.variation_of_information(a, b, normalise=False)
            bc = cs.variation_of_information(b, c, normalise=False)
            assert ac <= ab + bc + 1e-9

    def test_the_extreme_pair_equals_log_n(self):
        """One cluster vs all singletons is the maximum, which pins the log(n) normaliser to 1."""
        n = 64
        one = np.zeros(n, dtype=int)
        singletons = np.arange(n)
        assert cs.variation_of_information(one, singletons, normalise=False) == pytest.approx(np.log(n))
        assert cs.variation_of_information(one, singletons) == pytest.approx(1.0)


class TestPartitionAgreement:
    def test_identical_partitions_score_at_ceiling(self):
        a = _labels(200, 5, 0)
        out = cs.partition_agreement(a, a.copy())
        assert out["vi"] == 0.0 and out["vi_norm"] == 0.0
        assert out["ari"] == pytest.approx(1.0) and out["ami"] == pytest.approx(1.0)

    def test_independent_partitions_score_at_chance(self):
        """ARI/AMI are adjusted for chance, so ~0 is exactly what independence must give."""
        out = cs.partition_agreement(_labels(600, 10, 5), _labels(600, 10, 6))
        assert abs(out["ari"]) < 0.05
        assert abs(out["ami"]) < 0.05

    def test_label_permutation_is_irrelevant(self):
        a = _labels(200, 4, 7)
        relabelled = np.array([{0: 3, 1: 0, 2: 1, 3: 2}[x] for x in a])
        out = cs.partition_agreement(a, relabelled)
        assert out["vi"] == 0.0 and out["ari"] == pytest.approx(1.0)


# --------------------------------------------------------------------- planted recovery

@pytest.mark.parametrize("method", cs.DEFAULT_LINKAGES)
def test_two_noisy_views_of_planted_blobs_agree(method):
    """Three well-separated blobs, two independent noisy realisations, cut at the true k."""
    truth = _blobs(n_per=20, k=3, sd=0.25, seed=0)
    a, b = pdist(_noisy(truth, 0.1, 1)), pdist(_noisy(truth, 0.1, 2))
    la = cs.cut_tree(cs.build_linkage(a, method), [3], 60)[3]
    lb = cs.cut_tree(cs.build_linkage(b, method), [3], 60)[3]
    out = cs.partition_agreement(la, lb)
    assert out["ari"] > 0.99, f"{method}: ARI {out['ari']}"
    assert out["vi_norm"] < 0.01, f"{method}: VI {out['vi_norm']}"


def test_planted_blobs_have_high_cross_cohort_silhouette():
    truth = _blobs(n_per=20, k=3, sd=0.25, seed=0)
    a, b = pdist(_noisy(truth, 0.1, 1)), pdist(_noisy(truth, 0.1, 2))
    la = cs.cut_tree(cs.build_linkage(a, "average"), [3], 60)[3]
    lb = cs.cut_tree(cs.build_linkage(b, "average"), [3], 60)[3]
    sil = cs.silhouette_pair(squareform(a), squareform(b), la, lb)
    assert sil["sil_cross"] > 0.7
    assert sil["sil_ratio"] == pytest.approx(1.0, abs=0.1), "separation must survive the swap"


# --------------------------------------------------------------------- the continuum case
#
# The failure mode the design must be able to REPORT: cohorts can reproducibly agree on an
# arbitrary slicing of a space with no clusters in it.

def test_a_continuum_gives_agreement_without_separation():
    truth = _continuum(n=60, seed=0)
    a, b = pdist(_noisy(truth, 0.02, 1)), pdist(_noisy(truth, 0.02, 2))
    agreeing, unseparated = 0, 0
    for k in (3, 5, 8, 12):
        la = cs.cut_tree(cs.build_linkage(a, "average"), [k], 60)[k]
        lb = cs.cut_tree(cs.build_linkage(b, "average"), [k], 60)[k]
        if cs.partition_agreement(la, lb)["ari"] > 0.3:
            agreeing += 1
        if cs.silhouette_pair(squareform(a), squareform(b), la, lb)["sil_cross"] < 0.5:
            unseparated += 1
    assert agreeing > 0, "low-noise cohorts should still agree on *some* slicing"
    assert unseparated == 4, "but a continuum must never look well-separated"


def test_separation_distinguishes_blobs_from_a_continuum():
    """The single comparison that justifies carrying silhouette alongside VI."""
    def cross_sil(coords):
        a, b = pdist(_noisy(coords, 0.05, 1)), pdist(_noisy(coords, 0.05, 2))
        la = cs.cut_tree(cs.build_linkage(a, "average"), [3], len(coords))[3]
        lb = cs.cut_tree(cs.build_linkage(b, "average"), [3], len(coords))[3]
        return cs.silhouette_pair(squareform(a), squareform(b), la, lb)["sil_cross"]
    assert cross_sil(_blobs(n_per=20, k=3, sd=0.25, seed=0)) > 0.7
    assert cross_sil(_continuum(n=60, seed=0)) < 0.5


# --------------------------------------------------------------------- cluster-wise Jaccard

def test_cluster_wise_jaccard_matches_a_hand_computed_case():
    # A: {0,1} and {2,3}.  B: {0} and {1,2,3}.
    # A's cluster 0 = {0,1} vs best match {0}      -> 1/2
    # A's cluster 1 = {2,3} vs best match {1,2,3}  -> 2/3
    js = cs.cluster_wise_jaccard(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    np.testing.assert_allclose(js, [0.5, 2 / 3])


def test_jaccard_summary_reports_the_tail_not_just_the_mean():
    js = np.array([1.0, 1.0, 0.9, 0.2])
    out = cs.jaccard_summary(js)
    assert out["n_clusters"] == 4
    assert out["jaccard_mean"] == pytest.approx(0.775)
    assert out["frac_clusters_above_50"] == pytest.approx(0.75)
    assert out["frac_clusters_above_75"] == pytest.approx(0.75)


def test_identical_partitions_give_all_ones():
    a = _labels(100, 5, 0)
    np.testing.assert_allclose(cs.cluster_wise_jaccard(a, a.copy()), 1.0)


def test_one_stable_cluster_survives_an_otherwise_scrambled_partition():
    """The reason the distribution is returned rather than a mean."""
    a = np.array([0] * 10 + [1] * 10 + [2] * 10)
    b = np.array([0] * 10 + list(np.random.default_rng(0).integers(1, 5, 20)))
    js = cs.cluster_wise_jaccard(a, b)
    assert js[0] == pytest.approx(1.0), "the intact cluster must be identifiable"
    assert js[1:].max() < 0.8, "the scrambled ones must not be"


# --------------------------------------------------------------------- silhouette guards

def test_silhouette_returns_nan_at_the_ends_of_the_k_grid():
    d = squareform(pdist(_continuum(n=20, seed=0)))
    assert np.isnan(cs.safe_silhouette(d, np.zeros(20, dtype=int)))      # k = 1
    assert np.isnan(cs.safe_silhouette(d, np.arange(20)))                # k = n


def test_silhouette_ratio_is_nan_when_the_denominator_vanishes():
    """The ratio is meaningless when within-cohort separation is ~0; NaN beats a huge number."""
    coords = _continuum(n=40, seed=3)
    d = squareform(pdist(coords))
    labels = _labels(40, 4, 9)          # random labels on a continuum -> silhouette near 0
    out = cs.silhouette_pair(d, d, labels, labels)
    if out["sil_within"] < cs._MIN_WITHIN_SILHOUETTE:
        assert np.isnan(out["sil_ratio"])


def test_silhouette_ratio_is_nan_when_the_denominator_is_negative():
    """A negative within-silhouette means there was no separation to lose, so the ratio is nonsense.

    Guarding on `abs(within)` instead of `within` lets a negative denominator through, and on real
    isotropic data at k=12 that produced sil_ratio = +1.84 from cross = -0.043 over within = -0.023,
    i.e. the metric claimed separation *improved* out of sample.
    """
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(120, 8))          # isotropic: no clusters at any k
    d = squareform(pdist(coords))
    z = cs.build_linkage(pdist(coords), "average")
    for k in (12, 20, 30):
        labels = cs.cut_tree(z, [k], 120)[k]
        out = cs.silhouette_pair(d, d, labels, labels)
        if out["sil_within"] < 0:
            assert np.isnan(out["sil_ratio"]), f"k={k}: negative denominator leaked through"


def test_cross_silhouette_is_directional():
    a, b = _blobs(seed=0), _blobs(seed=1)
    da, db = squareform(pdist(a)), squareform(pdist(b))
    la = cs.cut_tree(cs.build_linkage(pdist(a), "average"), [3], 60)[3]
    lb = cs.cut_tree(cs.build_linkage(pdist(b), "average"), [3], 60)[3]
    out = cs.silhouette_pair(da, db, la, lb)
    assert out["sil_cross_ab"] != out["sil_cross_ba"], "both directions must be computed"
    assert out["sil_cross"] == pytest.approx(np.nanmean([out["sil_cross_ab"], out["sil_cross_ba"]]))


# --------------------------------------------------------------------- dendrogram agreement

def test_baker_gamma_is_one_for_identical_trees():
    z = cs.build_linkage(pdist(_blobs(seed=0)), "average")
    assert cs.baker_gamma(cs.cophenetic_ranks(z), cs.cophenetic_ranks(z)) == pytest.approx(1.0)


def test_baker_gamma_is_scale_invariant():
    """Rank-based, so an arbitrary rescaling of an MDS solution must not move it."""
    d = pdist(_blobs(seed=0))
    za, zb = cs.build_linkage(d, "average"), cs.build_linkage(d * 37.0, "average")
    assert cs.baker_gamma(cs.cophenetic_ranks(za), cs.cophenetic_ranks(zb)) == pytest.approx(1.0)


def test_baker_gamma_falls_for_unrelated_trees():
    za = cs.build_linkage(pdist(_blobs(seed=0)), "average")
    zb = cs.build_linkage(pdist(_continuum(n=60, seed=1)), "average")
    assert cs.baker_gamma(cs.cophenetic_ranks(za), cs.cophenetic_ranks(zb)) < 0.8


def test_cophenetic_fidelity_is_high_for_a_clusterable_space():
    d = pdist(_blobs(n_per=20, k=3, sd=0.2, seed=0))
    assert cs.cophenetic_fidelity(cs.build_linkage(d, "average"), d) > 0.8


# --------------------------------------------------------------------- sizes and cutting

def test_cluster_size_summary_describes_a_known_partition():
    out = cs.cluster_size_summary(np.array([0] * 8 + [1] * 2 + [2]))
    assert out["n_clusters_realised"] == 3
    assert out["size_min"] == 1 and out["size_max"] == 8
    assert out["largest_frac"] == pytest.approx(8 / 11)
    assert out["frac_singletons"] == pytest.approx(1 / 3)


def test_size_entropy_is_one_for_a_balanced_partition():
    balanced = cs.cluster_size_summary(np.repeat(np.arange(4), 10))
    assert balanced["size_entropy_norm"] == pytest.approx(1.0)


def test_cut_tree_drops_impossible_k_with_a_warning_not_an_error():
    z = cs.build_linkage(pdist(_blobs(n_per=5, k=2, seed=0)), "average")
    with pytest.warns(UserWarning, match="exceeding n_items"):
        cuts = cs.cut_tree(z, [2, 5, 500], n=10)
    assert set(cuts) == {2, 5}


def test_realised_cluster_count_is_recorded_not_assumed():
    """maxclust can return fewer than k under merge-height ties, so k is a request, not a promise."""
    z = cs.build_linkage(pdist(_blobs(n_per=20, k=3, sd=0.05, seed=0)), "average")
    for k, labels in cs.cut_tree(z, [2, 3, 5], 60).items():
        assert np.unique(labels).size <= k


def test_unknown_linkage_is_rejected():
    with pytest.raises(ValueError, match="method must be one of"):
        cs.build_linkage(pdist(_blobs(seed=0)), "nonsense")


# --------------------------------------------------------------------- the pair-level entry point

def test_compare_partitions_covers_the_grid():
    a, b = pdist(_blobs(seed=0)), pdist(_blobs(seed=1))
    df = cs.compare_partitions(a, b, ks=(2, 3, 5), linkages=("average", "ward"))
    assert len(df) == 6
    assert set(df["linkage"]) == {"average", "ward"}
    assert set(df["k"]) == {2, 3, 5}
    for col in ("vi", "vi_norm", "ari", "ami", "sil_cross", "sil_ratio",
                "jaccard_mean", "baker_gamma", "n_clusters_realised"):
        assert col in df.columns, col


def test_compare_partitions_scores_a_cohort_against_itself_perfectly():
    d = pdist(_blobs(seed=0))
    df = cs.compare_partitions(d, d.copy(), ks=(3,), linkages=("average",))
    row = df.iloc[0]
    assert row["vi"] == 0.0 and row["ari"] == pytest.approx(1.0)
    assert row["jaccard_mean"] == pytest.approx(1.0)
    assert row["baker_gamma"] == pytest.approx(1.0)
    assert row["sil_ratio"] == pytest.approx(1.0)


def test_compare_partitions_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        cs.compare_partitions(pdist(_blobs(seed=0)), pdist(_continuum(n=30, seed=0)))


def test_baker_gamma_is_constant_across_k_within_a_linkage():
    """It is a whole-tree statistic, so cutting at different k must not change it."""
    a, b = pdist(_blobs(seed=0)), pdist(_blobs(seed=1))
    df = cs.compare_partitions(a, b, ks=(2, 3, 5), linkages=("average",))
    assert df["baker_gamma"].nunique() == 1


# --------------------------------------------------------------------- store-level drivers

N_IMAGES, MAX_NDIM = 30, 4
META = ["num_subjects", "rep", "ndim", "niter", "stress", "status"]


def _conf_store(tmp_path, n_reps=4, num_subjects=(20, 50), ndim=3, statuses=None, seed=0):
    """A store whose cohorts are noisy views of one planted 3-blob structure."""
    from SpAM_Simulations.storage import ResultStore
    rng = np.random.default_rng(seed)
    truth = _blobs(n_per=N_IMAGES // 3, k=3, sd=0.25, seed=seed, ndim=MAX_NDIM)
    store = ResultStore.create(tmp_path / "s", confdist_len=N_IMAGES * (N_IMAGES - 1) // 2,
                               meta_columns=META, n_images=N_IMAGES, max_ndim=MAX_NDIM)
    for ns in num_subjects:
        for rep in range(n_reps):
            status = (statuses or {}).get((ns, rep), "success")
            meta = {"num_subjects": ns, "rep": rep, "ndim": ndim, "niter": 10,
                    "stress": 0.1, "status": status}
            if status not in ("success", "max_iters"):
                store.append(meta)
                continue
            coords = (truth + rng.normal(0, 0.1, truth.shape)).astype(np.float32)
            store.append(meta, confdist=pdist(coords).astype(np.float32), conf=coords)
    store.close()
    return ResultStore.open(tmp_path / "s")


def test_cluster_agreement_emits_one_row_per_group_linkage_k(tmp_path):
    store = _conf_store(tmp_path)
    df = cs.compute_cluster_agreement(store, ks=(2, 3, 5), linkages=("average", "ward"),
                                      verbose=False)
    assert len(df) == 2 * 2 * 3            # 2 num_subjects x 2 linkages x 3 k
    assert set(df["num_subjects"]) == {20, 50}
    assert (df["n_pairs"] == 6).all()       # C(4, 2)
    for col in ("mean_vi_norm", "sem_vi_norm", "mean_ari", "mean_sil_cross", "mean_jaccard_mean"):
        assert col in df.columns, col


def test_cluster_agreement_recovers_the_planted_structure(tmp_path):
    store = _conf_store(tmp_path)
    df = cs.compute_cluster_agreement(store, ks=(3,), linkages=("average",), verbose=False)
    assert (df["mean_ari"] > 0.9).all()
    assert (df["mean_vi_norm"] < 0.1).all()


def test_drivers_exclude_failed_fits(tmp_path):
    """Only successful, confdist-bearing rows may be compared."""
    store = _conf_store(tmp_path, n_reps=4, num_subjects=(20,),
                        statuses={(20, 0): "error", (20, 1): "disconnected"})
    df = cs.compute_cluster_agreement(store, ks=(3,), linkages=("average",), verbose=False)
    assert df["n_reps"].iloc[0] == 2 and df["n_pairs"].iloc[0] == 1


def test_a_group_with_one_usable_rep_is_skipped_not_crashed(tmp_path):
    store = _conf_store(tmp_path, n_reps=2, num_subjects=(20,), statuses={(20, 0): "error"})
    df = cs.compute_cluster_agreement(store, ks=(3,), linkages=("average",), verbose=False)
    assert df.empty


def test_drivers_reject_a_store_without_configurations(tmp_path):
    from SpAM_Simulations.storage import ResultStore
    store = ResultStore.create(tmp_path / "nc", confdist_len=10, meta_columns=META)
    store.append({"num_subjects": 1, "rep": 0, "ndim": 2, "niter": 1, "stress": 0.1,
                  "status": "success"}, confdist=np.zeros(10, dtype=np.float32))
    store.close()
    with pytest.raises(ValueError, match="cannot be clustered"):
        cs.compute_cluster_agreement(ResultStore.open(tmp_path / "nc"), verbose=False)


def test_dendrogram_agreement_is_k_free(tmp_path):
    store = _conf_store(tmp_path)
    df = cs.compute_dendrogram_agreement(store, linkages=("average", "complete"), verbose=False)
    assert "k" not in df.columns
    assert len(df) == 2 * 2                 # 2 num_subjects x 2 linkages
    assert (df["mean_baker_gamma"] > 0.5).all()
    assert "mean_cophenetic_fidelity" in df.columns


def test_cluster_sizes_describe_the_discovered_partition(tmp_path):
    store = _conf_store(tmp_path)
    df = cs.compute_cluster_sizes(store, ks=(3,), linkages=("average",), verbose=False)
    assert (df["mean_n_clusters_realised"] == 3).all()
    assert (df["mean_size_max"] <= N_IMAGES).all()
    assert "mean_size_entropy_norm" in df.columns


def test_drivers_read_conf_not_confdist(tmp_path):
    """The whole point of the conf-only download: clustering must not touch confdists.f32."""
    store = _conf_store(tmp_path)
    (store.path / "confdists.f32").unlink()
    from SpAM_Simulations.storage import ResultStore
    reopened = ResultStore.open(store.path)
    assert not reopened.has_confdists
    df = cs.compute_cluster_agreement(reopened, ks=(3,), linkages=("average",), verbose=False)
    assert len(df) == 2 and df["mean_ari"].notna().all()


def test_padded_conf_rows_are_trimmed_to_ndim(tmp_path):
    """Rows are stored zero-padded to max_ndim; clustering must use only the fit's own ndim."""
    store = _conf_store(tmp_path, ndim=2)
    df = cs.compute_cluster_agreement(store, ks=(3,), linkages=("average",), verbose=False)
    assert df["mean_ari"].notna().all(), "padding columns would make every point coincide"


# --------------------------------------------------------------------- k selection / diagnostics

def _agreement_frame(vi_by_k, sil=0.6, group="a"):
    """A minimal agreement frame with per-k means, for exercising the selection logic."""
    return pd.DataFrame([
        {"num_subjects": 50, "ndim": 5, "linkage": group, "k": k,
         "mean_vi_norm": vi, "sem_vi_norm": 0.001,
         "mean_sil_cross": sil, "mean_sil_ratio": 0.9}
        for k, vi in vi_by_k.items()
    ])


def test_select_k_prefers_the_smaller_of_two_indistinguishable_granularities():
    """The one-SE rule again: a flat curve must not drift to fine k on noise."""
    df = _agreement_frame({3: 0.20, 10: 0.199, 50: 0.198})
    df["sem_vi_norm"] = 0.05
    assert cs.select_k(df)["k_star"].iloc[0] == 3


def test_select_k_follows_a_real_optimum():
    df = _agreement_frame({3: 0.60, 10: 0.10, 50: 0.62})
    assert cs.select_k(df)["k_star"].iloc[0] == 10


def test_select_k_respects_vi_being_lower_is_better():
    """Selecting on VI must not invert: the argmax rule should land on the MINIMUM."""
    df = _agreement_frame({3: 0.60, 10: 0.10, 50: 0.62})
    assert cs.select_k(df, rule="argmax")["k_star"].iloc[0] == 10


def test_select_k_emits_one_row_per_group():
    df = pd.concat([_agreement_frame({3: 0.5, 10: 0.2}, group="average"),
                    _agreement_frame({3: 0.2, 10: 0.5}, group="ward")])
    out = cs.select_k(df)
    assert len(out) == 2
    assert set(out["linkage"]) == {"average", "ward"}


def test_select_k_rejects_a_missing_criterion():
    with pytest.raises(ValueError, match="mean_nonsense"):
        cs.select_k(_agreement_frame({3: 0.5}), criterion="nonsense")


def test_diagnostics_flag_a_flat_curve():
    """No granularity distinguishably better than any other means there is no k* to find."""
    out = cs.continuum_diagnostics(_agreement_frame({3: 0.400, 10: 0.401, 50: 0.402}))
    assert out["is_flat"].iloc[0]


def test_diagnostics_flag_agreement_without_separation():
    """The continuum signature: cohorts agree on a cut of a space that has no clusters in it."""
    out = cs.continuum_diagnostics(_agreement_frame({3: 0.05, 10: 0.40, 50: 0.60}, sil=0.01))
    assert out["is_arbitrary_slicing"].iloc[0]
    assert not out["is_flat"].iloc[0], "a real gradient must not be called flat"


def test_diagnostics_pass_a_genuinely_clustered_case():
    out = cs.continuum_diagnostics(_agreement_frame({3: 0.02, 10: 0.40, 50: 0.70}, sil=0.65))
    assert not out["is_flat"].iloc[0]
    assert not out["is_arbitrary_slicing"].iloc[0]
    assert out["k_star"].iloc[0] == 3
    assert out["vi_norm_at_k_star"].iloc[0] == pytest.approx(0.02)


def test_diagnostics_report_the_range_used_for_the_flat_verdict():
    out = cs.continuum_diagnostics(_agreement_frame({3: 0.10, 10: 0.50}))
    assert out["vi_norm_range"].iloc[0] == pytest.approx(0.40)


def test_diagnostics_end_to_end_on_a_store(tmp_path):
    store = _conf_store(tmp_path)
    agreement = cs.compute_cluster_agreement(store, ks=(2, 3, 5), linkages=("average",),
                                             verbose=False)
    out = cs.continuum_diagnostics(agreement)
    assert len(out) == 2                      # one row per num_subjects
    for col in ("k_star", "vi_norm_range", "sil_cross_at_k_star", "is_flat",
                "is_arbitrary_slicing"):
        assert col in out.columns, col
