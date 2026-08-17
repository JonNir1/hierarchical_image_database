"""Tests for the density-based descriptive pass.

The load-bearing property is the one agglomerative clustering cannot provide: an image that belongs
to no cluster must be *labelled* as such rather than absorbed into the nearest group. Most of these
tests therefore plant known isolated points and check they come back as noise.
"""
import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from SpAM_Simulations.measures import density_clustering as dc

N_IMAGES, MAX_NDIM = 60, 4
META = ["num_subjects", "rep", "ndim", "niter", "stress", "status"]


def _blobs_with_outliers(n_per=18, k=3, n_outliers=6, sd=0.2, seed=0, ndim=MAX_NDIM):
    """``k`` tight blobs plus ``n_outliers`` scattered points that belong to none of them."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 10, size=(k, ndim))
    pts = [c + rng.normal(0, sd, size=(n_per, ndim)) for c in centres]
    pts.append(rng.normal(0, 25, size=(n_outliers, ndim)))
    return np.vstack(pts).astype(np.float32)


# --------------------------------------------------------------------- labelling

def test_isolated_points_are_labelled_noise_not_absorbed():
    """The whole reason this module exists: AC would put these outliers in some cluster."""
    coords = _blobs_with_outliers(n_per=18, k=3, n_outliers=6)
    labels = dc.hdbscan_labels(pdist(coords), min_cluster_size=5)
    assert len(set(labels) - {dc.NOISE_LABEL}) == 3
    # The last 6 points are the planted outliers.
    assert (labels[-6:] == dc.NOISE_LABEL).all()
    assert (labels[:-6] != dc.NOISE_LABEL).all()


def test_labelling_is_deterministic():
    """Any RNG in the clusterer would inflate apparent between-cohort disagreement."""
    d = pdist(_blobs_with_outliers())
    np.testing.assert_array_equal(dc.hdbscan_labels(d, 5), dc.hdbscan_labels(d, 5))


def test_min_cluster_size_below_two_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        dc.hdbscan_labels(pdist(_blobs_with_outliers()), min_cluster_size=1)


def test_the_distance_matrix_is_not_modified_in_place():
    """sklearn's `copy` defaults to False; callers here reuse a cached square matrix."""
    condensed = pdist(_blobs_with_outliers())
    before = squareform(condensed.astype(np.float64)).copy()
    dc.hdbscan_labels(condensed, 5)
    np.testing.assert_array_equal(squareform(condensed.astype(np.float64)), before)


# --------------------------------------------------------------------- summaries

def test_noise_summary_counts_clusters_and_noise():
    labels = np.array([0, 0, 0, 1, 1, -1, -1])
    s = dc.noise_summary(labels)
    assert s == {"n_images": 7, "n_clusters": 2, "frac_noise": pytest.approx(2 / 7),
                 "n_clustered": 5, "cluster_size_mean": pytest.approx(2.5),
                 "cluster_size_median": pytest.approx(2.5), "cluster_size_max": 3}


def test_noise_summary_handles_an_all_noise_labelling():
    s = dc.noise_summary(np.full(5, dc.NOISE_LABEL))
    assert s["n_clusters"] == 0 and s["frac_noise"] == 1.0 and s["cluster_size_max"] == 0
    assert np.isnan(s["cluster_size_mean"])


# --------------------------------------------------------------------- noise agreement

def test_noise_agreement_on_identical_labellings():
    a = np.array([0, 0, 1, -1, -1])
    out = dc.noise_agreement(a, a.copy())
    assert out["noise_jaccard"] == 1.0
    assert out["noise_kappa"] == pytest.approx(1.0)
    assert out["both_noise_frac"] == pytest.approx(0.4)


def test_noise_agreement_on_disjoint_noise_sets():
    out = dc.noise_agreement(np.array([-1, -1, 0, 0]), np.array([0, 0, -1, -1]))
    assert out["noise_jaccard"] == 0.0
    assert out["both_noise_frac"] == 0.0
    assert out["either_noise_frac"] == 1.0


def test_noise_agreement_is_nan_when_neither_cohort_flags_anything():
    """0/0 is not perfect agreement; reporting 1.0 would assert agreement about nothing."""
    out = dc.noise_agreement(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]))
    assert np.isnan(out["noise_jaccard"])
    assert np.isnan(out["noise_kappa"])


def test_noise_agreement_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="match in length"):
        dc.noise_agreement(np.zeros(4), np.zeros(5))


def test_hand_computed_noise_jaccard():
    # noise sets {0,1,2} and {2,3}: intersection {2}, union {0,1,2,3} -> 1/4
    a = np.array([-1, -1, -1, 0, 0])
    b = np.array([0, 0, -1, -1, 0])
    assert dc.noise_agreement(a, b)["noise_jaccard"] == pytest.approx(0.25)


# --------------------------------------------------------------------- restricted ARI

def test_clustered_ari_ignores_images_either_cohort_called_noise():
    """Including noise as a cluster would let 'most things are isolated' score as agreement."""
    a = np.array([0, 0, 1, 1, -1])
    b = np.array([0, 0, 1, 1, 3])
    out = dc.clustered_ari(a, b)
    assert out["n_shared_clustered"] == 4
    assert out["ari_shared_clustered"] == pytest.approx(1.0)
    assert out["frac_shared_clustered"] == pytest.approx(0.8)


def test_clustered_ari_is_nan_when_almost_nothing_is_jointly_clustered():
    out = dc.clustered_ari(np.array([-1, -1, 0]), np.array([0, -1, -1]))
    assert np.isnan(out["ari_shared_clustered"])
    assert out["n_shared_clustered"] < 2


# --------------------------------------------------------------------- pair comparison

def test_compare_density_partitions_one_row_per_setting():
    a = _blobs_with_outliers(seed=1)
    b = _blobs_with_outliers(seed=1) + np.random.default_rng(2).normal(0, 0.05, (N_IMAGES, MAX_NDIM))
    out = dc.compare_density_partitions(pdist(a), pdist(b), min_cluster_sizes=(3, 5, 10))
    assert list(out["min_cluster_size"]) == [3, 5, 10]
    for col in ("mean_frac_noise", "noise_jaccard", "noise_kappa", "ari_shared_clustered"):
        assert col in out.columns
    # Two near-identical cohorts should agree strongly on both structure and isolation.
    assert (out["ari_shared_clustered"] > 0.9).all()
    assert (out["noise_jaccard"] > 0.5).all()


def test_two_cohorts_agree_on_which_images_are_isolated():
    """The claim the module is for: isolation is a reproducible per-image property."""
    truth = _blobs_with_outliers(seed=3)
    rng = np.random.default_rng(4)
    a = truth + rng.normal(0, 0.05, truth.shape)
    b = truth + rng.normal(0, 0.05, truth.shape)
    la, lb = dc.hdbscan_labels(pdist(a), 5), dc.hdbscan_labels(pdist(b), 5)
    assert dc.noise_agreement(la, lb)["noise_kappa"] > 0.8


# --------------------------------------------------------------------- store drivers

def _store(tmp_path, n_reps=3, seed=0):
    from SpAM_Simulations.core.storage import ResultStore
    rng = np.random.default_rng(seed)
    truth = _blobs_with_outliers(seed=seed)
    store = ResultStore.create(tmp_path / "s", confdist_len=N_IMAGES * (N_IMAGES - 1) // 2,
                               meta_columns=META, n_images=N_IMAGES, max_ndim=MAX_NDIM)
    for rep in range(n_reps):
        coords = (truth + rng.normal(0, 0.05, truth.shape)).astype(np.float32)
        store.append({"num_subjects": 50, "rep": rep, "ndim": MAX_NDIM, "niter": 10,
                      "stress": 0.1, "status": "success"},
                     confdist=pdist(coords).astype(np.float32), conf=coords)
    store.close()
    return ResultStore.open(tmp_path / "s")


def test_compute_density_agreement_emits_one_row_per_group_and_setting(tmp_path):
    df = dc.compute_density_agreement(_store(tmp_path), min_cluster_sizes=(3, 5), verbose=False)
    assert len(df) == 2
    assert (df["n_pairs"] == 3).all()          # C(3, 2)
    for col in ("mean_frac_noise", "mean_noise_kappa", "mean_ari_shared_clustered",
                "mean_n_clusters", "mean_cluster_size"):
        assert col in df.columns, col
    assert "mean_cluster_size_mean" not in df.columns   # the field is renamed, not doubled


def test_density_agreement_skips_a_single_rep_group(tmp_path):
    df = dc.compute_density_agreement(_store(tmp_path, n_reps=1), min_cluster_sizes=(5,),
                                      verbose=False)
    assert df.empty


def test_density_agreement_needs_configurations(tmp_path):
    from SpAM_Simulations.core.storage import ResultStore
    store = ResultStore.create(tmp_path / "nc", confdist_len=10, meta_columns=META)
    store.append({"num_subjects": 1, "rep": 0, "ndim": 2, "niter": 1, "stress": 0.1,
                  "status": "success"}, confdist=np.zeros(10, dtype=np.float32))
    store.close()
    with pytest.raises(ValueError, match="cannot be clustered"):
        dc.compute_density_agreement(ResultStore.open(tmp_path / "nc"), verbose=False)


def test_isolated_images_is_per_image_and_flags_the_planted_outliers(tmp_path):
    df = dc.isolated_images(_store(tmp_path), min_cluster_size=5, verbose=False)
    assert len(df) == N_IMAGES
    assert set(df["image"]) == set(range(N_IMAGES))
    assert df["frac_cohorts_noise"].between(0, 1).all()
    # The last 6 planted points are isolated in every cohort; the blob members in none.
    outliers = df[df["image"] >= N_IMAGES - 6]
    members = df[df["image"] < N_IMAGES - 6]
    assert (outliers["frac_cohorts_noise"] == 1.0).all()
    assert (members["frac_cohorts_noise"] == 0.0).all()


def test_mean_sem_ignores_nans():
    """NaN is a legitimate value here (undefined Jaccard), not a failure to be propagated."""
    out = dc._mean_sem([1.0, np.nan, 3.0], "x")
    assert out["mean_x"] == pytest.approx(2.0)
    assert out["sem_x"] > 0
    assert np.isnan(dc._mean_sem([np.nan, np.nan], "x")["mean_x"])


# --------------------------------------------------------------------- restricted VI
# The point of these: a noise class blocks VI on the FULL set, but restricting the ground set to
# the images every labelling clustered restores a genuine partition metric on that subset, which
# supports a real chained claim in a narrower, explicitly-scoped form.

def test_common_clustered_mask_intersects_every_labelling():
    a = np.array([0, 0, -1, 1, 1])
    b = np.array([0, -1, 0, 1, 1])
    c = np.array([0, 0, 0, 1, -1])
    np.testing.assert_array_equal(dc.common_clustered_mask([a, b, c]),
                                  np.array([True, False, False, True, False]))


def test_common_clustered_mask_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="match in length"):
        dc.common_clustered_mask([np.zeros(4), np.zeros(5)])


def test_restricted_vi_is_zero_for_labellings_that_agree_where_both_clustered():
    """Noise placement may differ; what is scored is the structure on the surviving subset."""
    a = np.array([0, 0, 1, 1, -1, 2, 2])
    b = np.array([5, 5, 7, 7, 3, -1, 9])
    # Index 4 is noise in a, index 5 is noise in b, so both drop; the surviving {0,1,2,3,6} carry
    # the same grouping under both labellings despite entirely different cluster numbering.
    out = dc.restricted_vi(a, b)
    assert out["n_shared"] == 5
    assert out["vi_restricted"] == pytest.approx(0.0, abs=1e-12)
    assert out["vi_restricted_norm"] == pytest.approx(0.0, abs=1e-12)


def test_restricted_vi_is_nan_when_the_shared_subset_is_too_small():
    """log(1) == 0 makes the normalised form undefined, and one item has no structure anyway."""
    out = dc.restricted_vi(np.array([-1, -1, 0]), np.array([0, -1, -1]))
    assert out["n_shared"] == 0
    assert np.isnan(out["vi_restricted"]) and np.isnan(out["vi_restricted_norm"])


def test_restricted_vi_matches_plain_vi_when_nothing_is_noise():
    """With no noise the restriction is a no-op, so it must reduce to the ordinary metric."""
    from SpAM_Simulations.measures.cluster_stability import variation_of_information
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 1, 1, 1, 2, 2])
    out = dc.restricted_vi(a, b)
    assert out["n_shared"] == 6
    assert out["vi_restricted"] == pytest.approx(variation_of_information(a, b, normalise=False))
    assert out["vi_restricted_norm"] == pytest.approx(variation_of_information(a, b))


def test_restricted_vi_honours_an_explicit_shared_mask():
    """The mask is how several labellings are forced onto ONE ground set for a chained claim."""
    a = np.array([0, 0, 1, 1, 2])
    b = np.array([0, 0, 1, 1, 2])
    mask = np.array([True, True, False, False, False])
    out = dc.restricted_vi(a, b, mask=mask)
    assert out["n_shared"] == 2
    assert out["vi_restricted"] == pytest.approx(0.0, abs=1e-12)


def test_pairwise_restricted_vi_obeys_the_triangle_inequality():
    """The property the whole restriction exists to buy.

    On one shared ground set the three restricted VIs live in a single metric space, so any leg is
    bounded by the sum of the other two. That is what lets an unmeasured cohort-to-reference
    distance be bounded by two measured ones, in the scoped form "restricted to these n images".
    """
    rng = np.random.default_rng(0)
    labellings = []
    for _ in range(3):
        la = rng.integers(0, 4, size=40)
        la[rng.random(40) < 0.2] = dc.NOISE_LABEL      # each marks a different ~20% as noise
        labellings.append(la)
    out = dc.pairwise_restricted_vi(labellings, names=["a", "b", "c"]).set_index(["a", "b"])
    ab, ac, bc = (out.loc[k, "vi_restricted"] for k in (("a", "b"), ("a", "c"), ("b", "c")))
    assert ac <= ab + bc + 1e-12
    assert ab <= ac + bc + 1e-12
    assert bc <= ab + ac + 1e-12
    # The scope of the claim must travel with it: one ground set, reported.
    assert out["n_shared"].nunique() == 1
    assert out["n_shared"].iloc[0] < 40      # the restriction really did drop images


def test_pairwise_restricted_vi_uses_one_ground_set_not_per_pair_intersections():
    """Per-pair intersections would put the terms in different metric spaces and break addition."""
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 0, 1, 1, 2, -1])
    c = np.array([0, 0, 1, 1, -1, -1])
    out = dc.pairwise_restricted_vi([a, b, c])
    assert (out["n_shared"] == 4).all()      # the a-b pair alone would have kept 5
    assert dc.restricted_vi(a, b)["n_shared"] == 5


def test_pairwise_restricted_vi_rejects_a_name_count_mismatch():
    with pytest.raises(ValueError, match="names for"):
        dc.pairwise_restricted_vi([np.zeros(3, int), np.zeros(3, int)], names=["only-one"])


def test_density_agreement_reports_restricted_vi(tmp_path):
    df = dc.compute_density_agreement(_store(tmp_path), min_cluster_sizes=(5,), verbose=False)
    for col in ("mean_vi_restricted", "sem_vi_restricted", "mean_vi_restricted_norm"):
        assert col in df.columns, col
    # Near-identical cohorts: the surviving subset should agree almost perfectly.
    assert df["mean_vi_restricted"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert "mean_n_shared" not in df.columns      # frac_shared_clustered already carries this
