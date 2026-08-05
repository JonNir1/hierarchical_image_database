"""End-to-end tests for the local cluster-analysis driver (pipeline steps g-i).

The driver is thin, so these check the two things that are actually its job: that it produces the
four CSVs the notebook and ``eval_helpers.load_run`` expect, and that a **conf-only** store - the
one the download command actually produces, with ``confdists.f32`` deliberately left on S3 - runs
through it unchanged.
"""
import numpy as np
import pytest
from scipy.spatial.distance import pdist

from SpAM_Simulations import run_cluster_analysis as rca
from SpAM_Simulations.storage import ResultStore

N_IMAGES, MAX_NDIM = 45, 4
META = ["num_subjects", "allocation_mode", "rep", "ndim", "niter", "stress", "status"]
OUTPUTS = ("cluster_agreement.csv", "dendrogram_agreement.csv", "cluster_sizes.csv",
           "k_selection.csv")


def _blobs(n_per, k, sd, seed, ndim):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 6, size=(k, ndim))
    return np.vstack([c + rng.normal(0, sd, size=(n_per, ndim)) for c in centres]).astype(np.float32)


def _store(tmp_path, *, n_reps=3, arms=(0.0, 1.0), sd=0.25, seed=0):
    """A two-arm store of noisy realisations of the same three planted blobs."""
    rng = np.random.default_rng(seed)
    truth = _blobs(N_IMAGES // 3, 3, sd, seed, MAX_NDIM)
    store = ResultStore.create(tmp_path / "s", confdist_len=N_IMAGES * (N_IMAGES - 1) // 2,
                               meta_columns=META, n_images=N_IMAGES, max_ndim=MAX_NDIM)
    for arm in arms:
        for rep in range(n_reps):
            coords = (truth + rng.normal(0, 0.1, truth.shape)).astype(np.float32)
            store.append({"num_subjects": 50, "allocation_mode": arm, "rep": rep,
                          "ndim": MAX_NDIM, "niter": 10, "stress": 0.1, "status": "success"},
                         confdist=pdist(coords).astype(np.float32), conf=coords)
    store.close()
    return tmp_path / "s"


def test_run_writes_the_four_tables(tmp_path):
    out = tmp_path / "out"
    frames = rca.run(_store(tmp_path), out, ks=(2, 3, 5), linkages=("average", "ward"),
                     verbose=False)
    for name in OUTPUTS:
        assert (out / name).is_file(), name
    assert set(frames) == {"cluster_agreement", "dendrogram_agreement", "cluster_sizes",
                           "k_selection"}
    # 2 arms x 2 linkages x 3 k, with the arm kept as its own group rather than pooled away.
    assert len(frames["cluster_agreement"]) == 2 * 2 * 3
    assert set(frames["cluster_agreement"]["allocation_mode"]) == {0.0, 1.0}


def test_k_selection_keeps_the_arms_apart(tmp_path):
    """Pooling the arms would hide exactly the difference the sweep exists to measure."""
    frames = rca.run(_store(tmp_path), tmp_path / "out", ks=(2, 3, 5),
                     linkages=("average",), verbose=False)
    sel = frames["k_selection"]
    assert set(sel["allocation_mode"]) == {0.0, 1.0}
    assert len(sel) == 2
    for col in ("k_star_vi", "k_star_sil", "is_flat", "is_arbitrary_slicing",
                "sil_cross_at_k_star_vi"):
        assert col in sel.columns


def test_planted_blobs_are_not_flagged_as_a_continuum(tmp_path):
    """The positive control: separated blobs give high agreement, high silhouette, no verdict."""
    frames = rca.run(_store(tmp_path, sd=0.2), tmp_path / "out", ks=(2, 3, 5, 8),
                     linkages=("average",), verbose=False)
    sel = frames["k_selection"]
    assert not sel["is_flat"].any()
    assert not sel["is_arbitrary_slicing"].any()
    assert (sel["sil_cross_at_k_star_vi"] > 0.5).all()


def test_vi_does_not_identify_the_number_of_clusters_but_silhouette_does(tmp_path):
    """The reason `k_star_vi` and `k_star_sil` are both reported.

    VI measures *reproducibility*, not correctness of granularity. On three well-separated planted
    blobs, cutting at k=2 merges the same two blobs in every cohort, so VI is exactly 0 there just
    as it is at the true k=3 - and the one-SE parsimony tiebreak therefore returns 2. Cross-cohort
    silhouette is what separates the two: it peaks at 3. Reading k* off VI alone would systematically
    under-report the granularity the data actually supports.
    """
    frames = rca.run(_store(tmp_path, sd=0.2), tmp_path / "out", ks=(2, 3, 5, 8),
                     linkages=("average",), verbose=False)
    agreement = frames["cluster_agreement"]
    one_arm = agreement[agreement["allocation_mode"] == 0.0].set_index("k")
    assert one_arm.loc[2, "mean_vi_norm"] == pytest.approx(0.0, abs=1e-9)
    assert one_arm.loc[3, "mean_vi_norm"] == pytest.approx(0.0, abs=1e-9)
    assert one_arm.loc[3, "mean_sil_cross"] > one_arm.loc[2, "mean_sil_cross"]

    sel = frames["k_selection"]
    assert (sel["k_star_vi"] == 2).all()       # conservative: the coarsest cut that reproduces
    assert (sel["k_star_sil"] == 3).all()      # the true planted structure


def test_runs_on_a_conf_only_store(tmp_path):
    """The download command excludes confdists.f32 on purpose; the driver must not need it.

    Every distance here is recomputed as ``pdist(conf)``, which is what the stored row held anyway,
    so dropping the large file changes the results by nothing at all.
    """
    path = _store(tmp_path)
    full = rca.run(path, tmp_path / "out_full", ks=(2, 3), linkages=("average",), verbose=False)

    (path / "confdists.f32").unlink()
    reopened = ResultStore.open(path)
    assert not reopened.has_confdists
    partial = rca.run(path, tmp_path / "out_partial", ks=(2, 3), linkages=("average",),
                      verbose=False)

    for name in OUTPUTS:
        assert (tmp_path / "out_partial" / name).is_file(), name
    np.testing.assert_allclose(full["cluster_agreement"]["mean_vi_norm"].to_numpy(),
                               partial["cluster_agreement"]["mean_vi_norm"].to_numpy())


def test_main_parses_arguments_and_writes_output(tmp_path):
    path = _store(tmp_path, n_reps=2, arms=(0.0,))
    rc = rca.main(["--store", str(path), "--out", str(tmp_path / "cli"),
                   "--ks", "2,3", "--linkages", "average", "--quiet"])
    assert rc == 0
    assert (tmp_path / "cli" / "cluster_agreement.csv").is_file()


def test_select_by_drops_columns_the_frame_does_not_have(tmp_path):
    """A single-arm run has no `allocation_mode`; grouping must degrade rather than raise."""
    import pandas as pd
    assert rca._select_by(pd.DataFrame(columns=["num_subjects", "ndim", "linkage", "k"])) == [
        "num_subjects", "ndim", "linkage"]


def test_report_names_the_continuum_case(tmp_path, capsys):
    """A flat/arbitrary group must be announced, not quietly reported as a k*."""
    import pandas as pd
    k_selection = pd.DataFrame([{"num_subjects": 50, "ndim": 4, "linkage": "average", "k_star_vi": 3,
                                 "vi_norm_at_k_star_vi": 0.4, "sil_cross_at_k_star_vi": 0.01,
                                 "sil_ratio_at_k_star_vi": 0.1, "is_flat": True,
                                 "is_arbitrary_slicing": True}])
    rca._report(k_selection, pd.DataFrame())
    out = capsys.readouterr().out
    assert "CONTINUUM" in out
    assert "distance threshold" in out


@pytest.mark.parametrize("missing", ["store", "out"])
def test_main_requires_store_and_out(missing):
    args = {"store": "x", "out": "y"}
    args.pop(missing)
    with pytest.raises(SystemExit):
        rca.main([f"--{k}" for pair in args.items() for k in pair])
