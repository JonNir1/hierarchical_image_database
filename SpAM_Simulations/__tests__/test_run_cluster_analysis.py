"""End-to-end tests for the local cluster-analysis driver (pipeline steps g-i).

The driver is thin, so these check the two things that are actually its job: that it produces the
six CSVs the notebook and ``eval_helpers.load_run`` expect, and that a **conf-only** store - the
one the download command actually produces, with ``confdists.f32`` deliberately left on S3 - runs
through it unchanged.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from SpAM_Simulations import run_cluster_analysis as rca
from SpAM_Simulations.storage import ResultStore

N_IMAGES, MAX_NDIM = 45, 4
META = ["num_subjects", "allocation_mode", "rep", "ndim", "niter", "stress", "status"]
OUTPUTS = ("cluster_agreement.csv", "dendrogram_agreement.csv", "cluster_sizes.csv",
           "k_selection.csv", "density_agreement.csv", "isolated_images.csv")


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


def test_run_writes_every_table(tmp_path):
    out = tmp_path / "out"
    frames = rca.run(_store(tmp_path), out, ks=(2, 3, 5), linkages=("average", "ward"),
                     min_cluster_sizes=(3,), verbose=False)
    for name in OUTPUTS:
        assert (out / name).is_file(), name
    assert set(frames) == {"cluster_agreement", "dendrogram_agreement", "cluster_sizes",
                           "k_selection", "density_agreement", "isolated_images"}
    # 2 arms x 2 linkages x 3 k, with the arm kept as its own group rather than pooled away.
    assert len(frames["cluster_agreement"]) == 2 * 2 * 3
    assert set(frames["cluster_agreement"]["allocation_mode"]) == {0.0, 1.0}


def test_k_selection_keeps_the_arms_apart(tmp_path):
    """Pooling the arms would hide exactly the difference the sweep exists to measure."""
    frames = rca.run(_store(tmp_path), tmp_path / "out", ks=(2, 3, 5),
                     linkages=("average",), min_cluster_sizes=(3,), verbose=False)
    sel = frames["k_selection"]
    assert set(sel["allocation_mode"]) == {0.0, 1.0}
    assert len(sel) == 2
    for col in ("k_star_vi", "k_star_sil", "is_flat", "is_arbitrary_slicing",
                "sil_cross_at_k_star_vi"):
        assert col in sel.columns


def test_planted_blobs_are_not_flagged_as_a_continuum(tmp_path):
    """The positive control: separated blobs give high agreement, high silhouette, no verdict."""
    frames = rca.run(_store(tmp_path, sd=0.2), tmp_path / "out", ks=(2, 3, 5, 8),
                     linkages=("average",), min_cluster_sizes=(3,), verbose=False)
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
                     linkages=("average",), min_cluster_sizes=(3,), verbose=False)
    agreement = frames["cluster_agreement"]
    one_arm = agreement[agreement["allocation_mode"] == 0.0].set_index("k")
    assert one_arm.loc[2, "mean_vi_norm"] == pytest.approx(0.0, abs=1e-9)
    assert one_arm.loc[3, "mean_vi_norm"] == pytest.approx(0.0, abs=1e-9)
    assert one_arm.loc[3, "mean_sil_cross"] > one_arm.loc[2, "mean_sil_cross"]

    sel = frames["k_selection"]
    assert (sel["k_star_vi"] == 2).all()       # conservative: the coarsest cut that reproduces
    assert (sel["k_star_sil"] == 3).all()      # the true planted structure


def test_both_granularities_are_scored_at_both_k(tmp_path):
    """Each k* carries all three metrics, so the trade is readable in either direction.

    On planted blobs the finer `k_star_sil` costs *nothing* in reproducibility (VI is 0 at both)
    while nearly doubling the separation. With only the VI-side columns you could see that the two
    k* differ but not whether taking the finer one was free or expensive.
    """
    frames = rca.run(_store(tmp_path, sd=0.2), tmp_path / "out", ks=(2, 3, 5, 8),
                     linkages=("average",), min_cluster_sizes=(3,), verbose=False)
    sel = frames["k_selection"]
    for metric in rca.AT_K_METRICS:
        for suffix in ("vi", "sil"):
            col = f"{metric}_at_k_star_{suffix}"
            assert col in sel.columns, col
            assert sel[col].notna().all(), col

    np.testing.assert_allclose(sel["vi_norm_at_k_star_sil"].to_numpy(),
                               sel["vi_norm_at_k_star_vi"].to_numpy(), atol=1e-12)
    assert (sel["sil_cross_at_k_star_sil"] > sel["sil_cross_at_k_star_vi"]).all()


def test_metrics_at_k_reads_the_agreement_curve_at_the_chosen_k(tmp_path):
    """The attached values must be the curve's own, not a re-derivation that could drift."""
    import pandas as pd
    agreement = pd.DataFrame({
        "num_subjects": [50] * 3, "linkage": ["average"] * 3, "k": [2, 3, 5],
        "mean_vi_norm": [0.1, 0.2, 0.3], "mean_sil_cross": [0.9, 0.5, 0.4],
        "mean_sil_ratio": [1.0, 0.8, 0.6],
    })
    by = ["num_subjects", "linkage"]
    frame = pd.DataFrame([{"num_subjects": 50, "linkage": "average", "k_star_sil": 5}])
    out = rca._metrics_at_k(frame, agreement, by, "sil")
    assert out.loc[0, "vi_norm_at_k_star_sil"] == 0.3
    assert out.loc[0, "sil_cross_at_k_star_sil"] == 0.4
    assert out.loc[0, "sil_ratio_at_k_star_sil"] == 0.6


def test_metrics_at_k_keeps_a_group_whose_k_has_no_agreement_row(tmp_path):
    """A left merge, so an unmatched group gets NaN rather than vanishing from the table."""
    import pandas as pd
    agreement = pd.DataFrame({"num_subjects": [50], "linkage": ["average"], "k": [2],
                              "mean_vi_norm": [0.1], "mean_sil_cross": [0.9],
                              "mean_sil_ratio": [1.0]})
    frame = pd.DataFrame([{"num_subjects": 50, "linkage": "average", "k_star_vi": 99}])
    out = rca._metrics_at_k(frame, agreement, ["num_subjects", "linkage"], "vi")
    assert len(out) == 1
    assert np.isnan(out.loc[0, "vi_norm_at_k_star_vi"])


def test_runs_on_a_conf_only_store(tmp_path):
    """The download command excludes confdists.f32 on purpose; the driver must not need it.

    Every distance here is recomputed as ``pdist(conf)``, which is what the stored row held anyway,
    so dropping the large file changes the results by nothing at all.
    """
    path = _store(tmp_path)
    full = rca.run(path, tmp_path / "out_full", ks=(2, 3), linkages=("average",), min_cluster_sizes=(3,), verbose=False)

    (path / "confdists.f32").unlink()
    reopened = ResultStore.open(path)
    assert not reopened.has_confdists
    partial = rca.run(path, tmp_path / "out_partial", ks=(2, 3), linkages=("average",),
                      min_cluster_sizes=(3,), verbose=False)

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


def test_the_density_pass_is_reported_and_kept_out_of_the_vi_tables(tmp_path):
    """HDBSCAN runs alongside AC, but its noise-class scores must not leak into the VI frame."""
    frames = rca.run(_store(tmp_path), tmp_path / "out", ks=(2, 3), linkages=("average",),
                     min_cluster_sizes=(3, 5), verbose=False)
    density = frames["density_agreement"]
    # One row per (arm, min_cluster_size): the arm stays a group here too, for the same reason it
    # does in the VI tables - the arms might differ and pooling would hide it.
    assert len(density) == 2 * 2
    assert sorted(density["min_cluster_size"].unique()) == [3, 5]
    assert set(density["allocation_mode"]) == {0.0, 1.0}
    for col in ("mean_frac_noise", "mean_noise_kappa", "mean_ari_shared_clustered"):
        assert col in density.columns, col
    # The VI-bearing frames must carry no density columns: a noise class is not a partition, and
    # mixing the two would let a non-metric score be read as though it composed.
    for frame_name in ("cluster_agreement", "k_selection"):
        cols = set(frames[frame_name].columns)
        assert not {c for c in cols if "noise" in c or "min_cluster_size" in c}, frame_name


def test_isolated_images_table_is_per_image(tmp_path):
    frames = rca.run(_store(tmp_path), tmp_path / "out", ks=(2, 3), linkages=("average",),
                     min_cluster_sizes=(3,), density_mcs=3, verbose=False)
    iso = frames["isolated_images"]
    assert set(iso["image"]) == set(range(N_IMAGES))
    assert (iso["min_cluster_size"] == 3).all()
    assert iso["frac_cohorts_noise"].between(0, 1).all()


# --------------------------------------------------------------------- k-selection grouping

def _agreement_frame(n_configs=3, ks=(2, 3, 5), linkages=("average",)):
    """An agreement frame with a swept parameter BEYOND the intended reporting axes."""
    rows = []
    for softness in range(n_configs):          # the extra swept axis
        for linkage in linkages:
            for k in ks:
                rows.append({"num_subjects": 30, "allocation_mode": 0.0, "ndim": 5,
                             "canvas_softness": float(softness), "linkage": linkage, "k": k,
                             "n_reps": 10, "n_pairs": 22,
                             "mean_vi_norm": 0.1 + 0.01 * k, "sem_vi_norm": 0.001,
                             "mean_sil_cross": 0.2 - 0.01 * k, "sem_sil_cross": 0.001,
                             "mean_sil_ratio": 0.5, "sem_sil_ratio": 0.01})
    return pd.DataFrame(rows)


def test_k_selection_has_one_row_per_configuration():
    """The bug this guards: 144 groups once became 186,624 rows and a 35 MB CSV.

    The agreement frame carries a row per swept parameter combination. Grouping by only the
    reporting axes left many rows per (group, k), so the metric merges multiplied the table.
    """
    frame = _agreement_frame(n_configs=3, ks=(2, 3, 5), linkages=("average", "ward"))
    selection = rca.build_k_selection(frame)
    assert len(selection) == 3 * 2, "one row per (configuration, linkage)"
    key = rca._select_by(frame)
    assert not selection.duplicated(subset=key).any()


def test_select_by_keeps_every_swept_parameter():
    frame = _agreement_frame()
    key = rca._select_by(frame)
    assert "canvas_softness" in key, "a swept axis must not be collapsed silently"
    assert "k" not in key and "n_reps" not in key
    assert not any(c.startswith(("mean_", "sem_")) for c in key)


def test_metrics_at_k_refuses_a_non_unique_lookup():
    """Failing loudly beats multiplying the table by 36 twice over."""
    frame = _agreement_frame(n_configs=3)
    bad_by = ["num_subjects", "allocation_mode", "ndim", "linkage"]   # omits canvas_softness
    selection = pd.DataFrame([{**{c: frame[c].iloc[0] for c in bad_by}, "k_star_vi": 2}])
    with pytest.raises(ValueError, match="duplicate rows"):
        rca._metrics_at_k(selection, frame, bad_by, "vi")


def test_metrics_at_k_reads_the_curve_at_the_chosen_k():
    frame = _agreement_frame(n_configs=1, ks=(2, 3, 5))
    selection = rca.build_k_selection(frame)
    row = selection.iloc[0]
    expected = frame[(frame["k"] == row["k_star_vi"]) & (frame["linkage"] == row["linkage"])]
    assert row["vi_norm_at_k_star_vi"] == pytest.approx(expected["mean_vi_norm"].iloc[0])
