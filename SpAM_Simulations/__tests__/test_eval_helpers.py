"""Tests for eval_helpers.py - pure pandas/plotly, no R, no simulation.

eval_helpers is **task-v3 only** (breaking change from the old multi-version loader): levers are
``num_subjects, trials_per_subject, images_per_trial, subjects_noise_scale, subjects_noise_df,
frac_trials_repeated, perspective_dispersion`` - no ``frac_images_repeated``, no doubled-image SNR.
"""
import numpy as np
import pandas as pd
import pytest

from SpAM_Simulations.notebooks import eval_helpers as eh


# --------------------------------------------------------------------- fixtures
def _write_run(tmp_path, coverage, stability, embedding_stability, mds_meta):
    run_dir = tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    (run_dir / "mds_store").mkdir(parents=True)
    coverage.to_csv(run_dir / "out" / "coverage.csv", index=False)
    stability.to_csv(run_dir / "out" / "stability.csv", index=False)
    embedding_stability.to_csv(run_dir / "out" / "embedding_stability.csv", index=False)
    mds_meta.to_csv(run_dir / "mds_store" / "meta.csv", index=False)
    return run_dir


_STATUSES = ["success", "max_iters", "disconnected", "error"]


def _build_frames(combos):
    """Build (coverage, stability, embedding_stability, mds_meta) DataFrames for the given list of
    lever-combo dicts. Cycles through all 4 MDS statuses so every fixture exercises the 'error'
    path. Coverage rows carry the test-retest columns (NaN where frac_trials_repeated == 0)."""
    coverage_rows, stability_rows, embstab_rows, mds_rows = [], [], [], []
    counter = 0
    for combo in combos:
        fr = combo.get("frac_trials_repeated", 0.0)
        retest = 0.85 if fr > 0 else np.nan
        for rep in range(2):
            coverage_rows.append({**combo, "rep": rep, "num_images": 50,
                                   "average_img_obs": 3.0, "img_coverage": 90.0,
                                   "num_pairs": 1225, "average_pair_obs": 1.5,
                                   "pair_coverage": 70.0, "num_connected_components": 1,
                                   "mean_test_retest": retest, "median_test_retest": retest,
                                   "frac_nan_test_retest": (0.0 if fr > 0 else 1.0)})
        stability_rows.append({**combo, "rep_i": 0, "rep_j": 1, "spearman": 0.8})
        for ndim in (2, 3):
            embstab_rows.append({**combo, "ndim": ndim, "n_reps": 2,
                                  "mean_spearman": 0.7 + 0.01 * ndim, "sem_spearman": 0.02})
            for rep in range(2):
                status = _STATUSES[counter % len(_STATUSES)]
                counter += 1
                mds_rows.append({**combo, "rep": rep, "ndim": ndim, "niter": 10.0,
                                  "stress": 0.1, "status": status, "confdist_row": -1})
    return (pd.DataFrame(coverage_rows), pd.DataFrame(stability_rows),
            pd.DataFrame(embstab_rows), pd.DataFrame(mds_rows))


@pytest.fixture
def tiny_run(tmp_path):
    """A minimal v3 run: num_subjects x noise vary, repetition/perspective levers held constant."""
    combos = [
        dict(num_subjects=n, trials_per_subject=5, images_per_trial=8,
             subjects_noise_scale=s, subjects_noise_df=1,
             frac_trials_repeated=0.0, perspective_dispersion=0.0)
        for n in (10, 20) for s in (0.0, 0.5)
    ]
    return _write_run(tmp_path, *_build_frames(combos))


@pytest.fixture
def tiny_run_swept(tmp_path):
    """A v3 run sweeping both new/kept levers: frac_trials_repeated and perspective_dispersion."""
    combos = [
        dict(num_subjects=n, trials_per_subject=8, images_per_trial=7,
             subjects_noise_scale=0.3, subjects_noise_df=1,
             frac_trials_repeated=fr, perspective_dispersion=pd_)
        for n in (10, 20) for fr in (0.0, 0.25) for pd_ in (0.0, 0.3)
    ]
    return _write_run(tmp_path, *_build_frames(combos))


# --------------------------------------------------------------------- load_run
def test_load_run_resolves_path_and_fixes_version_at_v3(tiny_run):
    run = eh.load_run(tiny_run)
    assert run.task_version == 3.0
    assert "frac_images_repeated" not in run.levers   # dropped in v3
    assert run.levers["num_subjects"] == [10, 20]
    assert len(run.coverage) == 8  # 2 num_subjects * 2 noise_scale * 2 reps


def test_load_run_exposes_v3_levers(tiny_run_swept):
    run = eh.load_run(tiny_run_swept)
    assert run.levers["frac_trials_repeated"] == [0.0, 0.25]
    assert run.levers["perspective_dispersion"] == [0.0, 0.3]


def test_lever_columns_are_v3_set():
    assert "perspective_dispersion" in eh.LEVER_COLUMNS
    assert "frac_trials_repeated" in eh.LEVER_COLUMNS
    assert "frac_images_repeated" not in eh.LEVER_COLUMNS  # dropped in v3


def test_load_run_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="run directory not found"):
        eh.load_run(tmp_path / "does_not_exist")


def test_load_run_missing_files_raises_naming_each(tmp_path):
    run_dir = tmp_path / "partial_run"
    (run_dir / "out").mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_csv(run_dir / "out" / "coverage.csv", index=False)
    with pytest.raises(FileNotFoundError) as exc_info:
        eh.load_run(run_dir)
    msg = str(exc_info.value)
    assert "stability.csv" in msg and "embedding_stability.csv" in msg and "meta.csv" in msg


# --------------------------------------------------------------------- plateau_num_subjects
def test_plateau_num_subjects_finds_knee_per_ndim():
    # ndim=2 plateaus at N=30 (0.90 within tol of the 0.91 asymptote); ndim=3 only at N=50.
    rows = []
    for n, s2, s3 in [(10, 0.70, 0.60), (30, 0.90, 0.75), (50, 0.91, 0.90)]:
        rows.append(dict(ndim=2, num_subjects=n, mean_spearman=s2, sem_spearman=0.01))
        rows.append(dict(ndim=3, num_subjects=n, mean_spearman=s3, sem_spearman=0.01))
    out = eh.plateau_num_subjects(pd.DataFrame(rows), tol=0.02).set_index("ndim")
    assert out.loc[2, "plateau_num_subjects"] == 30
    assert out.loc[3, "plateau_num_subjects"] == 50
    assert out.loc[3, "max_num_subjects"] == 50  # asymptote read from the largest N


def test_plateau_num_subjects_flags_unsaturated_sweep():
    # monotonically rising to the very end -> plateau N == max N (sweep not yet saturated)
    rows = [dict(ndim=2, num_subjects=n, mean_spearman=v, sem_spearman=0.01)
            for n, v in [(10, 0.5), (20, 0.7), (30, 0.9)]]
    out = eh.plateau_num_subjects(pd.DataFrame(rows), tol=0.001)
    assert out.loc[0, "plateau_num_subjects"] == out.loc[0, "max_num_subjects"] == 30


# --------------------------------------------------------------------- format_value
def test_format_value_rounds_floats_to_4_sig_figs():
    assert eh.format_value(1 / 7) == "0.1429"
    assert eh.format_value(0.8) == "0.8"


def test_format_value_leaves_non_floats_alone():
    assert eh.format_value(20) == "20" and eh.format_value("success") == "success"


# --------------------------------------------------------------------- split_varying_constant
def test_split_varying_constant_drops_constants_and_absent():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [5, 5, 5], "rep": [0, 1, 0]})
    varying, constant = eh.split_varying_constant(df, ["a", "b", "c"])
    assert varying == ["a"] and constant == {"b": 5}  # "c" dropped (absent)


def test_constants_caption_format():
    caption = eh.constants_caption({"trials_per_subject": 10, "subjects_noise_df": 1})
    assert caption == "trials_per_subject = 10, subjects_noise_df = 1"


# --------------------------------------------------------------------- faceted figure builders
def test_faceted_metric_figure_drops_constant_trace_and_captions(tiny_run):
    run = eh.load_run(tiny_run)
    summary = (
        run.coverage.groupby(["num_subjects", "trials_per_subject", "subjects_noise_scale"])
        .agg(img_mean=("img_coverage", "mean"), img_sem=("img_coverage", "sem")).reset_index()
    )
    fig = eh.faceted_metric_figure(
        summary, x="num_subjects", metrics=[("img_mean", "img_sem", "Images")],
        col_by="trials_per_subject", trace_by=["subjects_noise_scale", "trials_per_subject"],
        title="t",
    )
    trace_names = {t.name for t in fig.data}
    assert trace_names == {"subjects_noise_scale=0", "subjects_noise_scale=0.5"}
    assert "trials_per_subject = 5" in fig.layout.title.subtitle.text


def test_faceted_lever_figure_row_titles_on_left_rotated_bottom_to_top(tiny_run_swept):
    run = eh.load_run(tiny_run_swept)
    summary = (
        run.stability.groupby(["num_subjects", "perspective_dispersion"])
        .agg(spearman_mean=("spearman", "mean")).reset_index()
    )
    fig = eh.faceted_lever_figure(
        summary, x="num_subjects", y="spearman_mean", y_sem="spearman_mean",
        row_by="perspective_dispersion",
    )
    row_annotations = [a for a in fig.layout.annotations if a.text.startswith("perspective_dispersion")]
    assert len(row_annotations) == 2  # one per distinct perspective_dispersion value
    for ann in row_annotations:
        assert ann.x < 0.5 and ann.textangle == -90


def test_faceted_lever_figure_drops_absent_lever(tiny_run):
    run = eh.load_run(tiny_run)
    # frac_images_repeated is not a column on a v3 run - must not crash, must not appear.
    fig = eh.faceted_lever_figure(
        run.stability, x="num_subjects", y="spearman", y_sem="spearman",
        trace_by=["subjects_noise_scale", "frac_images_repeated"],
    )
    assert all("frac_images_repeated" not in t.name for t in fig.data)


# --------------------------------------------------------------------- available_configs / filter_to_config
def test_available_configs_skips_absent_levers(tiny_run):
    run = eh.load_run(tiny_run)
    secondary = [l for l in eh.LEVER_COLUMNS if l != "num_subjects"]
    configs = eh.available_configs(run.mds_meta, secondary)
    assert "frac_images_repeated" not in configs.columns
    assert "perspective_dispersion" in configs.columns


def test_available_configs_includes_present_levers(tiny_run_swept):
    run = eh.load_run(tiny_run_swept)
    secondary = [l for l in eh.LEVER_COLUMNS if l != "num_subjects"]
    configs = eh.available_configs(run.mds_meta, secondary)
    assert "frac_trials_repeated" in configs.columns and "perspective_dispersion" in configs.columns


def test_filter_to_config_matches_float_after_csv_roundtrip(tmp_path):
    """`pd.to_csv`/`read_csv` does not round-trip float64 exactly, so a hand-typed literal like
    `1 / 7` must still match via `np.isclose`, not exact `==`."""
    combos = [dict(num_subjects=10, trials_per_subject=5, images_per_trial=8,
                    subjects_noise_scale=0.5, subjects_noise_df=1,
                    frac_trials_repeated=0.0, perspective_dispersion=1 / 7)]
    run = eh.load_run(_write_run(tmp_path, *_build_frames(combos)))
    assert len(eh.filter_to_config(run.mds_meta, {"perspective_dispersion": 1 / 7})) > 0


# --------------------------------------------------------------------- convergence_bar_figure
def test_convergence_bar_figure_includes_error_status(tiny_run):
    run = eh.load_run(tiny_run)
    fig = eh.convergence_bar_figure(run.mds_meta)
    assert {t.name for t in fig.data} == set(eh.DEFAULT_STATUS_LABELS.values())
    colors = {t.name: t.marker.color for t in fig.data}
    assert colors["error"] == eh.DEFAULT_STATUS_COLORS["error"]


def test_convergence_bar_figure_x_is_ndim_facet_is_num_subjects(tiny_run):
    run = eh.load_run(tiny_run)
    fig = eh.convergence_bar_figure(run.mds_meta)
    assert fig.layout.xaxis.type == "category"
    panel_titles = {a.text for a in fig.layout.annotations if a.text.startswith("num_subjects")}
    assert panel_titles == {"num_subjects = 10", "num_subjects = 20"}
    assert set(fig.data[0].x) == {2, 3}


def test_convergence_bar_figure_default_grid_is_2x2_for_4_panels():
    combos = [dict(num_subjects=n, trials_per_subject=5, images_per_trial=8,
                   subjects_noise_scale=0.5, subjects_noise_df=1,
                   frac_trials_repeated=0.0, perspective_dispersion=0.0)
              for n in (10, 20, 30, 40)]
    _, _, _, mds_meta = _build_frames(combos)
    fig = eh.convergence_bar_figure(mds_meta)
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 4
    assert fig.get_subplot(row=2, col=2) is not None  # a true 2x2


# --------------------------------------------------------------------- _grid_dims
@pytest.mark.parametrize("n, expected", [
    (1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2)),
    (5, (2, 3)), (9, (3, 3)), (12, (4, 3)), (13, (4, 4)), (16, (4, 4)),
])
def test_grid_dims_default_tiers(n, expected):
    assert eh._grid_dims(n) == expected


def test_grid_dims_respects_explicit_override():
    assert eh._grid_dims(5, max_cols=2) == (3, 2)


# --------------------------------------------------------------------- pre_post_mds_stability_figure
def test_pre_post_mds_stability_figure_traces(tiny_run):
    run = eh.load_run(tiny_run)
    fig = eh.pre_post_mds_stability_figure(run.embedding_stability, run.stability)
    names = [t.name for t in fig.data]
    assert names[0] == "Pre-MDS" and fig.data[0].line.dash == "dash"
    assert set(names[1:]) == {"ndim=2", "ndim=3"}


# --------------------------------------------------------------------- condition_slices
def test_condition_slices_splits_by_both_condition_levers(tiny_run_swept):
    run = eh.load_run(tiny_run_swept)  # frac_trials_repeated {0, 0.25} x perspective_dispersion {0, 0.3}
    slices = list(eh.condition_slices(run.mds_meta))
    captions = [c for c, _ in slices]
    assert captions == [
        "frac_trials_repeated=0, perspective_dispersion=0",
        "frac_trials_repeated=0, perspective_dispersion=0.3",
        "frac_trials_repeated=0.25, perspective_dispersion=0",
        "frac_trials_repeated=0.25, perspective_dispersion=0.3",
    ]
    for _, sub in slices:  # each slice fixes both levers -> exactly one point per (num_subjects, ndim)
        assert sub["frac_trials_repeated"].nunique() == 1
        assert sub["perspective_dispersion"].nunique() == 1


def test_condition_slices_single_unlabelled_when_constant(tiny_run):
    run = eh.load_run(tiny_run)  # both condition levers constant at 0.0
    slices = list(eh.condition_slices(run.mds_meta))
    assert len(slices) == 1
    caption, sub = slices[0]
    assert caption == "" and len(sub) == len(run.mds_meta)


# --------------------------------------------------------------------- test_retest_figure
def test_test_retest_figure_drops_undefined_slice(tiny_run_swept):
    run = eh.load_run(tiny_run_swept)
    fig = eh.test_retest_figure(run.coverage, x="perspective_dispersion")
    # only the defined frac_trials_repeated=0.25 slice yields a trace; the all-NaN fr=0 slice drops
    assert [t.name for t in fig.data] == ["frac_trials_repeated=0.25"]


def test_test_retest_figure_raises_without_column(tmp_path):
    combos = [dict(num_subjects=10, trials_per_subject=5, images_per_trial=8,
                   subjects_noise_scale=0.5, subjects_noise_df=1,
                   frac_trials_repeated=0.0, perspective_dispersion=0.0)]
    coverage, stability, embstab, mds_meta = _build_frames(combos)
    coverage = coverage.drop(columns=["mean_test_retest"])  # simulate a run lacking the metric
    run = eh.load_run(_write_run(tmp_path, coverage, stability, embstab, mds_meta))
    with pytest.raises(ValueError, match="mean_test_retest"):
        eh.test_retest_figure(run.coverage)


# --------------------------------------------------------------------- optional result frames

class TestOptionalFrames:
    """The newer tables must be reachable, and their absence must not break a v3-era run.

    load_run used to hard-require exactly four files, which is why embedding_generalizability.csv,
    item_generalizability.csv and topk_jaccard.csv were written by the task-v4 sweep and then never
    read by anything.
    """

    _OPTIONAL = ("embedding_generalizability", "item_generalizability", "topk_jaccard",
                 "recovery_vs_gt", "cluster_agreement", "dendrogram_agreement",
                 "cluster_sizes", "k_selection", "design_only")

    def _minimal_run(self, tmp_path):
        combos = [{"num_subjects": 10, "trials_per_subject": 8, "images_per_trial": 10,
                   "subjects_noise_scale": 0.5, "subjects_noise_df": 5,
                   "frac_trials_repeated": 0.25, "perspective_dispersion": 0.2}]
        return _write_run(tmp_path, *_build_frames(combos))

    def test_a_run_without_them_still_loads(self, tmp_path):
        run = eh.load_run(self._minimal_run(tmp_path))
        for name in self._OPTIONAL:
            assert getattr(run, name) is None, name

    def test_present_files_are_loaded(self, tmp_path):
        run_dir = self._minimal_run(tmp_path)
        frame = pd.DataFrame([{"num_subjects": 10, "ndim": 5, "linkage": "average", "k": 3,
                               "mean_vi_norm": 0.1, "sem_vi_norm": 0.01}])
        frame.to_csv(run_dir / "out" / "cluster_agreement.csv", index=False)
        frame.to_csv(run_dir / "out" / "topk_jaccard.csv", index=False)
        run = eh.load_run(run_dir)
        assert run.cluster_agreement is not None and len(run.cluster_agreement) == 1
        assert run.topk_jaccard is not None
        assert run.cluster_sizes is None, "absent ones stay None"

    def test_a_missing_required_file_still_raises(self, tmp_path):
        run_dir = self._minimal_run(tmp_path)
        (run_dir / "out" / "coverage.csv").unlink()
        with pytest.raises(FileNotFoundError, match="coverage.csv"):
            eh.load_run(run_dir)
