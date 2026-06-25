"""Tests for eval_helpers.py - pure pandas/plotly, no R, no simulation."""
import numpy as np
import pandas as pd
import pytest

from SpAM_Simulations import eval_helpers as eh


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
    """Build (coverage, stability, embedding_stability, mds_meta) DataFrames for the given
    list of lever-combo dicts. Cycles through all 4 MDS statuses so every fixture exercises
    the 'error' status path (the one real local run has none)."""
    coverage_rows, stability_rows, embstab_rows, mds_rows = [], [], [], []
    counter = 0
    for combo in combos:
        for rep in range(2):
            coverage_rows.append({**combo, "rep": rep, "num_images": 50,
                                   "average_img_obs": 3.0, "img_coverage": 90.0,
                                   "num_pairs": 1225, "average_pair_obs": 1.5,
                                   "pair_coverage": 70.0, "num_connected_components": 1})
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
def tiny_run_task_v0_1(tmp_path):
    combos = [
        dict(num_subjects=n, trials_per_subject=5, images_per_trial=8,
             subjects_noise_scale=s, subjects_noise_df=1)
        for n in (10, 20) for s in (0.0, 0.5)
    ]
    return _write_run(tmp_path, *_build_frames(combos))


@pytest.fixture
def tiny_run_task_v2_3(tmp_path):
    combos = [
        dict(num_subjects=n, trials_per_subject=5, images_per_trial=8,
             subjects_noise_scale=0.5, subjects_noise_df=1, frac_images_repeated=f)
        for n in (10, 20) for f in (0.0, 0.2)
    ]
    coverage, stability, embstab, mds_meta = _build_frames(combos)
    coverage["mean_snr"] = 1.2
    coverage["median_snr"] = 1.1
    coverage["frac_nan_snr"] = 0.1
    return _write_run(tmp_path, coverage, stability, embstab, mds_meta)


@pytest.fixture
def tiny_run_task_v2_4(tmp_path):
    # Deployment-style v2.4 grid: frac_images_repeated fixed at 0.0 (so SNR is undefined),
    # frac_trials_repeated swept (so test-retest is defined only for its > 0 slice).
    combos = [
        dict(num_subjects=n, trials_per_subject=8, images_per_trial=7,
             subjects_noise_scale=s, subjects_noise_df=1,
             frac_images_repeated=0.0, frac_trials_repeated=fr)
        for n in (10, 20) for s in (0.3, 0.6) for fr in (0.0, 0.25)
    ]
    coverage, stability, embstab, mds_meta = _build_frames(combos)
    coverage["mean_snr"] = np.nan
    coverage["median_snr"] = np.nan
    coverage["frac_nan_snr"] = 1.0
    has_repeat = coverage["frac_trials_repeated"] > 0
    coverage["mean_test_retest"] = np.where(has_repeat, 0.85, np.nan)
    coverage["median_test_retest"] = coverage["mean_test_retest"]
    coverage["frac_nan_test_retest"] = np.where(has_repeat, 0.0, 1.0)
    return _write_run(tmp_path, coverage, stability, embstab, mds_meta)


# --------------------------------------------------------------------- load_run
def test_load_run_resolves_absolute_path_and_detects_task_v0_1(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    assert run.task_version == 0.1
    assert "frac_images_repeated" not in run.levers
    assert run.levers["num_subjects"] == [10, 20]
    assert len(run.coverage) == 8  # 2 num_subjects * 2 noise_scale * 2 reps


def test_load_run_detects_task_v2_3(tiny_run_task_v2_3):
    run = eh.load_run(tiny_run_task_v2_3)
    assert run.task_version == 2.3
    assert run.levers["frac_images_repeated"] == [0.0, 0.2]


def test_load_run_detects_task_v2_4(tiny_run_task_v2_4):
    run = eh.load_run(tiny_run_task_v2_4)
    assert run.task_version == 2.4
    assert run.levers["frac_trials_repeated"] == [0.0, 0.25]
    assert run.levers["frac_images_repeated"] == [0.0]


def test_lever_columns_include_both_repetition_levers():
    # LEVER_COLUMNS is derived from the widest (task-v2.4) NamedTuple, so both levers are known.
    assert "frac_images_repeated" in eh.LEVER_COLUMNS
    assert "frac_trials_repeated" in eh.LEVER_COLUMNS


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
    assert "stability.csv" in msg
    assert "embedding_stability.csv" in msg
    assert "meta.csv" in msg


# --------------------------------------------------------------------- format_value
def test_format_value_rounds_floats_to_4_sig_figs():
    assert eh.format_value(1 / 7) == "0.1429"
    assert eh.format_value(1 / 3) == "0.3333"
    assert eh.format_value(0.8) == "0.8"


def test_format_value_leaves_non_floats_alone():
    assert eh.format_value(20) == "20"
    assert eh.format_value("success") == "success"


# --------------------------------------------------------------------- split_varying_constant
def test_split_varying_constant_drops_constants_and_absent():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [5, 5, 5], "rep": [0, 1, 0]})
    varying, constant = eh.split_varying_constant(df, ["a", "b", "c"])
    assert varying == ["a"]
    assert constant == {"b": 5}  # "c" silently dropped (absent from df.columns)


def test_constants_caption_format():
    caption = eh.constants_caption({"trials_per_subject": 10, "subjects_noise_df": 1})
    assert caption == "trials_per_subject = 10, subjects_noise_df = 1"


def test_constants_caption_rounds_floats():
    caption = eh.constants_caption({"frac_images_repeated": 1 / 7})
    assert caption == "frac_images_repeated = 0.1429"


# --------------------------------------------------------------------- faceted figure builders
def test_faceted_metric_figure_drops_constant_trace_and_captions(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    summary = (
        run.coverage.groupby(["num_subjects", "trials_per_subject", "subjects_noise_scale"])
        .agg(img_mean=("img_coverage", "mean"), img_sem=("img_coverage", "sem"))
        .reset_index()
    )
    fig = eh.faceted_metric_figure(
        summary, x="num_subjects", metrics=[("img_mean", "img_sem", "Images")],
        col_by="trials_per_subject", trace_by=["subjects_noise_scale", "trials_per_subject"],
        title="t",
    )
    # trials_per_subject is constant (=5) everywhere -> dropped from trace names and captioned;
    # only subjects_noise_scale (2 values) varies -> 2 traces. format_value renders 0.0 as "0".
    trace_names = {t.name for t in fig.data}
    assert trace_names == {"subjects_noise_scale=0", "subjects_noise_scale=0.5"}
    # the caption is a native title.subtitle (rendered directly below the title), not a
    # manually-positioned annotation that can drift/overlap with column titles.
    assert "trials_per_subject = 5" in fig.layout.title.subtitle.text


def test_faceted_lever_figure_trace_count(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    fig = eh.faceted_lever_figure(
        run.stability, x="num_subjects", y="spearman", y_sem="spearman",
        row_by="images_per_trial", col_by="trials_per_subject",
        trace_by=["subjects_noise_scale"],
    )
    assert len(fig.data) == 2  # 2 noise scales; single row/col since both are constant here


def test_faceted_lever_figure_row_titles_on_left_rotated_bottom_to_top(tiny_run_task_v2_3):
    run = eh.load_run(tiny_run_task_v2_3)
    summary = (
        run.stability.groupby(["num_subjects", "frac_images_repeated"])
        .agg(spearman_mean=("spearman", "mean")).reset_index()
    )
    fig = eh.faceted_lever_figure(
        summary, x="num_subjects", y="spearman_mean", y_sem="spearman_mean",
        row_by="frac_images_repeated",
    )
    row_annotations = [a for a in fig.layout.annotations if a.text.startswith("frac_images_repeated")]
    assert len(row_annotations) == 2  # one per distinct frac_images_repeated value
    for ann in row_annotations:
        assert ann.x < 0.5  # left side, not the right side make_subplots' row_titles would use
        assert ann.textangle == -90  # rotated to read bottom-to-top, like a y-axis title


def test_faceted_lever_figure_drops_absent_lever(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    # frac_images_repeated isn't a column on a task-v0.1 run - must not crash, must not appear.
    fig = eh.faceted_lever_figure(
        run.stability, x="num_subjects", y="spearman", y_sem="spearman",
        trace_by=["subjects_noise_scale", "frac_images_repeated"],
    )
    assert all("frac_images_repeated" not in t.name for t in fig.data)


# --------------------------------------------------------------------- available_configs / filter_to_config
def test_available_configs_skips_absent_levers(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    secondary = [l for l in eh.LEVER_COLUMNS if l != "num_subjects"]
    configs = eh.available_configs(run.mds_meta, secondary)
    assert "frac_images_repeated" not in configs.columns
    assert "subjects_noise_scale" in configs.columns


def test_available_configs_includes_present_levers(tiny_run_task_v2_3):
    run = eh.load_run(tiny_run_task_v2_3)
    secondary = [l for l in eh.LEVER_COLUMNS if l != "num_subjects"]
    configs = eh.available_configs(run.mds_meta, secondary)
    assert "frac_images_repeated" in configs.columns


def test_filter_to_config_skips_absent_keys(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    filtered = eh.filter_to_config(
        run.mds_meta, {"subjects_noise_scale": 0.5, "frac_images_repeated": 1 / 3}
    )
    assert len(filtered) > 0
    assert (filtered["subjects_noise_scale"] == 0.5).all()


def test_filter_to_config_matches_float_after_csv_roundtrip(tmp_path):
    """Regression test: `pd.to_csv`/`pd.read_csv` does not round-trip float64 exactly (confirmed
    - 1/7 written then read back differs in its last bit), so a hand-typed FOCUS_CONFIG literal
    like `1 / 7` must still match via `np.isclose`, not silently miss via exact `==`."""
    combos = [dict(num_subjects=10, trials_per_subject=5, images_per_trial=8,
                    subjects_noise_scale=0.5, subjects_noise_df=1, frac_images_repeated=1 / 7)]
    run_dir = _write_run(tmp_path, *_build_frames(combos))
    run = eh.load_run(run_dir)
    filtered = eh.filter_to_config(run.mds_meta, {"frac_images_repeated": 1 / 7})
    assert len(filtered) > 0


# --------------------------------------------------------------------- convergence_bar_figure
def test_convergence_bar_figure_includes_error_status(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    fig = eh.convergence_bar_figure(run.mds_meta)
    trace_names = {t.name for t in fig.data}
    assert trace_names == set(eh.DEFAULT_STATUS_LABELS.values())
    colors = {t.name: t.marker.color for t in fig.data}
    assert colors["error"] == eh.DEFAULT_STATUS_COLORS["error"]


def test_convergence_bar_figure_x_is_ndim_facet_is_num_subjects(tiny_run_task_v0_1):
    """Matches evaluation_v0_1.ipynb's original convergence plot: x=ndim (categorical), one
    subplot per num_subjects value."""
    run = eh.load_run(tiny_run_task_v0_1)
    fig = eh.convergence_bar_figure(run.mds_meta)
    assert fig.layout.xaxis.type == "category"
    panel_titles = {a.text for a in fig.layout.annotations if a.text.startswith("num_subjects")}
    assert panel_titles == {"num_subjects = 10", "num_subjects = 20"}
    assert set(fig.data[0].x) == {2, 3}  # x values are ndim, not num_subjects


def test_convergence_bar_figure_wraps_panels_at_explicit_max_cols():
    """5 num_subjects values with an explicit max_cols=2 override -> 3 rows x 2 cols, with the
    trailing (6th) grid cell left empty rather than crowding all 5 panels into one row."""
    combos = [
        dict(num_subjects=n, trials_per_subject=5, images_per_trial=8,
             subjects_noise_scale=0.5, subjects_noise_df=1)
        for n in (10, 20, 30, 40, 50)
    ]
    _, _, _, mds_meta = _build_frames(combos)
    fig = eh.convergence_bar_figure(mds_meta, max_cols=2)
    panel_titles = sorted(a.text for a in fig.layout.annotations if a.text.startswith("num_subjects"))
    assert panel_titles == [f"num_subjects = {n}" for n in (10, 20, 30, 40, 50)]
    # exactly 5 subplot axes allocated (not 6) - the trailing cell genuinely has no subplot.
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 5


def test_convergence_bar_figure_default_grid_is_2x2_for_4_panels():
    """4 num_subjects values, no explicit max_cols -> a clean 2x2 grid, not a lopsided 3+1."""
    combos = [
        dict(num_subjects=n, trials_per_subject=5, images_per_trial=8,
             subjects_noise_scale=0.5, subjects_noise_df=1)
        for n in (10, 20, 30, 40)
    ]
    _, _, _, mds_meta = _build_frames(combos)
    fig = eh.convergence_bar_figure(mds_meta)
    assert len([k for k in fig.layout if k.startswith("xaxis")]) == 4
    assert fig.get_subplot(row=2, col=2) is not None  # a true 2x2, not 4x1 or 1x4


# --------------------------------------------------------------------- _grid_dims
@pytest.mark.parametrize("n, expected", [
    (1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2)),  # <=4 panels -> at most 2 columns
    (5, (2, 3)), (9, (3, 3)), (12, (4, 3)),               # <=12 panels -> at most 3 columns
    (13, (4, 4)), (16, (4, 4)),                           # >12 panels -> at most 4 columns
])
def test_grid_dims_default_tiers(n, expected):
    assert eh._grid_dims(n) == expected


def test_grid_dims_respects_explicit_override():
    assert eh._grid_dims(5, max_cols=2) == (3, 2)


# --------------------------------------------------------------------- pre_post_mds_stability_figure
def test_pre_post_mds_stability_figure_traces(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    fig = eh.pre_post_mds_stability_figure(run.embedding_stability, run.stability)
    names = [t.name for t in fig.data]
    assert names[0] == "Pre-MDS"
    assert fig.data[0].line.dash == "dash"
    assert set(names[1:]) == {"ndim=2", "ndim=3"}


# --------------------------------------------------------------------- repeat_lever_slices
def test_repeat_lever_slices_v2_4_splits_by_trial_repeats(tiny_run_task_v2_4):
    run = eh.load_run(tiny_run_task_v2_4)
    slices = list(eh.repeat_lever_slices(run.mds_meta))
    captions = [c for c, _ in slices]
    assert captions == ["frac_trials_repeated=0", "frac_trials_repeated=0.25"]
    for _, sub in slices:  # each slice carries exactly one frac_trials_repeated value
        assert sub["frac_trials_repeated"].nunique() == 1


def test_repeat_lever_slices_v2_3_splits_by_image_repeats(tiny_run_task_v2_3):
    run = eh.load_run(tiny_run_task_v2_3)
    captions = [c for c, _ in eh.repeat_lever_slices(run.mds_meta)]
    assert captions == ["frac_images_repeated=0", "frac_images_repeated=0.2"]


def test_repeat_lever_slices_v0_1_is_a_single_unlabelled_slice(tiny_run_task_v0_1):
    run = eh.load_run(tiny_run_task_v0_1)
    slices = list(eh.repeat_lever_slices(run.mds_meta))
    assert len(slices) == 1
    caption, sub = slices[0]
    assert caption == ""
    assert len(sub) == len(run.mds_meta)


# --------------------------------------------------------------------- test_retest_figure
def test_test_retest_figure_drops_undefined_slice(tiny_run_task_v2_4):
    run = eh.load_run(tiny_run_task_v2_4)
    fig = eh.test_retest_figure(run.coverage)
    # only the defined frac_trials_repeated=0.25 slice yields a trace; the all-NaN fr=0 slice drops
    assert [t.name for t in fig.data] == ["frac_trials_repeated=0.25"]


def test_test_retest_figure_raises_without_column(tiny_run_task_v2_3):
    run = eh.load_run(tiny_run_task_v2_3)
    with pytest.raises(ValueError, match="mean_test_retest"):
        eh.test_retest_figure(run.coverage)
