"""Tests for eval_helpers.py - pure pandas/plotly, no R, no simulation."""
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
def tiny_run_uniform(tmp_path):
    combos = [
        dict(num_subjects=n, trials_per_subject=5, images_per_trial=8,
             subjects_noise_scale=s, subjects_noise_df=1)
        for n in (10, 20) for s in (0.0, 0.5)
    ]
    return _write_run(tmp_path, *_build_frames(combos))


@pytest.fixture
def tiny_run_realistic(tmp_path):
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


# --------------------------------------------------------------------- load_run
def test_load_run_resolves_absolute_path_and_detects_uniform(tiny_run_uniform):
    run = eh.load_run(tiny_run_uniform)
    assert run.is_realistic is False
    assert "frac_images_repeated" not in run.levers
    assert run.levers["num_subjects"] == [10, 20]
    assert len(run.coverage) == 8  # 2 num_subjects * 2 noise_scale * 2 reps


def test_load_run_detects_realistic(tiny_run_realistic):
    run = eh.load_run(tiny_run_realistic)
    assert run.is_realistic is True
    assert run.levers["frac_images_repeated"] == [0.0, 0.2]


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


# --------------------------------------------------------------------- split_varying_constant
def test_split_varying_constant_drops_constants_and_absent():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [5, 5, 5], "rep": [0, 1, 0]})
    varying, constant = eh.split_varying_constant(df, ["a", "b", "c"])
    assert varying == ["a"]
    assert constant == {"b": 5}  # "c" silently dropped (absent from df.columns)


def test_constants_caption_format():
    caption = eh.constants_caption({"trials_per_subject": 10, "subjects_noise_df": 1})
    assert caption == "trials_per_subject = 10, subjects_noise_df = 1"


# --------------------------------------------------------------------- faceted figure builders
def test_faceted_metric_figure_drops_constant_trace_and_captions(tiny_run_uniform):
    run = eh.load_run(tiny_run_uniform)
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
    # only subjects_noise_scale (2 values) varies -> 2 traces.
    trace_names = {t.name for t in fig.data}
    assert trace_names == {"subjects_noise_scale=0.0", "subjects_noise_scale=0.5"}
    caption_text = " ".join(a.text for a in fig.layout.annotations if a.text)
    assert "trials_per_subject = 5" in caption_text


def test_faceted_lever_figure_trace_count(tiny_run_uniform):
    run = eh.load_run(tiny_run_uniform)
    fig = eh.faceted_lever_figure(
        run.stability, x="num_subjects", y="spearman", y_sem="spearman",
        row_by="images_per_trial", col_by="trials_per_subject",
        trace_by=["subjects_noise_scale"],
    )
    assert len(fig.data) == 2  # 2 noise scales; single row/col since both are constant here


def test_faceted_lever_figure_drops_absent_lever(tiny_run_uniform):
    run = eh.load_run(tiny_run_uniform)
    # frac_images_repeated isn't a column on a uniform run - must not crash, must not appear.
    fig = eh.faceted_lever_figure(
        run.stability, x="num_subjects", y="spearman", y_sem="spearman",
        trace_by=["subjects_noise_scale", "frac_images_repeated"],
    )
    assert all("frac_images_repeated" not in t.name for t in fig.data)


# --------------------------------------------------------------------- available_configs / filter_to_config
def test_available_configs_skips_absent_levers(tiny_run_uniform):
    run = eh.load_run(tiny_run_uniform)
    secondary = [l for l in eh.LEVER_COLUMNS if l != "num_subjects"]
    configs = eh.available_configs(run.mds_meta, secondary)
    assert "frac_images_repeated" not in configs.columns
    assert "subjects_noise_scale" in configs.columns


def test_available_configs_includes_present_levers(tiny_run_realistic):
    run = eh.load_run(tiny_run_realistic)
    secondary = [l for l in eh.LEVER_COLUMNS if l != "num_subjects"]
    configs = eh.available_configs(run.mds_meta, secondary)
    assert "frac_images_repeated" in configs.columns


def test_filter_to_config_skips_absent_keys(tiny_run_uniform):
    run = eh.load_run(tiny_run_uniform)
    filtered = eh.filter_to_config(
        run.mds_meta, {"subjects_noise_scale": 0.5, "frac_images_repeated": 1 / 3}
    )
    assert len(filtered) > 0
    assert (filtered["subjects_noise_scale"] == 0.5).all()


# --------------------------------------------------------------------- convergence_bar_figure
def test_convergence_bar_figure_includes_error_status(tiny_run_uniform):
    run = eh.load_run(tiny_run_uniform)
    fig = eh.convergence_bar_figure(run.mds_meta)
    trace_names = {t.name for t in fig.data}
    assert trace_names == set(eh.DEFAULT_STATUS_LABELS.values())
    colors = {t.name: t.marker.color for t in fig.data}
    assert colors["error"] == eh.DEFAULT_STATUS_COLORS["error"]


# --------------------------------------------------------------------- pre_post_mds_stability_figure
def test_pre_post_mds_stability_figure_traces(tiny_run_uniform):
    run = eh.load_run(tiny_run_uniform)
    fig = eh.pre_post_mds_stability_figure(run.embedding_stability, run.stability)
    names = [t.name for t in fig.data]
    assert names[0] == "Pre-MDS"
    assert fig.data[0].line.dash == "dash"
    assert set(names[1:]) == {"ndim=2", "ndim=3"}
