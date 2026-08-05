"""Shared loading/plotting helpers for `evaluate_simulation.ipynb`.

Plays the same role for the read-only evaluation notebook that `pipeline.py` plays for
generation: a flat module scoped to "what the notebook needs". Unlike `pipeline.py`, nothing
here computes a simulation, runs MDS, or touches R/rpy2 - it only reads the four small files a
completed run already wrote (`out/coverage.csv`, `out/stability.csv`,
`out/embedding_stability.csv`, `mds_store/meta.csv`) and turns already-aggregated columns into
Plotly figures. `mds_store/confdists.f32` and `store_info.json` are never read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# plotly is only needed by the figure builders below; the pure-pandas helpers (e.g.
# plateau_num_subjects) are imported headless by the EC2 calibration prelude, whose minimal venv
# has no plotly. Keep it optional so importing this module never requires plotly - the plotting
# functions still raise a clear AttributeError on `go`/`make_subplots`/`px_colors` if called without it.
try:
    import plotly.express.colors as px_colors
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PALETTE = px_colors.qualitative.Plotly
except ModuleNotFoundError:
    px_colors = go = make_subplots = None
    _PALETTE = None

from SpAM_Simulations.task_v3_experiment import TaskV3ExperimentParameters

# Derived from the task-v3 experiment NamedTuple so it can't drift from task_v3_experiment.py:
# ["num_subjects", "trials_per_subject", "images_per_trial", "subjects_noise_scale",
#  "subjects_noise_df", "frac_trials_repeated", "perspective_dispersion"]. This notebook/module is
# v3-only (see load_run): the legacy frac_images_repeated lever and the doubled-image SNR diagnostic
# are gone; older v0.1/v2.3/v2.4 runs keep their own evaluation_task_v2_*.ipynb notebooks. Levers
# absent from a given run's CSVs are still filtered out wherever this list is used (via
# `if col in df.columns`).
LEVER_COLUMNS = list(TaskV3ExperimentParameters._fields)

DEFAULT_STATUS_LABELS = {
    "success": "converged", "max_iters": "max_iters",
    "disconnected": "disconnected", "error": "error",
}
DEFAULT_STATUS_COLORS = {
    "converged": "#2ca02c", "max_iters": "#ff7f0e",
    "disconnected": "#d62728", "error": "#7f7f7f",
}


def format_value(val: object) -> str:
    """Render a lever value for display: floats are rounded to 4 significant figures
    (e.g. 0.14285714285714285 -> '0.1429') instead of printed at full precision."""
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)


@dataclass
class RunData:
    """Everything a completed run wrote, loaded once and reused by every figure cell.

    The optional frames are ``None`` when the run did not produce them, so a v3-era run still
    loads. Every consumer must therefore check before using one.
    """
    run_dir: Path
    coverage: pd.DataFrame
    stability: pd.DataFrame
    embedding_stability: pd.DataFrame
    mds_meta: pd.DataFrame
    levers: Dict[str, list]
    task_version: float
    embedding_generalizability: Optional[pd.DataFrame] = None
    item_generalizability: Optional[pd.DataFrame] = None
    topk_jaccard: Optional[pd.DataFrame] = None
    recovery_vs_gt: Optional[pd.DataFrame] = None
    cluster_agreement: Optional[pd.DataFrame] = None
    dendrogram_agreement: Optional[pd.DataFrame] = None
    cluster_sizes: Optional[pd.DataFrame] = None
    k_selection: Optional[pd.DataFrame] = None
    design_only: Optional[pd.DataFrame] = None


# Absent -> FileNotFoundError. These four define a loadable run.
_REQUIRED_FILES = {
    "coverage": ("out", "coverage.csv"),
    "stability": ("out", "stability.csv"),
    "embedding_stability": ("out", "embedding_stability.csv"),
    "mds_meta": ("mds_store", "meta.csv"),
}

# Absent -> None. Which of these exist depends on the sweep: the generalizability/top-k/recovery
# tables come from the task-v4 EC2 script, the cluster tables from the local post-processing pass.
_OPTIONAL_FILES = {
    "embedding_generalizability": ("out", "embedding_generalizability.csv"),
    "item_generalizability": ("out", "item_generalizability.csv"),
    "topk_jaccard": ("out", "topk_jaccard.csv"),
    "recovery_vs_gt": ("out", "recovery_vs_gt.csv"),
    "cluster_agreement": ("out", "cluster_agreement.csv"),
    "dendrogram_agreement": ("out", "dendrogram_agreement.csv"),
    "cluster_sizes": ("out", "cluster_sizes.csv"),
    "k_selection": ("out", "k_selection.csv"),
    "design_only": ("out", "design_only.csv"),
}


def load_run(run_results_dir: str | Path) -> RunData:
    """Resolve `run_results_dir` under this module's own directory (`SpAM_Simulations/`),
    not the notebook kernel's cwd, and load a run's small result files.

    Four files are required; the rest load as ``None`` when absent. Splitting them this way is what
    makes the newer tables reachable at all: this loader used to hard-require exactly the four, so
    `embedding_generalizability.csv`, `item_generalizability.csv` and `topk_jaccard.csv` were
    written by the task-v4 sweep and then never read by anything.

    Raises FileNotFoundError naming every missing *required* path explicitly.
    """
    run_dir = Path(__file__).resolve().parent / run_results_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    required = {name: run_dir.joinpath(*parts) for name, parts in _REQUIRED_FILES.items()}
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"run directory {run_dir} is missing expected file(s): {missing}")

    frames = {name: pd.read_csv(path) for name, path in required.items()}
    optional = {
        name: (pd.read_csv(run_dir.joinpath(*parts))
               if run_dir.joinpath(*parts).is_file() else None)
        for name, parts in _OPTIONAL_FILES.items()
    }
    levers = {
        col: sorted(frames["mds_meta"][col].unique())
        for col in LEVER_COLUMNS
        if col in frames["mds_meta"].columns
    }
    return RunData(
        run_dir=run_dir,
        coverage=frames["coverage"],
        stability=frames["stability"],
        embedding_stability=frames["embedding_stability"],
        mds_meta=frames["mds_meta"],
        levers=levers,
        task_version=3.0,  # this module is task-v3-only (breaking change from the multi-version loader)
        **optional,
    )


def plateau_num_subjects(
        embedding_stability_df: pd.DataFrame, *, x: str = "num_subjects",
        y: str = "mean_spearman", group_by: Sequence[str] = ("ndim",), tol: float = 0.01,
        higher_is_better: bool = True,
) -> pd.DataFrame:
    """Smallest ``num_subjects`` whose stability is within ``tol`` of the group's asymptote.

    With per-subject perspective dispersion the stability-vs-N curve saturates *below* 1.0, so the
    old "first N above a fixed Spearman threshold" rule no longer applies - the convergence target
    is instead the curve's own plateau. For each group (default one row per ``ndim``) this takes the
    asymptote as the stability at the largest ``num_subjects`` and reports the smallest ``num_subjects``
    reaching ``asymptote - tol``. Returns one row per group with ``plateau_num_subjects``,
    ``asymptote``, and ``max_num_subjects`` (the N the asymptote was read from - if the plateau N
    equals it, the sweep has not yet saturated and needs larger N).

    Set ``higher_is_better=False`` for a *disparity* measure that improves downward - notably
    ``compute_embedding_generalizability``'s ``mean_procrustes_m2``, which falls toward 0 as N
    grows. The plateau criterion then becomes ``asymptote + tol``. Leaving it True on a disparity
    column would return the smallest N in the sweep every time, since every point sits above an
    asymptote that is the curve's minimum.
    """
    group_by = [c for c in group_by if c in embedding_stability_df.columns]
    rows = []
    grouped = ([((), embedding_stability_df)] if not group_by
               else embedding_stability_df.groupby(list(group_by)))
    for key, grp in grouped:
        grp = grp.dropna(subset=[y]).sort_values(x)
        if grp.empty:
            continue
        asymptote = grp[y].iloc[-1]
        reached = (grp[grp[y] >= asymptote - tol] if higher_is_better
                   else grp[grp[y] <= asymptote + tol])
        plateau_n = reached[x].iloc[0] if not reached.empty else grp[x].iloc[-1]
        key_tuple = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(group_by, key_tuple)),
            "plateau_num_subjects": int(plateau_n),
            "asymptote": float(asymptote),
            "max_num_subjects": int(grp[x].iloc[-1]),
        })
    return pd.DataFrame(rows)


def lever_summary_table(levers: Dict[str, list]) -> go.Figure:
    """One row per lever column and its distinct values."""
    rows = list(levers.items())
    return go.Figure(go.Table(
        header=dict(values=["Lever", "Distinct Values"], align="left"),
        cells=dict(values=[
            [name for name, _ in rows],
            [", ".join(format_value(v) for v in vals) for _, vals in rows],
        ], align="left"),
    ))


def split_varying_constant(
        df: pd.DataFrame, candidate_levers: Sequence[str]
) -> Tuple[List[str], Dict[str, object]]:
    """Partition `candidate_levers` into (varying, constant) based on `df`.

    A lever absent from `df.columns` is dropped from both outputs; a lever with exactly one
    distinct value in `df` is "constant" (returned as {lever: value}); anything else "varies"
    (kept, order preserved). This is the shared mechanism behind the project-wide rule that a
    feature shared across all traces belongs in a caption/annotation, not the trace name.
    """
    varying, constant = [], {}
    for col in candidate_levers:
        if col not in df.columns:
            continue
        values = df[col].unique()
        if len(values) == 1:
            constant[col] = values[0]
        else:
            varying.append(col)
    return varying, constant


def constants_caption(constants: Dict[str, object]) -> str:
    """Render {lever: value} as 'trials_per_subject = 10, subjects_noise_df = 1'."""
    return ", ".join(f"{k} = {format_value(v)}" for k, v in constants.items())


def _title_with_caption(title: str, constants: Dict[str, object]) -> Optional[dict]:
    """Build a `layout.title` dict with `constants_caption(constants)` as a native Plotly
    subtitle - rendered directly below the main title, never overlapping column titles
    (unlike a manually-positioned annotation at a guessed y-coordinate)."""
    if not title and not constants:
        return None
    title_dict = dict(text=title or "")
    if constants:
        title_dict["subtitle"] = dict(text=constants_caption(constants), font=dict(size=11, color="gray"))
    return title_dict


def _trace_name(varying: Sequence[str], combo: tuple) -> str:
    return "<br>".join(f"{col}={format_value(val)}" for col, val in zip(varying, combo))


def faceted_metric_figure(
        df: pd.DataFrame, x: str, metrics: Sequence[Tuple[str, str, str]], *,
        col_by: Optional[str] = None, trace_by: Sequence[str] = (),
        title: str = "", x_title: str = "",
) -> go.Figure:
    """Rows = `metrics` (mean_col, sem_col, row_label); cols = `col_by`'s distinct values (or
    a single column if None); traces = `trace_by`, after dropping any constant/absent levers
    via `split_varying_constant` (captioned via `constants_caption` as the title's subtitle).

    `df` must already carry the mean/sem columns named in `metrics` - aggregation happens in
    the calling notebook cell, not here.
    """
    col_values = sorted(df[col_by].unique()) if col_by else [None]
    col_titles = [f"{col_by} = {format_value(v)}" for v in col_values] if col_by else None
    varying, constants = split_varying_constant(df, trace_by)

    fig = make_subplots(
        rows=len(metrics), cols=len(col_values),
        column_titles=col_titles,
        shared_xaxes=True, shared_yaxes=True,
        x_title=x_title or x,
        vertical_spacing=0.06, horizontal_spacing=0.03,
    )
    combos = [tuple()] if not varying else sorted(set(
        tuple(row) for row in df[list(varying)].drop_duplicates().itertuples(index=False)
    ))
    for c, col_val in enumerate(col_values):
        sub = df if col_val is None else df[df[col_by] == col_val]
        for r, (mean_col, sem_col, row_label) in enumerate(metrics):
            for i, combo in enumerate(combos):
                mask = pd.Series(True, index=sub.index)
                for col, val in zip(varying, combo):
                    mask &= sub[col] == val
                trace_df = sub[mask].sort_values(x)
                name = _trace_name(varying, combo) or row_label
                fig.add_trace(
                    row=r + 1, col=c + 1, trace=go.Scatter(
                        x=trace_df[x], y=trace_df[mean_col],
                        error_y=dict(type="data", array=trace_df[sem_col].fillna(0), visible=True),
                        name=name, legendgroup=name,
                        showlegend=(c == 0 and r == 0 and bool(varying)),
                        mode="lines+markers", line=dict(color=_PALETTE[i % len(_PALETTE)]),
                    )
                )
            if c == 0:
                fig.update_yaxes(row=r + 1, col=c + 1, title=dict(text=row_label))

    fig.update_layout(
        title=_title_with_caption(title, constants),
        height=300 * len(metrics), width=max(500, 480 * len(col_values)),
        template="plotly_white",
    )
    return fig


def faceted_lever_figure(
        df: pd.DataFrame, x: str, y: str, y_sem: str, *,
        row_by: Optional[str] = None, col_by: Optional[str] = None,
        trace_by: Sequence[str] = (), title: str = "", x_title: str = "", y_title: str = "",
) -> go.Figure:
    """Rows = `row_by`'s distinct values, cols = `col_by`'s distinct values, traces =
    `trace_by` (dropping constant/absent levers via `split_varying_constant`/
    `constants_caption`). `df[y]`/`df[y_sem]` must already be aggregated at the
    (row_by, col_by, trace_by, x) granularity - no aggregation performed here.
    """
    row_values = sorted(df[row_by].unique()) if row_by else [None]
    col_values = sorted(df[col_by].unique()) if col_by else [None]
    col_titles = [f"{col_by} = {format_value(v)}" for v in col_values] if col_by else None
    varying, constants = split_varying_constant(df, trace_by)

    fig = make_subplots(
        rows=len(row_values), cols=len(col_values),
        column_titles=col_titles,
        shared_xaxes=True, shared_yaxes=True,
        x_title=x_title or x,
        vertical_spacing=0.06, horizontal_spacing=0.03,
    )
    combos = [tuple()] if not varying else sorted(set(
        tuple(row) for row in df[list(varying)].drop_duplicates().itertuples(index=False)
    ))
    for r, row_val in enumerate(row_values):
        row_df = df if row_val is None else df[df[row_by] == row_val]
        for c, col_val in enumerate(col_values):
            sub = row_df if col_val is None else row_df[row_df[col_by] == col_val]
            for i, combo in enumerate(combos):
                mask = pd.Series(True, index=sub.index)
                for col, val in zip(varying, combo):
                    mask &= sub[col] == val
                trace_df = sub[mask].sort_values(x)
                name = _trace_name(varying, combo) or y
                fig.add_trace(
                    row=r + 1, col=c + 1, trace=go.Scatter(
                        x=trace_df[x], y=trace_df[y],
                        error_y=dict(type="data", array=trace_df[y_sem].fillna(0), visible=True),
                        name=name, legendgroup=name,
                        showlegend=(r == 0 and c == 0 and bool(varying)),
                        mode="lines+markers", line=dict(color=_PALETTE[i % len(_PALETTE)]),
                    )
                )
            if c == 0 and y_title:
                fig.update_yaxes(row=r + 1, col=c + 1, title=dict(text=y_title))

    margin = {}
    if row_by:
        # Left side, rotated to read bottom-to-top (matching the y-axis-title convention),
        # unlike make_subplots' own row_titles (right side, top-to-bottom) - placed manually
        # since make_subplots offers no side/angle control for them.
        for r, row_val in enumerate(row_values):
            y0, y1 = fig.get_subplot(row=r + 1, col=1).yaxis.domain
            fig.add_annotation(
                text=f"{row_by} = {format_value(row_val)}", showarrow=False,
                xref="paper", yref="paper", x=-0.09, y=(y0 + y1) / 2,
                textangle=-90, xanchor="center", yanchor="middle", font=dict(size=12),
            )
        margin["l"] = 110

    fig.update_layout(
        title=_title_with_caption(title, constants),
        height=300 * len(row_values), width=max(500, 480 * len(col_values)),
        template="plotly_white", margin=margin,
    )
    return fig


def available_configs(mds_meta: pd.DataFrame, secondary_levers: Sequence[str]) -> pd.DataFrame:
    """Distinct combinations of `secondary_levers` present in `mds_meta` (i.e. every lever
    except `num_subjects`), for picking a drill-down `FOCUS_CONFIG`. Levers absent from
    `mds_meta.columns` are skipped."""
    cols = [c for c in secondary_levers if c in mds_meta.columns]
    if not cols:
        return pd.DataFrame()
    return mds_meta[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)


def filter_to_config(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Filter `df` to rows matching `config`, skipping any key not present as a column in
    `df` (e.g. a `frac_images_repeated` key on a task-v0.1 run).

    Float values are matched with `np.isclose`, not `==`: a value written to CSV and read
    back can differ from the original in its last bit or two (confirmed - `pd.to_csv` does
    not round-trip `float64` exactly), so a hand-typed literal like `1 / 7` would otherwise
    silently fail to match a config that does exist in `df`.
    """
    mask = pd.Series(True, index=df.index)
    for col, val in config.items():
        if col not in df.columns:
            continue
        mask &= np.isclose(df[col], val) if isinstance(val, float) else df[col] == val
    return df[mask]


def _grid_dims(n: int, max_cols: Optional[int] = None) -> Tuple[int, int]:
    """(nrows, ncols) for laying out `n` small-multiple panels.

    If `max_cols` isn't given, picks a column count by panel count - 2 up to 4 panels (a
    clean 2x2 rather than a lopsided 3+1), 3 up to 12 (3x3 at the common case of 9), 4 beyond
    that - then takes however many rows are needed. A single fixed column count would either
    crowd everything into one ever-widening row or leave an awkward partial row for small n.
    """
    if max_cols is None:
        max_cols = 2 if n <= 4 else 3 if n <= 12 else 4
    ncols = min(n, max_cols)
    nrows = -(-n // ncols)  # ceil division
    return nrows, ncols


def convergence_bar_figure(
        mds_meta: pd.DataFrame, *, normalize: bool = True, max_cols: Optional[int] = None,
        title: str = "Convergence Status by Dimension",
) -> go.Figure:
    """x=ndim (categorical), bars stacked by status (success/max_iters/disconnected/error),
    one subplot per `num_subjects` value (matching `evaluation_v0_1.ipynb`'s original convergence
    plot), laid out via `_grid_dims` rather than one ever-widening row. `mds_meta` is assumed
    already filtered to one fixed secondary-lever configuration (still varies over
    `num_subjects`/`ndim`/`rep`).
    """
    df = mds_meta.copy()
    df["status_label"] = df["status"].map(DEFAULT_STATUS_LABELS).fillna("error")
    n_subjs = sorted(df["num_subjects"].unique())

    n = len(n_subjs)
    nrows, ncols = _grid_dims(n, max_cols)
    specs = [[{} if (r * ncols + c) < n else None for c in range(ncols)] for r in range(nrows)]

    fig = make_subplots(
        rows=nrows, cols=ncols, specs=specs,
        subplot_titles=[f"num_subjects = {n_subj}" for n_subj in n_subjs],
        shared_yaxes=True, x_title="ndim",
        vertical_spacing=0.18 if nrows > 1 else 0.06,
    )
    statuses = list(DEFAULT_STATUS_LABELS.values())
    for i, n_subj in enumerate(n_subjs):
        r, c = divmod(i, ncols)
        sub = df[df["num_subjects"] == n_subj]
        counts = (
            sub.groupby("ndim")["status_label"].value_counts()
            .unstack(fill_value=0).reindex(columns=statuses, fill_value=0).reset_index()
        )
        for status in statuses:
            fig.add_trace(
                row=r + 1, col=c + 1, trace=go.Bar(
                    x=counts["ndim"], y=counts[status], name=status,
                    marker_color=DEFAULT_STATUS_COLORS[status],
                    legendgroup=status, showlegend=(i == 0),
                )
            )
        fig.update_xaxes(row=r + 1, col=c + 1, type="category")
        if c == 0:
            fig.update_yaxes(row=r + 1, col=c + 1, title=dict(text="% of reps" if normalize else "count"))
    fig.update_layout(
        title=dict(text=title), barmode="stack",
        barnorm="percent" if normalize else None,
        height=320 * nrows, width=max(500, 380 * ncols),
        template="plotly_white",
    )
    return fig


def pre_post_mds_stability_figure(
        embedding_stability_df: pd.DataFrame, stability_df: pd.DataFrame, *,
        x: str = "num_subjects", title: str = "Embedding Stability vs. Pre-MDS Baseline",
) -> go.Figure:
    """Single (non-faceted) figure: one black-dashed 'Pre-MDS' trace from
    `stability_df['spearman']` (mean+SEM across rep-pairs, grouped by `x`), plus one colored
    solid trace per `ndim` already present in `embedding_stability_df` (its
    `mean_spearman`/`sem_spearman` used as-is). Both inputs are assumed already filtered to
    one fixed configuration.
    """
    fig = go.Figure()
    raw = (
        stability_df.dropna(subset=["spearman"])
        .groupby(x)["spearman"].agg(mean="mean", sem="sem").reset_index().sort_values(x)
    )
    fig.add_trace(go.Scatter(
        x=raw[x], y=raw["mean"],
        error_y=dict(type="data", array=raw["sem"].fillna(0), visible=True),
        name="Pre-MDS", mode="lines+markers", line=dict(color="black", dash="dash"),
    ))
    for i, ndim in enumerate(sorted(embedding_stability_df["ndim"].unique())):
        sub = embedding_stability_df[embedding_stability_df["ndim"] == ndim].sort_values(x)
        fig.add_trace(go.Scatter(
            x=sub[x], y=sub["mean_spearman"],
            error_y=dict(type="data", array=sub["sem_spearman"].fillna(0), visible=True),
            name=f"ndim={ndim}", mode="lines+markers",
            line=dict(color=_PALETTE[i % len(_PALETTE)]),
        ))
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=dict(text=x)), yaxis=dict(title=dict(text="Spearman R")),
        template="plotly_white", width=700, height=450,
    )
    return fig


# "Condition" levers sliced into separate by-dimension figures (rather than folded into a trace
# or averaged over): each shifts the achievable stability ceiling, so pooling them would both hide
# that effect and - where two rows land in the same (x, trace) cell - plot duplicate markers.
# frac_trials_repeated = whole-trial test-retest repeats; perspective_dispersion = between-subject
# disagreement. Task v3.0 dropped frac_images_repeated. Order controls caption ordering.
CONDITION_SLICE_LEVERS = ["frac_trials_repeated", "perspective_dispersion"]


def condition_slices(df: pd.DataFrame):
    """Yield ``(caption, sub_df)`` for each distinct combination of the *varying* condition levers
    present in `df` (`frac_trials_repeated` / `perspective_dispersion`).

    A run where none of them vary yields a single ``("", df)`` pair, so a caller can loop uniformly
    whether or not they are swept. Slicing (one figure per combination) - rather than averaging -
    keeps each level's ceiling distinct and avoids duplicate markers when a lever varies but isn't
    a facet/trace dimension of the figure.
    """
    varying = [c for c in CONDITION_SLICE_LEVERS if c in df.columns and df[c].nunique() > 1]
    if not varying:
        yield "", df
        return
    combos = df[varying].drop_duplicates().sort_values(varying)
    for row in combos.itertuples(index=False):
        config = dict(zip(varying, row))
        caption = ", ".join(f"{k}={format_value(v)}" for k, v in config.items())
        yield caption, filter_to_config(df, config)


def test_retest_figure(
        coverage_df: pd.DataFrame, *, x: str = "subjects_noise_scale",
        trace_by: str = "frac_trials_repeated",
        title: str = "Test-Retest Reliability vs. Subject Noise",
) -> go.Figure:
    """Single figure: mean per-subject test-retest reliability (`mean_test_retest`, already in
    `out/coverage.csv`) against `x`, one trace per `trace_by` value.

    Aggregates `mean_test_retest` (mean +/- SEM) across reps and every lever other than `x`/
    `trace_by` - an overview view, mirroring the framing of the other overview cells. Rows where
    the reliability is undefined (the all-NaN `frac_trials_repeated == 0` slice) are dropped, so
    they simply don't appear.

    :raises ValueError: if `coverage_df` has no `mean_test_retest` column.
    """
    if "mean_test_retest" not in coverage_df.columns:
        raise ValueError(
            "coverage_df has no 'mean_test_retest' column - test-retest reliability needs a run "
            "with frac_trials_repeated > 0 (set RUN_RESULTS_DIR accordingly)."
        )
    df = coverage_df.dropna(subset=["mean_test_retest"])
    trace_vals = sorted(df[trace_by].unique()) if trace_by in df.columns else [None]
    fig = go.Figure()
    for i, tv in enumerate(trace_vals):
        sub = df if tv is None else df[df[trace_by] == tv]
        agg = (sub.groupby(x)["mean_test_retest"]
               .agg(mean="mean", sem="sem").reset_index().sort_values(x))
        name = "all" if tv is None else f"{trace_by}={format_value(tv)}"
        fig.add_trace(go.Scatter(
            x=agg[x], y=agg["mean"],
            error_y=dict(type="data", array=agg["sem"].fillna(0), visible=True),
            name=name, mode="lines+markers", line=dict(color=_PALETTE[i % len(_PALETTE)]),
        ))
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title=dict(text=x)),
        yaxis=dict(title=dict(text="mean test-retest Spearman r")),
        template="plotly_white", width=700, height=450,
        legend=dict(title=dict(text=trace_by)),
    )
    return fig
