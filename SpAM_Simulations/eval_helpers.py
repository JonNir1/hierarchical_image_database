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

import pandas as pd
import plotly.express.colors as px_colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from SpAM_Simulations.task_v2_3_experiment import TaskV2_3ExperimentParameters

# Derived from the NamedTuple (not hardcoded) so it can't drift from experiment.py/
# task_v2_3_experiment.py: ["num_subjects", "trials_per_subject", "images_per_trial",
# "subjects_noise_scale", "subjects_noise_df", "frac_images_repeated"]
LEVER_COLUMNS = list(TaskV2_3ExperimentParameters._fields)

DEFAULT_STATUS_LABELS = {
    "success": "converged", "max_iters": "max_iters",
    "disconnected": "disconnected", "error": "error",
}
DEFAULT_STATUS_COLORS = {
    "converged": "#2ca02c", "max_iters": "#ff7f0e",
    "disconnected": "#d62728", "error": "#7f7f7f",
}

_PALETTE = px_colors.qualitative.Plotly


@dataclass
class RunData:
    """Everything a completed run wrote, loaded once and reused by every figure cell."""
    run_dir: Path
    coverage: pd.DataFrame
    stability: pd.DataFrame
    embedding_stability: pd.DataFrame
    mds_meta: pd.DataFrame
    levers: Dict[str, list]
    is_task_v2_3: bool


def load_run(run_results_dir: str | Path) -> RunData:
    """Resolve `run_results_dir` under this module's own directory (`SpAM_Simulations/`),
    not the notebook kernel's cwd, and load the four small result files.

    Raises FileNotFoundError naming every missing path explicitly if the run directory or
    any of the four expected files is absent.
    """
    run_dir = Path(__file__).resolve().parent / run_results_dir
    expected = {
        "coverage": run_dir / "out" / "coverage.csv",
        "stability": run_dir / "out" / "stability.csv",
        "embedding_stability": run_dir / "out" / "embedding_stability.csv",
        "mds_meta": run_dir / "mds_store" / "meta.csv",
    }
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    missing = [str(p) for p in expected.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"run directory {run_dir} is missing expected file(s): {missing}")

    frames = {name: pd.read_csv(path) for name, path in expected.items()}
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
        is_task_v2_3="frac_images_repeated" in levers,
    )


def lever_summary_table(levers: Dict[str, list]) -> go.Figure:
    """One row per lever column, its distinct values, and their count."""
    rows = list(levers.items())
    return go.Figure(go.Table(
        header=dict(values=["Lever", "Distinct Values", "Count"], align="left"),
        cells=dict(values=[
            [name for name, _ in rows],
            [", ".join(str(v) for v in vals) for _, vals in rows],
            [len(vals) for _, vals in rows],
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
    return ", ".join(f"{k} = {v}" for k, v in constants.items())


def _trace_name(varying: Sequence[str], combo: tuple) -> str:
    return "<br>".join(f"{col}={val}" for col, val in zip(varying, combo))


def faceted_metric_figure(
        df: pd.DataFrame, x: str, metrics: Sequence[Tuple[str, str, str]], *,
        col_by: Optional[str] = None, trace_by: Sequence[str] = (),
        title: str = "", x_title: str = "",
) -> go.Figure:
    """Rows = `metrics` (mean_col, sem_col, row_label); cols = `col_by`'s distinct values (or
    a single column if None); traces = `trace_by`, after dropping any constant/absent levers
    via `split_varying_constant` (captioned via `constants_caption` as a layout annotation).

    `df` must already carry the mean/sem columns named in `metrics` - aggregation happens in
    the calling notebook cell, not here.
    """
    col_values = sorted(df[col_by].unique()) if col_by else [None]
    col_titles = [f"{col_by} = {v}" for v in col_values] if col_by else None
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
        title=dict(text=title) if title else None,
        height=300 * len(metrics), width=max(500, 480 * len(col_values)),
        template="plotly_white",
    )
    if constants:
        fig.add_annotation(
            text=constants_caption(constants), showarrow=False,
            xref="paper", yref="paper", x=0.5, y=1.0 - (0.03 if title else -0.02),
            yshift=20, font=dict(size=11, color="gray"),
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
    row_titles = [f"{row_by} = {v}" for v in row_values] if row_by else None
    col_titles = [f"{col_by} = {v}" for v in col_values] if col_by else None
    varying, constants = split_varying_constant(df, trace_by)

    fig = make_subplots(
        rows=len(row_values), cols=len(col_values),
        row_titles=row_titles, column_titles=col_titles,
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

    fig.update_layout(
        title=dict(text=title) if title else None,
        height=300 * len(row_values), width=max(500, 480 * len(col_values)),
        template="plotly_white",
    )
    if constants:
        fig.add_annotation(
            text=constants_caption(constants), showarrow=False,
            xref="paper", yref="paper", x=0.5, y=1.0 - (0.03 if title else -0.02),
            yshift=20, font=dict(size=11, color="gray"),
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
    `df` (e.g. a `frac_images_repeated` key on a task-v0.1 run)."""
    mask = pd.Series(True, index=df.index)
    for col, val in config.items():
        if col in df.columns:
            mask &= df[col] == val
    return df[mask]


def convergence_bar_figure(
        mds_meta: pd.DataFrame, *, normalize: bool = True,
        title: str = "Convergence Status by Dimension",
) -> go.Figure:
    """x=ndim, bars stacked by status (success/max_iters/disconnected/error), `num_subjects`
    as small-multiple subplot columns. `mds_meta` is assumed already filtered to one fixed
    secondary-lever configuration (still varies over `num_subjects`/`ndim`/`rep`).
    """
    df = mds_meta.copy()
    df["status_label"] = df["status"].map(DEFAULT_STATUS_LABELS).fillna("error")
    n_subjs = sorted(df["num_subjects"].unique())

    fig = make_subplots(
        rows=1, cols=len(n_subjs),
        column_titles=[f"num_subjects = {n}" for n in n_subjs],
        shared_yaxes=True, x_title="ndim",
    )
    statuses = list(DEFAULT_STATUS_LABELS.values())
    for c, n_subj in enumerate(n_subjs):
        sub = df[df["num_subjects"] == n_subj]
        counts = (
            sub.groupby("ndim")["status_label"].value_counts()
            .unstack(fill_value=0).reindex(columns=statuses, fill_value=0).reset_index()
        )
        for status in statuses:
            fig.add_trace(
                row=1, col=c + 1, trace=go.Bar(
                    x=counts["ndim"], y=counts[status], name=status,
                    marker_color=DEFAULT_STATUS_COLORS[status],
                    legendgroup=status, showlegend=c == 0,
                )
            )
    fig.update_layout(
        title=dict(text=title), barmode="stack",
        barnorm="percent" if normalize else None,
        height=450, width=max(500, 420 * len(n_subjs)),
        template="plotly_white",
    )
    if normalize:
        fig.update_yaxes(title=dict(text="% of reps"), col=1)
    else:
        fig.update_yaxes(title=dict(text="count"), col=1)
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
