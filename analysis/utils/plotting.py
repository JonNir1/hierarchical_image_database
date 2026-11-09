"""
Shared plotly scaffolding for the per-group figure idioms used across
analysis/pilot/figures.py and analysis/behavioral/demographics_and_behavior.py: a
colorblind-safe categorical palette generator, a violin+per-key-dots panel, a
mean+/-SE-progression-over-x panel (thin per-subject lines underneath), and a
horizontal %-status bar chart, and an equivalence-test forest plot. None of this is specific to any one notebook's
grouping (task version, SHINE cohort, aggregation method, ...) -- callers
supply their own keys/frames/colors/labels.

Usage (from repo root):
    from analysis.utils.plotting import (
        role_palette, add_violin_dots_panel, add_progression_panel, fig_group_status_bar,
        fig_equivalence_forest,
    )
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colorblind-safe categorical hues already validated in this project (see
# analysis/pilot/figures.py's former _VERSION_PALETTE comment: "blue/orange/
# aqua/violet is one of the passing 4-hue combinations from the reference
# 8-hue theme"). A 5th+ hue needs the same manual CVD-validation step before
# being added here -- role_palette() raises rather than silently generating one.
_VALIDATED_HUES: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]  # blue/orange/green/purple

# Fixed alpha convention for each derived role, matching the original
# hand-written rgba() literals exactly.
_ROLE_ALPHAS = {"violin": 0.35, "violin_line": 0.9, "subj": 0.2, "dot": 0.9, "dot_err": 0.6}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def role_palette(keys: Sequence, *, hues: Sequence[str] = _VALIDATED_HUES) -> dict:
    """One of `hues` (in order) per key, each expanded into the 6-role dict
    (main/violin/violin_line/subj/dot/dot_err) via the fixed alpha convention
    above. Raises ValueError if there are more keys than hues -- a 5th+
    category needs a manually CVD-validated hue, not an automatically
    generated one."""
    if len(keys) > len(hues):
        raise ValueError(
            f"role_palette: {len(keys)} keys but only {len(hues)} validated hues available "
            f"({list(hues)}); add and manually CVD-validate a new hue before using more categories."
        )
    return {
        key: {"main": hue, **{role: _hex_to_rgba(hue, alpha) for role, alpha in _ROLE_ALPHAS.items()}}
        for key, hue in zip(keys, hues)
    }


def add_violin_dots_panel(
    fig: go.Figure, violins: dict, dots: dict[str, pd.DataFrame], *, row: int, col_idx: int = 1,
    xaxis_title: str = "", colors: dict, labels: dict | None = None, show_violin_legend: bool = True,
) -> None:
    """One horizontal violin + per-key dot overlay, one subplot row per key
    (iteration order of `violins`).

    `violins[key]` is the raw array-like distribution to plot (e.g. all
    trial-level values, or all subject-level values). `dots[key]` is a
    DataFrame with a "value" column (dot x-positions, pre-sorted by the
    caller) and an optional "error" column (shown as error_x) -- this is what
    lets the same function cover both analysis/pilot/figures.py's former
    _add_violin_dots_panel (dots = per-subject mean +/- SD, aggregated from a
    finer-grained trial-level violin) and
    analysis/behavioral/demographics_and_behavior.py's former inline violin+dots code
    (dots = the same already-per-subject values as the violin, no error
    bars).
    """
    labels = labels or {k: str(k) for k in violins}
    keys = list(violins)
    for i, key in enumerate(keys):
        c = colors[key]
        fig.add_trace(go.Violin(
            x=violins[key], y0=i, orientation="h", side="both",
            fillcolor=c["violin"], line_color=c["violin_line"],
            name=labels[key], legendgroup=key, showlegend=show_violin_legend,
            width=0.6, points=False,
        ), row=row, col=col_idx)

        dot_df = dots[key]
        y_dots = i + np.linspace(-0.15, 0.15, len(dot_df))
        has_error = "error" in dot_df.columns
        marker = {"color": c["dot"], "size": 8 if has_error else 7, "symbol": "circle"}
        if not has_error:
            marker["opacity"] = 0.8
        error_x = None
        if has_error:
            error_x = {
                "type": "data", "array": dot_df["error"].tolist(), "visible": True,
                "color": c["dot_err"], "thickness": 1.5, "width": 4,
            }
        fig.add_trace(go.Scatter(
            x=dot_df["value"], y=y_dots.tolist(), mode="markers", marker=marker, error_x=error_x,
            showlegend=False, legendgroup=key,
            name=f"{labels[key]} subject mean +/- SD" if has_error else f"{labels[key]} subjects",
        ), row=row, col=col_idx)

    fig.update_yaxes(
        range=[-0.55, len(keys) - 0.45],
        tickvals=list(range(len(keys))), ticktext=[labels[k] for k in keys],
        row=row, col=col_idx,
    )
    fig.update_xaxes(title_text=xaxis_title, row=row, col=col_idx)


def add_progression_panel(
    fig: go.Figure, frames: dict[str, pd.DataFrame], value_col: str, x_col: str, *, row: int, col_idx: int = 1,
    xaxis_title: str = "", yaxis_title: str = "", colors: dict, labels: dict | None = None,
    show_legend: bool = False,
) -> None:
    """Mean +/- SE line over `x_col` (e.g. trial number, repeat index), with
    thin per-participant lines underneath, one set of traces per key in
    `frames`. Unifies the identically-named _add_progression_panel() that
    used to live independently in both analysis/pilot/figures.py and
    analysis/behavioral/demographics_and_behavior.py.

    `show_legend` is per-call (not inferred from `row`) so a caller building
    a multi-row figure can show the legend once, on whichever call it likes
    (e.g. only its first row), rather than this function guessing "row == 1".
    """
    labels = labels or {k: str(k) for k in frames}
    for key, df in frames.items():
        c = colors[key]

        for _pid, grp in df.groupby("participant_id"):
            grp_sorted = grp.sort_values(x_col)
            fig.add_trace(go.Scatter(
                x=grp_sorted[x_col], y=grp_sorted[value_col],
                mode="lines", line={"color": c["subj"], "width": 1},
                showlegend=False, hoverinfo="skip", legendgroup=key,
            ), row=row, col=col_idx)

        agg = (
            df.groupby(x_col)[value_col]
            .agg(mean="mean", std="std", n="count")
            .assign(se=lambda x: x["std"] / np.sqrt(x["n"]))
            .reset_index()
        )
        x = agg[x_col].tolist()
        mean = agg["mean"].tolist()
        se = agg["se"].tolist()

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=[m + s for m, s in zip(mean, se)] + [m - s for m, s in zip(mean[::-1], se[::-1])],
            fill="toself", fillcolor=c["main"], opacity=0.2,
            line={"color": "rgba(0,0,0,0)"}, showlegend=False, hoverinfo="skip",
            legendgroup=key,
        ), row=row, col=col_idx)
        fig.add_trace(go.Scatter(
            x=x, y=mean, mode="lines+markers",
            line={"color": c["main"], "width": 3}, marker={"size": 7, "color": c["main"]},
            name=labels[key], legendgroup=key, showlegend=show_legend,
        ), row=row, col=col_idx)

    fig.update_xaxes(title_text=xaxis_title, row=row, col=col_idx)
    fig.update_yaxes(title_text=yaxis_title, row=row, col=col_idx)


def _bar_color_for(bar_colors: dict[str, str], panel_key, status) -> str:
    """`bar_colors` may be keyed by status (bar color varies within a panel)
    or by panel key (every bar in a panel shares one color) -- try status
    first, since a status label and a panel key are never expected to
    collide in practice."""
    if status in bar_colors:
        return bar_colors[status]
    return bar_colors[panel_key]


def fig_group_status_bar(
    panels: dict[str, pd.DataFrame], status_col: str, *, status_order: list[str],
    status_labels: dict[str, str], bar_colors: dict[str, str], panel_titles: dict[str, str],
    title: str = "", height: int = 300, width: int | None = None,
) -> go.Figure:
    """Horizontal %-of-rows bar chart of `status_col` categories, one panel
    per key in `panels`. Unifies analysis/pilot/figures.py's former
    fig_completion_status (bar_colors keyed by status) and
    analysis/behavioral/demographics_and_behavior.py's former fig_retention_status
    (bar_colors keyed by panel key, one color per panel)."""
    keys = list(panels)
    n_panels = len(keys)
    width = width or max(700, 260 * n_panels)

    fig = make_subplots(
        rows=1, cols=n_panels, subplot_titles=[panel_titles[k] for k in keys], horizontal_spacing=0.12,
    )
    for col_idx, key in enumerate(keys, start=1):
        df = panels[key]
        counts = df[status_col].value_counts()
        total = counts.sum()
        cats_present = [c for c in status_order if c in counts.index]
        pcts = [100 * counts[c] / total for c in cats_present]
        fig.add_trace(go.Bar(
            x=pcts, y=[status_labels[c] for c in cats_present], orientation="h", width=0.55,
            text=[f"{p:.1f}% (n={counts[c]})" for p, c in zip(pcts, cats_present)],
            textposition="auto", textfont={"size": 10}, insidetextfont={"color": "white", "size": 10},
            cliponaxis=False, marker_color=[_bar_color_for(bar_colors, key, c) for c in cats_present],
            showlegend=False,
        ), row=1, col=col_idx)
        fig.update_xaxes(
            range=[0, 100], tickvals=[0, 25, 50, 75, 100],
            title_text="% of participants", row=1, col=col_idx,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=col_idx)

    fig.update_layout(
        template="plotly_white", title=title, height=height, width=width,
        margin={"l": 160, "r": 60, "t": 70, "b": 50},
    )
    return fig


_EQUIVALENT_COLOR = "#2a78d6"
_NOT_EQUIVALENT_COLOR = "#eb6834"


def fig_equivalence_forest(results: dict[str, dict], *, x_title: str, title: str) -> go.Figure:
    """Forest plot of equivalence-test results (analysis.utils.equivalence's
    equivalence_test_*() dicts), one row per outcome: effect marker with its
    (1 - 2*alpha) CI, and the +/- bound drawn as two tick marks on the same row
    (bounds may differ per row). Rows whose CI lies inside the bound are blue,
    the rest orange."""
    fig = go.Figure()
    names = list(results)
    for name in names:
        result = results[name]
        color = _EQUIVALENT_COLOR if result["equivalent"] else _NOT_EQUIVALENT_COLOR
        fig.add_trace(go.Scatter(
            x=[result["effect"]], y=[name], mode="markers",
            marker={"color": color, "size": 10},
            error_x={
                "type": "data", "symmetric": False, "color": color, "thickness": 2, "width": 6,
                "array": [result["ci_high"] - result["effect"]], "arrayminus": [result["effect"] - result["ci_low"]],
            },
            showlegend=False,
            hovertemplate=f"{name}<br>effect=%{{x:.3f}}<br>CI [{result['ci_low']:.3f}, {result['ci_high']:.3f}]<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[-result["bound"], result["bound"]], y=[name, name], mode="markers",
            marker={"color": "rgba(80,80,80,0.8)", "size": 14, "symbol": "line-ns-open", "line": {"width": 2}},
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_vline(x=0, line={"color": "rgba(80,80,80,0.5)", "width": 1, "dash": "dot"})
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=names)
    fig.update_xaxes(title_text=x_title)
    fig.update_layout(
        template="plotly_white", title=title,
        height=140 + 55 * len(names), margin={"l": 160, "r": 40, "t": 70, "b": 60},
    )
    return fig
