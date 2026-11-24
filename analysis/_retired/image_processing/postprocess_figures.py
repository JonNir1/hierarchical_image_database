"""
Figures for postprocess_analysis.ipynb (SHINE category-neutrality check): the pre/post luminance
histogram, the pairwise-correlation histogram, and the per-feature bar grids with a bootstrap 95% CI.

Usage (from repo root):
    from analysis._retired.image_processing.postprocess_figures import (
        bootstrap_mean_ci, fig_luminance_hist, fig_correlation_hist, fig_feature_bars, L1_COLORS, L2_COLORS,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.utils.plotting import role_palette
from analysis.utils.stats import percentile_ci

_TEMPLATE = "plotly_white"
_STAGE_COLORS = role_palette(["pre_shine", "post_shine"])  # blue / orange

_DOMAIN_HUES = role_palette(["animate", "inanimate"])
L1_COLORS: dict[str, str] = {k: v["main"] for k, v in _DOMAIN_HUES.items()}
# L2 groups (animal_face, ..., natural_object) outnumber the 4 validated hues, so each takes its L1 parent's hue.
L2_COLORS: dict[str, str] = {
    g: L1_COLORS["animate" if g.startswith(("animal", "human")) else "inanimate"]
    for g in ("animal_body", "animal_face", "human_body", "human_face", "handmade_object", "natural_object")
}


def bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high): percentile bootstrap CI of the mean, NaNs dropped, seeded."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    low, high = percentile_ci(values[idx].mean(axis=1), level)
    return float(values.mean()), low, high


def fig_luminance_hist(all_feats: pd.DataFrame, *, feature: str = "mean_luminance") -> go.Figure:
    """Pre- vs post-SHINE distribution of *feature*, as % of images per bin."""
    fig = go.Figure()
    for stage, label in (("pre_shine", "Pre-SHINE"), ("post_shine", "Post-SHINE")):
        vals = all_feats.loc[all_feats["stage"] == stage, feature].to_numpy()
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=40, histnorm="percent", name=label, opacity=0.6,
            marker={"color": _STAGE_COLORS[stage]["main"]},
        ))
    fig.update_layout(
        template=_TEMPLATE, barmode="overlay", height=400, width=700,
        title="Pre- vs. post-SHINE luminance distribution",
        xaxis_title="Mean Luminance (HSV value channel)", yaxis_title="% of images",
        margin={"l": 70, "r": 40, "t": 60, "b": 60},
    )
    return fig


def fig_correlation_hist(cors: np.ndarray) -> go.Figure:
    """Histogram of sub-sampled pairwise pixel correlations with the mean marked."""
    fig = go.Figure(go.Histogram(x=cors, nbinsx=40, marker={"color": _STAGE_COLORS["pre_shine"]["main"]}))
    fig.add_vline(
        x=float(np.mean(cors)), line={"dash": "dash", "color": "#222222"},
        annotation_text=f"Mean = {np.mean(cors):.3f}", annotation_position="top right",
    )
    fig.update_layout(
        template=_TEMPLATE, height=400, width=600, showlegend=False,
        title="Pairwise pixel correlation (post-SHINE)",
        xaxis_title="Mean RGB Pearson correlation", yaxis_title="Number of pairs",
        margin={"l": 70, "r": 40, "t": 60, "b": 60},
    )
    return fig


def fig_feature_bars(
    df: pd.DataFrame, group_col: str, order: list[str], features: list[str], labels: dict[str, str], *,
    colors: dict[str, str], ylabel: str, title: str, rows: int, cols: int,
    x_notes: dict[str, str] | None = None, brackets: dict[str, list[tuple[int, int, float]]] | None = None,
    zero_line: bool = False, tickangle: int = 30, width: int = 1000, height: int = 700,
) -> go.Figure:
    """One panel per feature: group means with bootstrap 95% CI bars.

    *x_notes[feature]* is the panel's x-axis title (the omnibus or two-group p-value text);
    *brackets[feature]* lists significant pairs as (i, j, p) over positions in *order*, drawn above the bars.
    """
    x_notes, brackets = x_notes or {}, brackets or {}
    specs = [[{} for _ in range(cols)] for _ in range(rows)]
    for k in range(len(features), rows * cols):
        specs[k // cols][k % cols] = None
    fig = make_subplots(
        rows=rows, cols=cols, specs=specs, subplot_titles=[f"<b>{labels[f]}</b>" for f in features],
        horizontal_spacing=0.08, vertical_spacing=0.2,
    )
    for k, feat in enumerate(features):
        row, col = k // cols + 1, k % cols + 1
        stats_ = [bootstrap_mean_ci(df.loc[df[group_col] == g, feat].dropna().to_numpy()) for g in order]
        means, lows, highs = (np.array(x) for x in zip(*stats_))
        fig.add_trace(go.Bar(
            x=order, y=means, marker={"color": [colors[g] for g in order]}, showlegend=False,
            error_y={"type": "data", "symmetric": False, "array": highs - means, "arrayminus": means - lows},
        ), row=row, col=col)
        if zero_line:
            fig.add_hline(y=0, line={"dash": "dash", "width": 0.8, "color": "black"}, row=row, col=col)

        y_lo, y_hi = min(0.0, float(lows.min())), float(highs.max())
        span = (y_hi - y_lo) or 1.0
        step, leg = span * 0.08, span * 0.025
        pairs = brackets.get(feat, [])
        for n, (i, j, p) in enumerate(pairs):
            top = y_hi + step * (n + 1)
            fig.add_trace(go.Scatter(
                x=[order[i], order[i], order[j], order[j]], y=[top - leg, top, top, top - leg], mode="lines",
                line={"color": "black", "width": 1.2}, showlegend=False, hoverinfo="skip",
            ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[order[i]], y=[top + leg * 0.5], mode="text", text=["***" if p < 0.001 else "**" if p < 0.01 else "*"],
                textposition="top center", showlegend=False, hoverinfo="skip",
            ), row=row, col=col)
        fig.update_yaxes(
            title_text=ylabel, range=[y_lo - 0.05 * span, y_hi + step * (len(pairs) + 1)], row=row, col=col,
        )
        fig.update_xaxes(title_text=x_notes.get(feat, ""), title_font={"size": 11}, tickangle=tickangle, row=row, col=col)
    fig.update_layout(template=_TEMPLATE, title=title, width=width, height=height, margin={"l": 70, "r": 30, "t": 90, "b": 60})
    return fig
