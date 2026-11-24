"""
Shared plotly scaffolding for the per-group figure idioms used across
analysis/_retired/pilot/figures.py and analysis/s1_pre_vs_post_shine/demographics_and_behavior.py: a
colorblind-safe categorical palette generator, a violin+per-key-dots panel, a
mean+/-SE-progression-over-x panel (thin per-subject lines underneath), a
split violin (one group per side), a horizontal %-status bar chart, an
equivalence-test forest plot, and the RDM design language (an RDM's name fixes
its hue, marker symbol, shade and line dash; see the section at the end). The
group helpers are not specific to any one notebook's grouping (task version,
SHINE cohort, aggregation method, ...) -- callers supply their own
keys/frames/colors/labels.

Usage (from repo root):
    from analysis.utils.plotting import (
        role_palette, add_violin_dots_panel, add_progression_panel, add_split_violin,
        fig_group_status_bar, fig_equivalence_forest, verdict_color, add_verdict_legend, stat_scatter_kwargs,
        rdm_style, rdm_label, rdm_scatter_kwargs, rdm_tick_label, rdm_legend_order,
    )
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colorblind-safe categorical hues already validated in this project (see
# analysis/_retired/pilot/figures.py's former _VERSION_PALETTE comment: "blue/orange/
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
    lets the same function cover both analysis/_retired/pilot/figures.py's former
    _add_violin_dots_panel (dots = per-subject mean +/- SD, aggregated from a
    finer-grained trial-level violin) and
    analysis/s1_pre_vs_post_shine/demographics_and_behavior.py's former inline violin+dots code
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
    show_legend: bool = False, show_participants: bool = True,
) -> None:
    """Mean +/- SE line over `x_col` (e.g. trial number, repeat index), with
    thin per-participant lines underneath (unless `show_participants` is
    False), one set of traces per key in `frames`. Unifies the identically-named _add_progression_panel() that
    used to live independently in both analysis/_retired/pilot/figures.py and
    analysis/s1_pre_vs_post_shine/demographics_and_behavior.py.

    `show_legend` is per-call (not inferred from `row`) so a caller building
    a multi-row figure can show the legend once, on whichever call it likes
    (e.g. only its first row), rather than this function guessing "row == 1".
    """
    labels = labels or {k: str(k) for k in frames}
    for key, df in frames.items():
        c = colors[key]

        if show_participants:
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


def add_split_violin(
    fig: go.Figure, left_values, right_values, *, left_key, right_key, colors: dict, labels: dict,
    position: float = 0.0, width: float = 0.9, row: int = 1, col_idx: int = 1, secondary_y: bool = False,
    left_dots=None, right_dots=None, show_participants: bool = True, show_legend: bool = False,
    box_visible: bool = False, seed: int = 0,
) -> None:
    """One vertical split violin centred on x = `position`: `left_values` fill
    the left half (coloured by `colors[left_key]`), `right_values` the right
    half (`colors[right_key]`). Optional `left_dots` / `right_dots` (e.g.
    per-subject values when the violins hold trial-level data) are drawn as
    jittered markers within the matching half, unless `show_participants` is
    False. `show_legend` adds the two halves to the legend (grouped by key, so
    other traces with the same legendgroup toggle together). `secondary_y`
    draws on the subplot's secondary y-axis (needs a secondary_y spec in
    make_subplots). Several calls with different `position`s put several split
    violins on one subplot; the caller sets the x tick labels."""
    extra = {"secondary_y": True} if secondary_y else {}
    rng = np.random.default_rng(seed)
    for key, values, dots, side, sign in (
        (left_key, left_values, left_dots, "negative", -1), (right_key, right_values, right_dots, "positive", 1),
    ):
        c = colors[key]
        fig.add_trace(go.Violin(
            x0=position, y=np.asarray(values, dtype=float), width=width, side=side, points=False,
            fillcolor=c["violin"], line_color=c["violin_line"], meanline_visible=True, box_visible=box_visible,
            name=labels[key], legendgroup=key, showlegend=show_legend,
        ), row=row, col=col_idx, **extra)
        if not show_participants or dots is None:
            continue
        dots = np.asarray(dots, dtype=float)
        fig.add_trace(go.Scatter(
            x=position + sign * rng.uniform(0.04, 0.3 * width, size=len(dots)), y=dots, mode="markers",
            marker={"color": c["dot"], "size": 6, "opacity": 0.8}, showlegend=False, legendgroup=key,
            name=f"{labels[key]} subjects", hovertemplate="%{y:.3f}<extra></extra>",
        ), row=row, col=col_idx, **extra)
    fig.update_layout(violinmode="overlay")


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
    per key in `panels`. Unifies analysis/_retired/pilot/figures.py's former
    fig_completion_status (bar_colors keyed by status) and
    analysis/s1_pre_vs_post_shine/demographics_and_behavior.py's former fig_retention_status
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


# ---------------------------------------------------------------------------
# Shared encodings for what is not an RDM: verdicts, statistics, nulls, aggregation methods
# ---------------------------------------------------------------------------
#
# These never borrow an RDM family hue (see the RDM design language at the end of this module), so a colour
# in any figure means one thing only.
#
#   equivalence verdict -> blue (equivalent within the bound) / red (not equivalent), in every forest plot
#   agreement statistic -> greyscale shade + line dash + marker symbol (STAT_STYLE)
#   permutation null    -> light grey fill; the observed value is a black line (NULL_COLOR, OBSERVED_COLOR)
#   aggregation method  -> bar pattern; the bar colour stays the behavioral cohort's (BEHAV_AGGREGATION_PATTERN)

EQUIVALENT_COLOR = "#2a78d6"
NOT_EQUIVALENT_COLOR = "#d62728"
VERDICT_LABEL = {True: "equivalent within the bound", False: "not equivalent"}

NULL_COLOR = "rgba(150,150,150,0.55)"
OBSERVED_COLOR = "#000000"

STAT_STYLE: dict[str, dict[str, str]] = {
    "kendall": {"color": "#222222", "dash": "solid", "symbol": "circle", "label": "Kendall tau"},
    "kendall_a": {"color": "#222222", "dash": "solid", "symbol": "circle", "label": "Kendall tau-a"},
    "spearman": {"color": "#666666", "dash": "dash", "symbol": "square", "label": "Spearman rho"},
    "pearson": {"color": "#666666", "dash": "dashdot", "symbol": "triangle-up", "label": "Pearson r"},
    "one_minus_m2": {"color": "#999999", "dash": "dot", "symbol": "diamond", "label": "1 - Procrustes m2"},
}

BEHAV_AGGREGATION_PATTERN = {"mean": "", "simple": "/", "evidence": "."}


def verdict_color(equivalent: bool) -> str:
    return EQUIVALENT_COLOR if equivalent else NOT_EQUIVALENT_COLOR


def add_verdict_legend(fig: go.Figure, *, row: int | None = None, col: int | None = None) -> None:
    """Two legend-only entries naming the verdict colours (no data drawn)."""
    where = {} if row is None else {"row": row, "col": col}
    for equivalent in (True, False):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", marker={"color": verdict_color(equivalent), "size": 10},
            name=VERDICT_LABEL[equivalent], legendgroup=f"verdict_{equivalent}",
        ), **where)


def stat_scatter_kwargs(stat: str, *, width: float = 2.5, size: int = 8) -> dict:
    """go.Scatter keyword arguments (name, line, marker) for one agreement statistic."""
    s = STAT_STYLE[stat]
    return {"name": s["label"], "line": {"color": s["color"], "dash": s["dash"], "width": width},
            "marker": {"color": s["color"], "symbol": s["symbol"], "size": size}}


def fig_equivalence_forest(results: dict[str, dict], *, x_title: str, title: str) -> go.Figure:
    """Forest plot of equivalence-test results (analysis.utils.equivalence's
    equivalence_test_*() dicts), one row per outcome: effect marker with its
    (1 - 2*alpha) CI, and the +/- bound drawn as two tick marks on the same row
    (bounds may differ per row). Rows whose CI lies inside the bound are blue,
    the rest red (the shared verdict colours, named in the legend)."""
    fig = go.Figure()
    names = list(results)
    for name in names:
        result = results[name]
        color = verdict_color(result["equivalent"])
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
    add_verdict_legend(fig)
    fig.add_vline(x=0, line={"color": "rgba(80,80,80,0.5)", "width": 1, "dash": "dot"})
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=names)
    fig.update_xaxes(title_text=x_title)
    fig.update_layout(
        template="plotly_white", title=title,
        height=140 + 55 * len(names), margin={"l": 160, "r": 40, "t": 70, "b": 60},
    )
    return fig


# ---------------------------------------------------------------------------
# RDM design language: an RDM's name fixes how it is drawn
# ---------------------------------------------------------------------------
#
# Three independent properties of an RDM name map to three independent visual channels, so a reader can
# decode any marker or line without a legend lookup:
#
#   family          (what is measured)         -> hue
#       km = the curated Kiani-Mur tree (the reference: near-black, dashed), wn = WordNet, clip = CLIP,
#       sens = pixels, behav = SpAM behavior. The four non-reference families take the four CVD-validated
#       hues (_VALIDATED_HUES above), so no unvalidated hue is introduced.
#   variant         (which stimuli / cohort)    -> marker symbol, and shade within the family hue
#       pre = circle (full hue), post = diamond (lighter), joint = star (full hue), none (sem_*) = square.
#   representation  (raw RDM or MDS embedding)  -> marker fill (and line dash)
#       raw = filled, mds = the open version of the same symbol; a low-d embedding is a lighter shade
#       than a high-d one (dash-dot vs dotted lines).
#
# Names follow the analysis/results/rdms/ stems: sem_km, sem_wn, clip_pre, sens_post, behav_joint,
# behav_joint_mds_low / behav_joint_mds_high, and optionally a dimensionality (behav_joint_mds9).
#
# Usage: fig.add_trace(go.Scatter(x=x, y=y, **rdm_scatter_kwargs("clip_pre")))

RDM_REFERENCE_COLOR = "#222222"
RDM_FAMILY_HUE: dict[str, str] = {
    "km": RDM_REFERENCE_COLOR,
    "wn": _VALIDATED_HUES[2],     # green
    "clip": _VALIDATED_HUES[0],   # blue
    "sens": _VALIDATED_HUES[1],   # orange
    "behav": _VALIDATED_HUES[3],  # purple
}
RDM_FAMILY_LABEL = {"km": "KM tree", "wn": "WordNet", "clip": "CLIP", "sens": "pixels", "behav": "behavior"}
RDM_VARIANT_SYMBOL = {None: "square", "pre": "circle", "post": "diamond", "joint": "star"}
RDM_VARIANT_LIGHTEN = {None: 0.0, "pre": 0.0, "post": 0.45, "joint": 0.0}

_RDM_FAMILY_ORDER = list(RDM_FAMILY_HUE)
_RDM_VARIANT_ORDER = [None, "pre", "post", "joint"]
_RDM_NAME = re.compile(r"^(?P<stem>sem_km|sem_wn|clip|sens|behav)(?:_(?P<variant>pre|post|joint))?"
                   r"(?:_(?P<repr>mds)(?P<ndim>\d+)?(?:_(?P<tier>low|high))?)?$")
_RDM_STEM_FAMILY = {"sem_km": "km", "sem_wn": "wn", "clip": "clip", "sens": "sens", "behav": "behav"}


@dataclass(frozen=True)
class RdmStyle:
    family: str
    variant: str | None
    representation: str
    ndim: int | None
    tier: str | None
    color: str
    symbol: str
    dash: str
    label: str


def _lighten(hex_color: str, amount: float) -> str:
    """Mix `hex_color` toward white by `amount` in [0, 1]."""
    rgb = [int(hex_color.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4)]
    mixed = [round(c + (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def parse_rdm_name(name: str) -> tuple[str, str | None, str, int | None]:
    """(family, variant, representation, ndim) of an RDM name; ValueError for an unknown name."""
    m = _RDM_NAME.match(name)
    if m is None:
        raise ValueError(f"not an RDM name this style knows: {name!r}")
    ndim = int(m["ndim"]) if m["ndim"] else None
    return _RDM_STEM_FAMILY[m["stem"]], m["variant"], "mds" if m["repr"] else "raw", ndim


def _rdm_tier(name: str) -> str | None:
    m = _RDM_NAME.match(name)
    return m["tier"] if m else None


def rdm_label(name: str) -> str:
    """Human-readable label, e.g. 'CLIP pre', 'behavior joint, MDS d=9'."""
    family, variant, representation, ndim = parse_rdm_name(name)
    text = " ".join(filter(None, [RDM_FAMILY_LABEL[family], variant]))
    if representation == "mds":
        tier = _rdm_tier(name)
        text += ", MDS" + (f" d={ndim}" if ndim else "") + (f" {tier}-d" if tier else "")
    return text


def rdm_style(name: str, *, lighten: float = 0.0) -> RdmStyle:
    """Full visual specification of RDM `name`. `lighten` adds extra whitening (e.g. to grade a sweep)."""
    family, variant, representation, ndim = parse_rdm_name(name)
    tier = _rdm_tier(name)
    tier_lighten = 0.4 if tier == "low" else 0.0
    color = _lighten(RDM_FAMILY_HUE[family], min(1.0, RDM_VARIANT_LIGHTEN[variant] + tier_lighten + lighten))
    symbol = RDM_VARIANT_SYMBOL[variant] + ("-open" if representation == "mds" else "")
    dash = "dash" if family == "km" else ("solid" if representation == "raw" else ("dashdot" if tier == "low" else "dot"))
    return RdmStyle(family, variant, representation, ndim, tier, color, symbol, dash, rdm_label(name))


def rdm_marker(name: str, *, size: int = 10, lighten: float = 0.0) -> dict:
    s = rdm_style(name, lighten=lighten)
    return {"color": s.color, "symbol": s.symbol, "size": size, "line": {"color": s.color, "width": 1.5}}


def rdm_line(name: str, *, width: float = 2.5, lighten: float = 0.0) -> dict:
    s = rdm_style(name, lighten=lighten)
    return {"color": s.color, "dash": s.dash, "width": width}


def rdm_scatter_kwargs(name: str, *, size: int = 10, width: float = 2.5, lighten: float = 0.0) -> dict:
    """Keyword arguments for go.Scatter: name, marker, line and a per-family legend group."""
    s = rdm_style(name, lighten=lighten)
    return {
        "name": s.label, "legendgroup": s.family,
        "marker": rdm_marker(name, size=size, lighten=lighten), "line": rdm_line(name, width=width, lighten=lighten),
    }


def rdm_legend_order(names: list[str]) -> list[str]:
    """`names` sorted by family (reference first), then variant, then raw before MDS, then ndim."""
    def key(name: str) -> tuple:
        family, variant, representation, ndim = parse_rdm_name(name)
        tier = {None: 0, "low": 1, "high": 2}[_rdm_tier(name)]
        return (_RDM_FAMILY_ORDER.index(family), _RDM_VARIANT_ORDER.index(variant), representation != "raw", tier, ndim or 0)
    return sorted(names, key=key)


def rdm_tick_label(name: str) -> str:
    """HTML tick text in the RDM's colour, for categorical axes (heatmaps, bar charts)."""
    s = rdm_style(name)
    return f'<span style="color:{s.color}">{s.label}</span>'
