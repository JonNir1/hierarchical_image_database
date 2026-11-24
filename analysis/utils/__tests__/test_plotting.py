"""Tests for analysis.utils.plotting: the categorical palette generator and
the shared panel/figure builders, and the RDM design language. Light structural checks only (trace counts,
colors, axis ranges) -- matching this repo's existing convention of not
deep-testing plotly figure internals (neither analysis/_retired/pilot/figures.py nor
analysis/s1_pre_vs_post_shine/demographics_and_behavior.py have unit tests for their fig_*
functions; notebook execution is the check there)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.utils.plotting import (
    add_progression_panel, add_split_violin, add_violin_dots_panel, fig_equivalence_forest, fig_group_status_bar,
    rdm_style, role_palette,
)

# ---------------------------------------------------------------------------
# role_palette
# ---------------------------------------------------------------------------

def test_role_palette_assigns_hues_in_order_and_matches_original_version_palette():
    palette = role_palette(["pre", "post"])
    assert palette["pre"]["main"] == "#2a78d6"
    assert palette["post"]["main"] == "#eb6834"
    # Byte-identical to analysis/_retired/pilot/figures.py's former _VERSION_PALETTE[1.0].
    assert palette["pre"] == {
        "main": "#2a78d6",
        "violin": "rgba(42,120,214,0.35)",
        "violin_line": "rgba(42,120,214,0.9)",
        "subj": "rgba(42,120,214,0.2)",
        "dot": "rgba(42,120,214,0.9)",
        "dot_err": "rgba(42,120,214,0.6)",
    }


def test_role_palette_supports_four_hues_and_raises_beyond():
    assert len(role_palette(["a", "b", "c", "d"])) == 4
    with pytest.raises(ValueError, match="validated hues"):
        role_palette(["a", "b", "c", "d", "e"])


# ---------------------------------------------------------------------------
# add_violin_dots_panel
# ---------------------------------------------------------------------------

def test_add_violin_dots_panel_trace_count_error_bars_and_legend_only_on_violin():
    fig = make_subplots(rows=1, cols=1)
    colors = role_palette(["pre", "post"])
    violins = {"pre": np.array([1.0, 2.0, 3.0]), "post": np.array([4.0, 5.0])}
    dots = {
        "pre": pd.DataFrame({"value": [1.0, 2.0, 3.0], "error": [0.1, 0.2, 0.3]}),
        "post": pd.DataFrame({"value": [4.0, 5.0]}),  # no error column
    }
    add_violin_dots_panel(fig, violins, dots, row=1, colors=colors, labels={"pre": "Pre", "post": "Post"})

    assert len(fig.data) == 4  # violin + dots, x2 keys
    violin_traces = [t for t in fig.data if isinstance(t, go.Violin)]
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(violin_traces) == 2
    assert len(scatter_traces) == 2
    pre_dots = next(t for t in scatter_traces if t.legendgroup == "pre")
    post_dots = next(t for t in scatter_traces if t.legendgroup == "post")
    assert pre_dots.error_x.array is not None
    assert post_dots.error_x.array is None  # plotly always materializes an ErrorX() object; array is what matters
    assert all(t.showlegend is True for t in violin_traces)
    assert all(t.showlegend is False for t in scatter_traces)


# ---------------------------------------------------------------------------
# add_progression_panel
# ---------------------------------------------------------------------------

def _synthetic_progression_frame(n_participants: int, n_trials: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "participant_id": np.repeat([f"p{p}" for p in range(n_participants)], n_trials),
        "trial_id": np.tile(np.arange(1, n_trials + 1), n_participants),
        "value": rng.random(n_participants * n_trials),
    })


def test_add_progression_panel_trace_count():
    fig = make_subplots(rows=1, cols=1)
    colors = role_palette(["pre", "post"])
    frames = {
        "pre": _synthetic_progression_frame(3, 4),
        "post": _synthetic_progression_frame(2, 4),
    }
    add_progression_panel(fig, frames, "value", "trial_id", row=1, colors=colors)

    # per key: n_participants thin lines + 1 band + 1 mean line
    expected = (3 + 2) + 2 * 2
    assert len(fig.data) == expected

    fig_no_subjects = make_subplots(rows=1, cols=1)
    add_progression_panel(fig_no_subjects, frames, "value", "trial_id", row=1, colors=colors, show_participants=False)
    assert len(fig_no_subjects.data) == 2 * 2  # band + mean line per key only


@pytest.mark.parametrize("show_legend", [True, False])
def test_add_progression_panel_show_legend_is_per_call(show_legend):
    fig = make_subplots(rows=1, cols=1)
    colors = role_palette(["a"])
    frame = {"a": _synthetic_progression_frame(2, 3)}
    add_progression_panel(fig, frame, "value", "trial_id", row=1, colors=colors, show_legend=show_legend)
    mean_line = next(t for t in fig.data if t.mode == "lines+markers")
    assert mean_line.showlegend is show_legend


# ---------------------------------------------------------------------------
# add_split_violin
# ---------------------------------------------------------------------------

def test_add_split_violin_sides_dots_and_secondary_axis():
    colors = role_palette(["pre", "post"])
    labels = {"pre": "Pre", "post": "Post"}
    values = np.linspace(0, 1, 20)
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    add_split_violin(
        fig, values, values + 1, left_key="pre", right_key="post", colors=colors, labels=labels,
        left_dots=values[:5], right_dots=values[:4], secondary_y=True,
    )
    violins = [t for t in fig.data if t.type == "violin"]
    assert [v.side for v in violins] == ["negative", "positive"]
    assert [v.fillcolor for v in violins] == [colors["pre"]["violin"], colors["post"]["violin"]]
    dots = [t for t in fig.data if t.type == "scatter"]
    assert [len(d.x) for d in dots] == [5, 4]
    assert all(x < 0 for x in dots[0].x) and all(x > 0 for x in dots[1].x)
    assert {t.yaxis for t in fig.data} == {"y2"}

    fig_no_dots = make_subplots(rows=1, cols=1)
    add_split_violin(
        fig_no_dots, values, values, left_key="pre", right_key="post", colors=colors, labels=labels,
        left_dots=values, right_dots=values, show_participants=False,
    )
    assert [t.type for t in fig_no_dots.data] == ["violin", "violin"]


# ---------------------------------------------------------------------------
# fig_group_status_bar
# ---------------------------------------------------------------------------

_STATUS_ORDER = ["full data", "screened out"]
_STATUS_LABELS = {"full data": "Full data", "screened out": "Screened out"}


def test_fig_group_status_bar_percentages_sum_to_100():
    panels = {
        "pre": pd.DataFrame({"status": ["full data"] * 8 + ["screened out"] * 2}),
        "post": pd.DataFrame({"status": ["full data"] * 5 + ["screened out"] * 5}),
    }
    bar_colors_by_status = {"full data": "#2ecc71", "screened out": "#f39c12"}
    fig = fig_group_status_bar(
        panels, "status", status_order=_STATUS_ORDER, status_labels=_STATUS_LABELS,
        bar_colors=bar_colors_by_status, panel_titles={"pre": "Pre", "post": "Post"},
    )
    assert len(fig.data) == 2  # one Bar trace per panel
    for trace in fig.data:
        assert sum(trace.x) == pytest.approx(100.0)


@pytest.mark.parametrize("bar_colors, expected", [
    ({"full data": "#111111", "screened out": "#222222"}, ["#111111", "#222222"]),  # keyed by status
    ({"pre": "#333333"}, ["#333333", "#333333"]),  # keyed by panel
])
def test_fig_group_status_bar_color_by_status_vs_by_panel(bar_colors, expected):
    panels = {"pre": pd.DataFrame({"status": ["full data", "screened out"]})}
    fig = fig_group_status_bar(
        panels, "status", status_order=_STATUS_ORDER, status_labels=_STATUS_LABELS,
        bar_colors=bar_colors, panel_titles={"pre": "Pre"},
    )
    assert list(fig.data[0].marker.color) == expected


# ---------------------------------------------------------------------------
# fig_equivalence_forest
# ---------------------------------------------------------------------------

def test_fig_equivalence_forest_colors_rows_by_verdict_and_draws_bounds():
    results = {
        "Age": {"effect": 0.02, "ci_low": -0.10, "ci_high": 0.14, "bound": 0.3, "equivalent": True},
        "Moves": {"effect": 0.25, "ci_low": 0.05, "ci_high": 0.45, "bound": 0.3, "equivalent": False},
    }
    fig = fig_equivalence_forest(results, x_title="Rank-biserial r", title="t")
    assert len(fig.data) == 6  # effect+CI and bound-tick trace per outcome, then two legend-only verdict entries
    assert fig.data[0].marker.color == "#2a78d6"  # equivalent: blue
    assert fig.data[2].marker.color == "#d62728"  # not equivalent: red
    assert list(fig.data[1].x) == [-0.3, 0.3]


# ---------------------------------------------------------------------------
# RDM design language
# ---------------------------------------------------------------------------

_RDM_NAMES = ["sem_km", "sem_wn", "clip_pre", "clip_post", "sens_pre", "sens_post",
              "behav_joint", "behav_joint_mds_low", "behav_joint_mds_high"]


def test_rdm_channels_are_independent_and_unambiguous():
    styles = {n: rdm_style(n) for n in _RDM_NAMES}
    assert len({(s.color, s.symbol) for s in styles.values()}) == len(_RDM_NAMES)
    for family in ("clip", "sens"):
        pre, post = styles[f"{family}_pre"], styles[f"{family}_post"]
        assert pre.family == post.family and pre.color != post.color
        assert (pre.symbol, post.symbol) == ("circle", "diamond")
    raw, low, high = styles["behav_joint"], styles["behav_joint_mds_low"], styles["behav_joint_mds_high"]
    assert raw.color == high.color and high.symbol == low.symbol == raw.symbol + "-open"
    assert low.color != high.color
