"""Tests for analysis.utils.plotting: the categorical palette generator and
the shared panel/figure builders. Light structural checks only (trace counts,
colors, axis ranges) -- matching this repo's existing convention of not
deep-testing plotly figure internals (neither analysis/pilot/figures.py nor
analysis/behavioral/demographics_and_behavior.py have unit tests for their fig_*
functions; notebook execution is the check there)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.utils.plotting import (
    add_progression_panel, add_split_violin, add_violin_dots_panel, fig_equivalence_forest, fig_group_status_bar,
    role_palette,
)

# ---------------------------------------------------------------------------
# role_palette
# ---------------------------------------------------------------------------

def test_role_palette_assigns_hues_in_order():
    palette = role_palette(["pre", "post"])
    assert palette["pre"]["main"] == "#2a78d6"
    assert palette["post"]["main"] == "#eb6834"


def test_role_palette_matches_original_version_palette_values():
    """Byte-identical to analysis/pilot/figures.py's former _VERSION_PALETTE[1.0]."""
    palette = role_palette(["v1"])
    assert palette["v1"] == {
        "main": "#2a78d6",
        "violin": "rgba(42,120,214,0.35)",
        "violin_line": "rgba(42,120,214,0.9)",
        "subj": "rgba(42,120,214,0.2)",
        "dot": "rgba(42,120,214,0.9)",
        "dot_err": "rgba(42,120,214,0.6)",
    }


def test_role_palette_raises_when_more_keys_than_hues():
    with pytest.raises(ValueError, match="validated hues"):
        role_palette(["a", "b", "c", "d", "e"])


def test_role_palette_all_four_validated_hues_ok():
    palette = role_palette(["a", "b", "c", "d"])
    assert len(palette) == 4


# ---------------------------------------------------------------------------
# add_violin_dots_panel
# ---------------------------------------------------------------------------

def test_add_violin_dots_panel_trace_count_and_error_bars():
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


def test_add_violin_dots_panel_legend_only_on_violin():
    fig = make_subplots(rows=1, cols=1)
    colors = role_palette(["a"])
    add_violin_dots_panel(
        fig, {"a": np.array([1.0, 2.0])}, {"a": pd.DataFrame({"value": [1.0, 2.0]})},
        row=1, colors=colors,
    )
    violin = next(t for t in fig.data if isinstance(t, go.Violin))
    dots = next(t for t in fig.data if isinstance(t, go.Scatter))
    assert violin.showlegend is True
    assert dots.showlegend is False


# ---------------------------------------------------------------------------
# add_progression_panel
# ---------------------------------------------------------------------------

def _synthetic_progression_frame(n_participants: int, n_trials: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for p in range(n_participants):
        for t in range(1, n_trials + 1):
            rows.append({"participant_id": f"p{p}", "trial_id": t, "value": rng.random()})
    return pd.DataFrame(rows)


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


def test_add_progression_panel_show_legend_is_per_call():
    fig = make_subplots(rows=1, cols=1)
    colors = role_palette(["a"])
    frame = {"a": _synthetic_progression_frame(2, 3)}
    add_progression_panel(fig, frame, "value", "trial_id", row=1, colors=colors, show_legend=True)
    mean_line = next(t for t in fig.data if t.mode == "lines+markers")
    assert mean_line.showlegend is True

    fig2 = make_subplots(rows=1, cols=1)
    add_progression_panel(fig2, frame, "value", "trial_id", row=1, colors=colors, show_legend=False)
    mean_line2 = next(t for t in fig2.data if t.mode == "lines+markers")
    assert mean_line2.showlegend is False


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


def test_fig_group_status_bar_color_by_status_vs_by_panel():
    panels = {"pre": pd.DataFrame({"status": ["full data", "screened out"]})}

    fig_by_status = fig_group_status_bar(
        panels, "status", status_order=_STATUS_ORDER, status_labels=_STATUS_LABELS,
        bar_colors={"full data": "#111111", "screened out": "#222222"},
        panel_titles={"pre": "Pre"},
    )
    assert list(fig_by_status.data[0].marker.color) == ["#111111", "#222222"]

    fig_by_panel = fig_group_status_bar(
        panels, "status", status_order=_STATUS_ORDER, status_labels=_STATUS_LABELS,
        bar_colors={"pre": "#333333"}, panel_titles={"pre": "Pre"},
    )
    assert list(fig_by_panel.data[0].marker.color) == ["#333333", "#333333"]


# ---------------------------------------------------------------------------
# fig_equivalence_forest
# ---------------------------------------------------------------------------

def test_fig_equivalence_forest_colors_rows_by_verdict_and_draws_bounds():
    results = {
        "Age": {"effect": 0.02, "ci_low": -0.10, "ci_high": 0.14, "bound": 0.3, "equivalent": True},
        "Moves": {"effect": 0.25, "ci_low": 0.05, "ci_high": 0.45, "bound": 0.3, "equivalent": False},
    }
    fig = fig_equivalence_forest(results, x_title="Rank-biserial r", title="t")
    assert len(fig.data) == 4  # effect+CI trace and bound-tick trace per outcome
    assert fig.data[0].marker.color == "#2a78d6"
    assert fig.data[2].marker.color == "#eb6834"
    assert list(fig.data[1].x) == [-0.3, 0.3]
