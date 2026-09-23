"""Tests for analysis.utils.plotting: the categorical palette generator and
the shared panel/figure builders. Light structural checks only (trace counts,
colors, axis ranges) -- matching this repo's existing convention of not
deep-testing plotly figure internals (neither analysis/pilot/figures.py nor
analysis/behavioral/cohort_comparison.py have unit tests for their fig_*
functions; notebook execution is the check there)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.utils.plotting import (
    add_progression_panel, add_violin_dots_panel, fig_group_status_bar, role_palette,
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
