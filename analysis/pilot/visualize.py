"""
Trial arrangement visualizations for SpAM pilot data.

Coordinates in final_locations / init_locations / moves are already
normalised to [0, 1] by the parser, so renderings are screen-independent.

Usage (from repo root):
    from analysis.pilot.parser import load_pilot_data
    from analysis.pilot.visualize import plot_trial_grid

    data = load_pilot_data("data/pilot")
    df_s = data["trials"][data["trials"]["participant_id"] == pid]

    fig = plot_trial_grid(df_s)               # Plotly figure
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.utils.render import render_trial

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_trial_grid(
    df_subject: pd.DataFrame,
    trials_per_row: int = 2,
    output_width: int = 700,
    output_height: int = 530,
    thumbnail_px: int = 72,
) -> go.Figure:
    """
    Render all trials for one subject as a Plotly grid figure.

    Parameters
    ----------
    df_subject:
        Rows from the trials DataFrame filtered to a single participant,
        sorted by trial_number.
    trials_per_row:
        Number of trial panels per row (3–4 recommended).
    output_width, output_height, thumbnail_px:
        Passed through to :func:`render_trial`.
    """
    trials = list(df_subject.sort_values("trial_number").iterrows())
    n = len(trials)
    if n == 0:
        return go.Figure()

    n_cols = min(trials_per_row, n)
    n_rows = math.ceil(n / n_cols)

    subplot_titles = [f"Trial {row['trial_number']}" for _, row in trials]
    # Pad titles for empty cells
    subplot_titles += [""] * (n_rows * n_cols - n)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.08,
    )

    for i, (_, trial) in enumerate(trials):
        r = i // n_cols + 1
        c = i % n_cols + 1

        img = render_trial(
            trial,
            output_width=output_width,
            output_height=output_height,
            thumbnail_px=thumbnail_px,
        )
        fig.add_trace(go.Image(z=np.array(img)), row=r, col=c)

        # Hide tick labels; keep axes so the image fills the subplot
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)

    panel_w = output_width + 20   # add a small margin per panel
    panel_h = output_height + 50  # add room for the subplot title

    fig.update_layout(
        height=n_rows * panel_h,
        width=n_cols * panel_w,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title_text=(
            f"Trial arrangements — participant {df_subject['participant_id'].iloc[0]}"
        ),
    )
    return fig
