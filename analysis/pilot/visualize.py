"""
Trial arrangement visualizations for SpAM pilot data.

Coordinates in final_locations / init_locations / moves are already
normalised to [0, 1] by the parser, so renderings are screen-independent.

Usage (from repo root):
    from analysis.pilot.parser import load_pilot_data
    from analysis.pilot.visualize import render_trial, plot_trial_grid

    data = load_pilot_data("data/pilot")
    df_s = data["trials"][data["trials"]["participant_id"] == pid]

    img = render_trial(df_s.iloc[0])          # PIL Image
    fig = plot_trial_grid(df_s)               # Plotly figure
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

# analysis/pilot/visualize.py → analysis/pilot/ → analysis/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

_BG_COLOUR = (250, 250, 250)   # near-white canvas background


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_trial(
    trial: pd.Series,
    output_width: int = 700,
    output_height: int = 530,
    thumbnail_px: int = 72,
) -> Image.Image:
    """
    Render a single trial's final arrangement as a PIL Image.

    Coordinates in *trial["final_locations"]* are expected to be in [0, 1]
    (as stored by the parser).  The rendered image has size
    (*output_width* × *output_height*) pixels regardless of the subject's
    original screen size.

    Parameters
    ----------
    trial:
        A single row from the trials DataFrame.
    output_width, output_height:
        Pixel dimensions of the rendered image.
    thumbnail_px:
        Each stimulus is resized to fit within a *thumbnail_px* × *thumbnail_px*
        bounding box before being pasted onto the canvas.
    """
    locs_raw = trial.get("final_locations", None)
    if pd.isna(locs_raw) or locs_raw == "":
        return Image.new("RGB", (output_width, output_height), _BG_COLOUR)

    locs = json.loads(locs_raw)
    canvas = Image.new("RGB", (output_width, output_height), _BG_COLOUR)

    for item in locs:
        img_path = _REPO_ROOT / item["src"].lstrip("./")
        try:
            img = Image.open(img_path).convert("RGBA")
        except (FileNotFoundError, OSError):
            continue

        img.thumbnail((thumbnail_px, thumbnail_px), Image.LANCZOS)

        # Map [0, 1] → output pixel coordinates, centred on the image
        cx = round(item["x"] * output_width)
        cy = round(item["y"] * output_height)
        paste_x = cx - img.width // 2
        paste_y = cy - img.height // 2

        # Composite RGBA (transparent background images) onto canvas
        canvas.paste(img, (paste_x, paste_y), mask=img.split()[3])

    return canvas


def plot_trial_grid(
    df_subject: pd.DataFrame,
    trials_per_row: int = 3,
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
