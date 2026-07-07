"""
Shared trial-arrangement rendering and visualization, used by both
analysis/pilot and analysis/prod.

Usage (from repo root):
    from analysis.utils.visualize_trials import render_trial, visualize_trials

    img = render_trial(df_subject.iloc[0])   # PIL Image
    fig = visualize_trials(df_subject)       # Plotly figure, all trials
    fig = visualize_trials(df_subject, only_repeats=True)  # test-retest pairs only
"""
from __future__ import annotations

import json
import math
import statistics
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

from analysis.utils.spearman_r import pair_spearman_r

# analysis/utils/visualize_trials.py → analysis/utils/ → analysis/ → repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

_BG_COLOUR = (250, 250, 250)   # near-white canvas background


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_trial(
    trial: pd.Series,
    output_width: int = 900,
    output_height: int = 680,
    thumbnail_px: int = 100,
) -> Image.Image:
    """
    Render a single trial's final arrangement as a PIL Image.

    Coordinates in *trial["final_locations"]* are expected to be in [0, 1]
    (screen-independent). The rendered image has size
    (*output_width* × *output_height*) pixels regardless of the subject's
    original screen size.

    Parameters
    ----------
    trial:
        A single row from a trials DataFrame (or any mapping with a
        "final_locations" key holding a JSON string of {"src", "x", "y"}
        items, x/y normalised to [0, 1]).
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

    # Add half-a-thumbnail of padding on every side so images placed at the
    # canvas edge aren't clipped when centred on their coordinates.
    pad = thumbnail_px // 2
    canvas = Image.new("RGB", (output_width + 2 * pad, output_height + 2 * pad), _BG_COLOUR)

    for item in locs:
        img_path = _REPO_ROOT / item["src"].lstrip("./")
        try:
            img = Image.open(img_path).convert("RGBA")
        except (FileNotFoundError, OSError):
            continue

        img.thumbnail((thumbnail_px, thumbnail_px), Image.LANCZOS)

        # Map [0, 1] → padded canvas pixel coordinates, centred on the image
        cx = round(item["x"] * output_width) + pad
        cy = round(item["y"] * output_height) + pad
        paste_x = cx - img.width // 2
        paste_y = cy - img.height // 2

        # Composite RGBA (transparent background images) onto canvas
        canvas.paste(img, (paste_x, paste_y), mask=img.split()[3])

    return canvas


def _pair_and_remaining_rows(df_subject: pd.DataFrame) -> tuple[list[pd.Series], list[pd.Series]]:
    """
    Split *df_subject* (sorted by trial_number) into:
      - pair_rows: [original, repeat, original, repeat, ...] for every
        test-retest repeat trial, ordered by the repeat's presentation order
        (ascending trial_number of the repeat).
      - remaining_rows: every other trial, in presentation order, excluding
        anything already included in pair_rows.

    Requires "is_trial_repeat" and "repeat_of_trial_number" columns (present
    on every trials DataFrame produced by analysis.utils.parser, defaulting
    to False/NaN on task versions without the repeat mechanism).
    """
    df_subject = df_subject.sort_values("trial_number")
    trials_by_number = {int(row["trial_number"]): row for _, row in df_subject.iterrows()}

    repeat_mask = df_subject["is_trial_repeat"].astype(bool) & df_subject["repeat_of_trial_number"].notna()
    repeat_rows = df_subject[repeat_mask]

    pair_rows: list[pd.Series] = []
    used_numbers: set[int] = set()
    for _, rep_row in repeat_rows.iterrows():
        orig_num = int(rep_row["repeat_of_trial_number"])
        orig_row = trials_by_number.get(orig_num)
        if orig_row is None:
            continue
        pair_rows.append(orig_row)
        pair_rows.append(rep_row)
        used_numbers.add(orig_num)
        used_numbers.add(int(rep_row["trial_number"]))

    remaining_rows = [
        row for _, row in df_subject.iterrows() if int(row["trial_number"]) not in used_numbers
    ]

    return pair_rows, remaining_rows


def visualize_trials(
    df_subject: pd.DataFrame,
    only_repeats: bool = False,
    trials_per_row: int = 2,
    output_width: int = 900,
    output_height: int = 680,
    thumbnail_px: int = 100,
) -> go.Figure:
    """
    Render trials for one subject as a Plotly grid figure.

    Every test-retest pair (an original trial and its verbatim repeat) is
    annotated with their Spearman R (computed over shared pairwise image
    distances), and the figure's subtitle reports the subject's median R
    across all their pairs.

    Parameters
    ----------
    df_subject:
        Rows from the trials DataFrame filtered to a single participant.
    only_repeats:
        If True, show only the test-retest pairs (original trial followed by
        its verbatim repeat, one pair per row), ordered by the repeat's
        presentation order. If no repeat trials are present (e.g.
        task_version < 3.0, which has no repeat mechanism), a warning is
        raised and all trials are shown instead, as if only_repeats=False.
        If False (default), test-retest pairs come first (in the same order
        as only_repeats=True), followed by the remaining trials in order of
        presentation -- so the first 6 subplots are identical between the
        two modes when 3 repeat pairs are present.
    trials_per_row:
        Number of trial panels per row (3–4 recommended for full sessions;
        2 keeps each test-retest pair on its own row, which is required for
        the per-pair R annotation to sit alongside its row).
    output_width, output_height, thumbnail_px:
        Passed through to :func:`render_trial`.
    """
    pair_rows, remaining_rows = _pair_and_remaining_rows(df_subject)
    pair_r_values = [
        pair_spearman_r(pair_rows[k], pair_rows[k + 1]) for k in range(0, len(pair_rows), 2)
    ]

    if only_repeats:
        if not pair_rows:
            warnings.warn(
                "only_repeats=True but no test-retest repeat trials were found "
                "(task_version < 3.0?); showing all trials instead.",
                UserWarning,
                stacklevel=2,
            )
            trials = remaining_rows
            active_r_values: list[float | None] = []
        else:
            trials = pair_rows
            active_r_values = pair_r_values
    else:
        trials = pair_rows + remaining_rows
        active_r_values = pair_r_values

    n = len(trials)
    if n == 0:
        return go.Figure()

    n_cols = min(trials_per_row, n)
    n_rows = math.ceil(n / n_cols)

    # render_trial() pads its canvas by thumbnail_px on every side, so the
    # actual rendered array is this size -- not output_width x output_height.
    img_w = output_width + thumbnail_px
    img_h = output_height + thumbnail_px

    # Fixed *absolute* pixel gaps between panels (independent of row/col
    # count), converted to the fractions make_subplots expects. Sizing each
    # subplot's pixel domain to exactly img_w x img_h (rather than deriving
    # it from a fractional spacing of the whole figure) keeps images at
    # their true rendered size with no aspect-ratio-driven letterboxing,
    # however many rows a session has.
    gap_x_px = 20
    gap_y_px = 50  # also doubles as headroom for the subplot title
    plot_w = n_cols * img_w + (n_cols - 1) * gap_x_px
    plot_h = n_rows * img_h + (n_rows - 1) * gap_y_px
    horizontal_spacing = gap_x_px / plot_w if n_cols > 1 else 0
    vertical_spacing = gap_y_px / plot_h if n_rows > 1 else 0

    # For each pair (positions 2k, 2k+1 in `trials`), annotate its R value
    # beside the row if both trials land in the same row, otherwise fall
    # back to appending it onto the repeat trial's subplot title.
    row_pair_r: dict[int, float] = {}
    fallback_title_r: dict[int, float] = {}
    for k, r_value in enumerate(active_r_values):
        if r_value is None:
            continue
        i_orig, i_repeat = 2 * k, 2 * k + 1
        row_orig = i_orig // n_cols + 1
        row_repeat = i_repeat // n_cols + 1
        if row_orig == row_repeat and row_orig not in row_pair_r:
            row_pair_r[row_orig] = r_value
        else:
            fallback_title_r[i_repeat] = r_value

    subplot_titles = []
    for i, row in enumerate(trials):
        title = f"Trial {row['trial_number']}"
        if i in fallback_title_r:
            title += f"  (R={fallback_title_r[i]:.3f})"
        subplot_titles.append(title)
    # Pad titles for empty cells
    subplot_titles += [""] * (n_rows * n_cols - n)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    for i, trial in enumerate(trials):
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

    for row_number, r_value in row_pair_r.items():
        y0, y1 = fig.get_subplot(row=row_number, col=1).yaxis.domain
        fig.add_annotation(
            x=1.0, xref="paper", xanchor="left",
            y=(y0 + y1) / 2, yref="paper", yanchor="middle",
            text=f"R = {r_value:.3f}",
            showarrow=False,
            font={"size": 14, "color": "#0c447c"},
            align="left",
        )

    margin_l, margin_t, margin_b = 10, 80, 10
    right_margin = 100 if row_pair_r else 10

    valid_r_values = [r for r in active_r_values if r is not None]
    title_text = f"Trial arrangements — participant {df_subject['participant_id'].iloc[0]}"
    if valid_r_values:
        title_text += f"<br>median test-retest R = {statistics.median(valid_r_values):.3f}"

    fig.update_layout(
        height=plot_h + margin_t + margin_b,
        width=plot_w + margin_l + right_margin,
        margin={"l": margin_l, "r": right_margin, "t": margin_t, "b": margin_b},
        title_text=title_text,
    )
    return fig
